"""PII span detection on short texts (OCR lines, speech transcripts) with a
LOCAL LLM (Ollama) plus a regex safety net, and same-type surrogate
generation for `transform` mode.

The prompt/JSON schema follows systems/llm_openai_compatible.py from the text
benchmark (LLM emits {"spans":[{text,label}]}; offsets are recovered by
substring search because LLMs cannot count characters reliably).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

SYSTEM_PROMPT = """You are a de-identification engine for text extracted from video
(OCR of screens, documents, badges, whiteboards, monitors, or a speech transcript).
Identify every span that is personally identifying information: names of people,
dates tied to a person, street addresses / locations smaller than a state, phone
numbers, fax numbers, email, SSN, MRN, account numbers, licence / plate numbers,
other personal ID numbers, URLs, IP addresses.
Do NOT flag hashtags, slogans, brand/organization/product/shop/business names,
public signage words, field labels (e.g. "Name:", "Phone:"), or generic topics.

Return ONLY a JSON object {"spans": [{"text": ..., "label": ...}]} where label is
one of NAME, DATE, LOCATION, PHONE, EMAIL, ID, AGE, OTHER and text is copied
exactly from the input. Return {"spans": []} if there is none."""

LABELS = {"NAME", "DATE", "LOCATION", "PHONE", "EMAIL", "ID", "AGE", "OTHER"}


@dataclass
class Span:
    start: int
    end: int
    label: str
    text: str
    source: str = "llm"
    surrogate: str | None = None
    surrogate_source: str | None = None
    surrogate_note: str | None = None   # guardian-replace: why the local fallback was used


# ----------------------------------------------------------------- regex net
_REGEXES = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("PHONE", re.compile(r"(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")),
    ("ID", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),                 # SSN
    ("ID", re.compile(r"\b(?:\d[\s-]?){6,}\d\b")),                 # long digit runs
    ("DATE", re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b")),
    ("OTHER", re.compile(r"\bhttps?://\S+|\bwww\.\S+")),
    ("OTHER", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),         # IPv4
]


def regex_spans(text: str) -> list[Span]:
    out: list[Span] = []
    for label, rx in _REGEXES:
        for m in rx.finditer(text):
            out.append(Span(m.start(), m.end(), label, m.group(0), "regex"))
    return out


# ------------------------------------------------------------------ LLM
def _robust_json(content: str) -> dict:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.M | re.S)
    m = re.search(r"\{.*\}", stripped, re.S)
    for c in (content, stripped, m.group(0) if m else None):
        if not c:
            continue
        try:
            return json.loads(c)
        except (json.JSONDecodeError, TypeError):
            pass
    spans = [{"text": m.group(1), "label": m.group(2)} for m in re.finditer(
        r'"text"\s*:\s*"([^"]+)"\s*,\s*"label"\s*:\s*"([A-Z]+)"', content)]
    return {"spans": spans}


def locate(pairs: list[tuple[str, str]], text: str, source: str) -> list[Span]:
    """Map (surface, label) pairs to char offsets in `text` (all occurrences)."""
    out = []
    for surface, label in pairs:
        surface = (surface or "").strip()
        if len(surface) < 2:
            continue
        label = label.upper() if label and label.upper() in LABELS else "OTHER"
        idx = text.find(surface)
        if idx < 0:  # tolerate OCR/LLM whitespace drift
            pat = r"\s*".join(re.escape(ch) for ch in surface.replace(" ", ""))
            for m in re.finditer(pat, text):
                out.append(Span(m.start(), m.end(), label, m.group(0), source))
            continue
        while idx >= 0:
            out.append(Span(idx, idx + len(surface), label, surface, source))
            idx = text.find(surface, idx + 1)
    return out


def merge_spans(spans: list[Span], text: str | None = None) -> list[Span]:
    """Union of overlapping spans. With `text`, the merged span text is re-cut
    from it and the detector sources are joined (e.g. "llm+guardian")."""
    spans = sorted(spans, key=lambda s: (s.start, -s.end))
    out: list[Span] = []
    for s in spans:
        if out and s.start < out[-1].end:
            p = out[-1]
            src = p.source if s.source in p.source.split("+") else f"{p.source}+{s.source}"
            end = max(p.end, s.end)
            out[-1] = Span(p.start, end, p.label,
                           text[p.start:end] if text is not None else p.text, src)
            continue
        out.append(Span(s.start, s.end, s.label, s.text, s.source))
    return out


class LocalLLMPII:
    """PII detector backed by a local Ollama model (no cloud calls) + regex net."""

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434",
                 use_llm: bool = True, use_regex: bool = True, timeout: float = 180.0):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.use_llm, self.use_regex = use_llm, use_regex
        self._cache: dict[str, list[tuple[str, str]]] = {}
        self._line_cache: dict[str, list[Span]] = {}
        self.n_calls = 0
        self.llm_seconds = 0.0
        self.errors = 0

    def _llm(self, text: str) -> list[tuple[str, str]]:
        if text in self._cache:
            return self._cache[text]
        t0 = time.time()
        for attempt in range(3):  # shared Ollama server: retry transient 5xx
            r = requests.post(f"{self.base_url}/api/chat", timeout=self.timeout, json={
                "model": self.model, "stream": False, "format": "json", "think": False,
                "options": {"temperature": 0},
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": text}]})
            if r.status_code < 500:
                break
            time.sleep(2 * (attempt + 1))
        r.raise_for_status()
        self.n_calls += 1
        self.llm_seconds += time.time() - t0
        data = _robust_json(r.json().get("message", {}).get("content", ""))
        raw = data.get("spans", []) if isinstance(data, dict) else []
        pairs = [(s.get("text", ""), s.get("label", "OTHER")) for s in raw if isinstance(s, dict)]
        self._cache[text] = pairs
        return pairs

    def _safe_llm(self, text: str) -> list[tuple[str, str]]:
        try:
            return self._llm(text)
        except Exception as e:  # LLM down -> regex only, but say so
            self.errors += 1
            print(f"[pii] local LLM call failed ({e}); regex-only for this text")
            return []

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        if self.use_llm and text.strip():
            spans += locate(self._safe_llm(text), text, "llm")
        if self.use_regex:
            spans += regex_spans(text)
        return merge_spans(spans, text)

    def detect_many(self, lines: list[str]) -> list[list[Span]]:
        """Detect on several lines with ONE LLM call (unseen lines joined by
        newlines); spans are re-located per line; results cached per line."""
        todo = [l for l in dict.fromkeys(lines) if l.strip() and l not in self._line_cache]
        if todo:
            pairs = self._safe_llm("\n".join(todo)) if self.use_llm else []
            for l in todo:
                self._line_cache[l] = locate(pairs, l, "llm")
        out = []
        for l in lines:
            s = list(self._line_cache.get(l, []))
            if self.use_regex:
                s += regex_spans(l)
            out.append(merge_spans(s, l))
        return out


# ------------------------------------------------------------- surrogates
_FIRST = ["Daniel", "Priya", "Tomas", "Aisha", "Kenji", "Laura", "Mateo", "Ingrid",
          "Samuel", "Nadia", "Oliver", "Mei", "Hugo", "Leila", "Victor", "Clara"]
_LAST = ["Brennan", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Castillo", "Novak",
         "Whitfield", "Haddad", "Petrov", "Sorensen", "Delgado", "Achebe", "Kowalski"]
_PLACES = ["Riverside", "Maple Grove", "Eastbrook", "Harbor Point", "Cedar Hill"]
_NUMWORDS = "zero one two three four five six seven eight nine".split()


def _rng(key: str) -> random.Random:
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))


def surrogate(label: str, original: str, salt: str = "custodian") -> str:
    """Deterministic, format-preserving, same-type surrogate: the same original
    value maps to the same surrogate across frames (and across OCR and audio)."""
    rng = _rng(salt + "|" + label + "|" + re.sub(r"\W", "", original.lower()))
    if label == "NAME":
        return rng.choice(_FIRST) if len(original.split()) <= 1 else \
            f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
    if label == "LOCATION":
        return rng.choice(_PLACES)
    if label == "EMAIL":
        return f"{rng.choice(_FIRST).lower()}.{rng.choice(_LAST).lower()}@example.org"
    s = re.sub(r"\d", lambda _m: str(rng.randrange(10)), original)
    s = re.sub(r"\b(" + "|".join(_NUMWORDS) + r")\b", lambda _m: rng.choice(_NUMWORDS), s,
               flags=re.I)
    if s == original:
        s = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") if c.isalpha() else c for c in original)
    return s


def redact_preview(text: str) -> str:
    """For audit logs: never write raw PII; keep first char of each token."""
    return " ".join(t[0] + "*" * (len(t) - 1) if t else t for t in text.split(" "))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ===================================================== PII engine + backends
def value_key(text: str) -> str:
    """Format-free identity of a PII value: '(415) 555-0199' == '415-555-0199'."""
    return re.sub(r"[\W_]", "", text.lower())


def _reformat(original: str, surr: str) -> str:
    """Carry a registered surrogate over to another surface format of the same
    value (digits of `surr` poured into the digit slots of `original`)."""
    od, sd = re.findall(r"\d", original), re.findall(r"\d", surr)
    if od and len(od) == len(sd) and len(od) >= 0.6 * len(value_key(original)):
        it = iter(sd)
        return re.sub(r"\d", lambda _m: next(it), original)
    return surr


def _guess_label(text: str) -> str:
    if "@" in text:
        return "EMAIL"
    if re.fullmatch(r"\s*" + _REGEXES[1][1].pattern + r"\s*", text):
        return "PHONE"
    if _REGEXES[4][1].search(text):
        return "DATE"
    digits = sum(c.isdigit() for c in text)
    if digits >= max(2, 0.5 * len(value_key(text))):
        return "ID"
    return "NAME"


def _leaks(original: str, surr: str) -> bool:
    """True if the surrogate repeats an identifying token of the original
    (Guardian Layer sometimes leaves values verbatim or half-replaced)."""
    if not surr.strip() or value_key(surr) == value_key(original):
        return True
    if _trivial_variant(original, surr):
        return True
    ot = {t for t in re.findall(r"[A-Za-z]{3,}|\d{3,}", original.lower())}
    st = {t for t in re.findall(r"[A-Za-z]{3,}|\d{3,}", surr.lower())}
    return bool(ot & st)


def _trivial_variant(a: str, b: str) -> bool:
    """'Flood' -> 'Floods', 'Phone' -> 'Phones': an inflection, not a surrogate."""
    import difflib
    a, b = a.strip().lower(), b.strip().lower()
    if not a or not b or any(c.isdigit() for c in a):
        return False
    return a == b or b.startswith(a) or a.startswith(b) or \
        difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def _case_like(original: str, cand: str) -> str:
    """Copy the letter case pattern of `original` onto `cand` (UPPER / lower / Title)."""
    letters = "".join(c for c in original if c.isalpha())
    if not letters:
        return cand
    if letters.isupper() and len(letters) > 1:
        return cand.upper()
    if letters.islower():
        return cand.lower()
    if all(w[:1].isupper() for w in re.findall(r"[A-Za-z]+", original)):
        return re.sub(r"[A-Za-z]+", lambda m: m.group(0)[:1].upper() + m.group(0)[1:].lower(), cand)
    return cand


_EMAIL_FULL = re.compile(r"[A-Za-z0-9][\w.+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


def complete_replacement(original: str, cand: str | None) -> tuple[str | None, str]:
    """Guardian Layer surrogate check for `guardian-replace`.

    Returns (value, "ok") when `cand` is a REAL, COMPLETE same-type replacement
    of `original` (re-shaped to the original's surface format), else
    (None, reason).  A partial edit (e.g. only the area code of a phone number
    changed) or a value that keeps an identifying token is rejected; the
    caller then uses the local same-type surrogate."""
    import difflib
    o, c = original.strip(), (cand or "").strip()
    if not c:
        return None, "empty"
    if "\n" in c or re.search(r"[\[\]<>{}*#]|(.)\1{3,}", c):
        return None, "masking_artefact"          # e.g. "[NAME]", "wxxxxxxxx"
    if value_key(c) == value_key(o):
        return None, "unchanged"
    od, cd = re.findall(r"\d", o), re.findall(r"\d", c)
    if "@" in o:                                  # e-mail: valid address, new local part
        if not _EMAIL_FULL.fullmatch(c):
            return None, "wrong_type"
        ol, cl = o.split("@")[0].lower(), c.split("@")[0].lower()
        if _leaks(ol, cl) or difflib.SequenceMatcher(None, ol, cl).ratio() >= 0.6:
            return None, "partial"
        return (c.lower() if o.split("@")[0].islower() else c), "ok"
    if len(od) >= 3:                              # phone / ID / date / account number
        if len(cd) != len(od):
            return None, "wrong_format"
        same = [a == b for a, b in zip(od, cd)]
        run = best = 0
        for s in same:
            run = run + 1 if s else 0
            best = max(best, run)
        if sum(not s for s in same) <= len(od) / 2 or best >= 3:
            return None, "partial"
        it = iter(cd)                             # pour the new digits into the original format
        return re.sub(r"\d", lambda _m: next(it), o), "ok"
    if od:                                        # short mixed code, e.g. "7E"
        oa, ca = re.sub(r"\W", "", o), re.sub(r"\W", "", c)
        if len(oa) != len(ca) or sum(a != b for a, b in zip(oa.lower(), ca.lower())) <= len(oa) / 2:
            return None, "partial"
        return c, "ok"
    # names / places / other words: letters only, same number of words, every word new
    if re.search(r"[^A-Za-z\s.'-]", c):
        return None, "wrong_type"
    ow, cw = re.findall(r"[A-Za-z]+", o), re.findall(r"[A-Za-z]+", c)
    if not cw or len(ow) != len(cw):
        return None, "wrong_format"
    if _leaks(o, c) or any(difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.6
                           for a, b in zip(ow, cw)):
        return None, "partial"
    return _case_like(o, c), "ok"


def _trim_span(sp: Span, text: str) -> Span | None:
    """Strip separators Guardian Layer includes at span edges ('|dana@...')."""
    s, e = sp.start, sp.end
    while s < e and not (text[s].isalnum() or text[s] in "(+"):
        s += 1
    while e > s and not (text[e - 1].isalnum() or text[e - 1] == ")"):
        e -= 1
    if e - s < 2:
        return None
    out = Span(s, e, sp.label, text[s:e], sp.source)
    out._g_out = getattr(sp, "_g_out", None)
    return out


_TOK = re.compile(r"\w+|\s+|[^\w\s]")


class _Align:
    """Token-level alignment original -> rewritten text (difflib). Used to
    recover WHAT Guardian Layer changed (it returns text, not spans)."""

    def __init__(self, a: str, b: str):
        import difflib
        self.a, self.b = a, b
        ta, tb = _TOK.findall(a), _TOK.findall(b)
        oa, ob = [0], [0]
        for t in ta:
            oa.append(oa[-1] + len(t))
        for t in tb:
            ob.append(ob[-1] + len(t))
        sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
        # (a_start, a_end, b_start, b_end, tag) in CHARACTER offsets
        self.ops = [(oa[i1], oa[i2], ob[j1], ob[j2], tag) for tag, i1, i2, j1, j2 in sm.get_opcodes()]

    def edits(self) -> list[tuple[int, int, int, int]]:
        """Changed regions (replace/delete; pure inserts are attached to the
        preceding char) that touch a word character."""
        out = []
        for a1, a2, b1, b2, tag in self.ops:
            if tag == "equal":
                continue
            if a1 == a2:   # insertion: zero-width, ignore for masking
                continue
            if re.search(r"\w", self.a[a1:a2]):
                out.append((a1, a2, b1, b2))
        return out

    def map_range(self, s: int, e: int) -> tuple[int, int]:
        """Output range corresponding to original chars [s, e)."""
        bs = be = None
        for a1, a2, b1, b2, tag in self.ops:
            if a2 <= s and not (a1 == a2 == s):
                continue
            if a1 >= e and not (a1 == a2 == e and tag != "equal"):
                break
            if tag == "equal":
                lo, hi = b1 + (max(s, a1) - a1), b1 + (min(e, a2) - a1)
            else:
                lo, hi = b1, b2
            bs = lo if bs is None else min(bs, lo)
            be = hi if be is None else max(be, hi)
        if bs is None:
            return 0, 0
        return bs, be


class GuardianClient:
    """Custodian Labs Guardian Layer (`custodian_labs.GuardianLayer`) with a
    per-text cache, a hard call budget, 429 back-off and NO logging of text or
    key.  The hosted API is used only for research on synthetic / public demo
    data; Guardian Layer can also be deployed on-prem (same SDK, set
    CUSTODIAN_MASKING_BASE_URL)."""

    USAGE = Path.home() / ".cache" / "custodian_video" / "guardian_usage.json"

    def __init__(self, max_calls: int = 500, total_cap: int | None = None,
                 cache_path: str | None = None, masking_type: str = "transform"):
        self.max_calls = max_calls
        self.total_cap = int(os.environ.get("CUSTODIAN_GUARDIAN_TOTAL_CAP", total_cap or 2000))
        self.masking_type = masking_type
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text())
            except Exception:
                self._cache = {}
        self._g = None
        self.stats = {"http_calls": 0, "ok": 0, "http_429": 0, "errors": 0, "cache_hits": 0,
                      "budget_skips": 0, "unique_texts": 0, "latency_s": 0.0}
        self.disabled_reason = None

    @staticmethod
    def _load_key():
        if os.environ.get("CUSTODIAN_SDK_API_KEY") or os.environ.get("AI_SDK_API_KEY"):
            return
        for d in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]:
            f = d / ".env"
            if f.is_file():
                for line in f.read_text().splitlines():
                    if line.startswith("CUSTODIAN_SDK_API_KEY="):
                        v = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if v:
                            os.environ["CUSTODIAN_SDK_API_KEY"] = v
                            return

    def _client(self):
        if self._g is None:
            self._load_key()
            from custodian_labs import GuardianLayer
            # SDK-internal retries off: every HTTP request is counted here
            self._g = GuardianLayer(max_retries=0, timeout=60.0)
        return self._g

    def _usage(self, add: int = 0) -> int:
        try:
            u = json.loads(self.USAGE.read_text()) if self.USAGE.exists() else {"calls": 0}
        except Exception:
            u = {"calls": 0}
        if add:
            u["calls"] = u.get("calls", 0) + add
            self.USAGE.parent.mkdir(parents=True, exist_ok=True)
            self.USAGE.write_text(json.dumps(u))
        return u.get("calls", 0)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.masking_type}|{text}".encode()).hexdigest()

    def rewrite(self, text: str) -> dict | None:
        """{'out': rewritten text, 'sensitive_words': [...], 'replacements': {...}}
        or None (budget exhausted / API failure -> caller falls back to local)."""
        k = self._key(text)
        if k in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[k]
        if self.disabled_reason:
            self.stats["budget_skips"] += 1
            return None
        res = None
        for attempt in range(6):
            if self.stats["http_calls"] >= self.max_calls or self._usage() >= self.total_cap:
                self.disabled_reason = "call budget exhausted"
                self.stats["budget_skips"] += 1
                print("[guardian] call budget exhausted; local detector only from here on")
                return None
            t0 = time.time()
            self.stats["http_calls"] += 1
            self._usage(1)
            try:
                r = self._client().deidentify_text_outputs(
                    text, masking_type=self.masking_type, pii_entities=["ALL"])
                self.stats["latency_s"] += time.time() - t0
                meta = r.meta or {}
                res = {"out": r.outputs[0].text if r.outputs else text,
                       "sensitive_words": [w for w in meta.get("sensitive_words", []) if isinstance(w, str)],
                       "replacements": {k2: v for k2, v in (meta.get("top1_replacements") or {}).items()
                                        if isinstance(k2, str) and isinstance(v, str)}}
                self.stats["ok"] += 1
                break
            except Exception as e:
                self.stats["latency_s"] += time.time() - t0
                name = type(e).__name__
                status = getattr(e, "status_code", None)
                if status == 429 or "429" in str(e)[:200] or "RateLimit" in name:
                    self.stats["http_429"] += 1
                    time.sleep(10.0 * (attempt + 1))
                    continue
                self.stats["errors"] += 1
                if status in (401, 403) or "Authentication" in name:
                    self.disabled_reason = f"auth error ({name})"
                    print(f"[guardian] {self.disabled_reason}; local detector only")
                    return None
                print(f"[guardian] call failed ({name}); retrying")  # no text, no key
                time.sleep(2.0 * (attempt + 1))
        if res is None:
            return None
        self._cache[k] = res
        self.stats["unique_texts"] += 1
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache))
        return res


class PIIEngine:
    """Span detector + surrogate source used by the OCR and audio channels.

    backend
      local    : local LLM (Ollama) + regex; local surrogates.        [default]
      guardian : Guardian Layer + regex safety net (no local LLM);
                 Guardian Layer surrogates.
      both     : Guardian Layer + local LLM + regex (union);
                 Guardian Layer surrogates.
      guardian-replace : Guardian Layer supplies the REPLACEMENT VALUES,  [recommended with Guardian]
                 the union of detectors decides WHERE (see _combine_replace):
                 spans = local LLM + regex, plus Guardian `sensitive_words`
                 that overlap a typed local span or contain a digit / '@';
                 value = Guardian Layer's surrogate for the span when it is a
                 real, complete, same-format replacement (complete_replacement),
                 else the local same-type surrogate.  Guardian edits outside
                 these spans are never applied.

    Guardian Layer returns rewritten text, not spans.  Its edits are recovered
    by token alignment, but an edit is only USED (as the surrogate of a span)
    when it overlaps a span flagged by the local detector or listed in
    Guardian Layer's own `sensitive_words`; other edits (rewrites of non-PII
    words) are ignored and counted.  Masking always uses the UNION of local and
    Guardian spans, so values Guardian Layer leaves verbatim are still masked;
    a Guardian surrogate that repeats a token of the original is replaced by
    the local same-type surrogate.

    Surrogates go through one registry per run keyed by the format-free value,
    so the same value gets the same surrogate on screen and in speech.
    """

    def __init__(self, backend: str = "local", llm_model: str = "qwen2.5:7b",
                 llm_url: str = "http://localhost:11434", use_llm: bool = True,
                 guardian: GuardianClient | None = None, strict_guardian: bool = True):
        self.backend = backend
        self.strict_guardian = strict_guardian
        self.local = LocalLLMPII(llm_model, llm_url, use_llm=use_llm and backend != "guardian")
        self.guardian = guardian if backend in ("guardian", "both", "guardian-replace") else None
        if self.guardian is None and backend != "local":
            self.guardian = GuardianClient()
        self.registry: dict[str, tuple[str, str]] = {}   # value_key -> (surrogate, source)
        self.gstats = {"guardian_flagged_raw": 0, "guardian_spans": 0, "local_spans": 0, "union_spans": 0,
                       "guardian_only_spans": 0, "local_only_spans(guardian_missed)": 0,
                       "guardian_edits": 0, "guardian_edits_used": 0,
                       "guardian_edits_ignored_non_pii": 0,
                       "guardian_sensitive_left_verbatim": 0,
                       "guardian_only_dropped_unchanged": 0, "guardian_only_dropped_inflection": 0,
                       "surrogates_from_guardian": 0, "surrogates_local_fallback": 0}
        if backend == "guardian-replace":
            self.gstats.update({"guardian_only_dropped_no_digit": 0, "guardian_extension_dropped": 0,
                                "candidates_rejected": {}, "fallback_reasons": {},
                                "registry_upgraded_to_guardian": 0})
        self.frozen = False   # set by the pipeline once on-screen values are rendered

    # --- passthrough stats used by the pipeline
    @property
    def n_calls(self):
        return self.local.n_calls

    @property
    def llm_seconds(self):
        return self.local.llm_seconds

    @property
    def use_llm(self):
        return self.local.use_llm

    def describe(self) -> str:
        loc = f"Ollama local LLM {self.local.model} + regex" if self.local.use_llm else "regex"
        if self.backend == "local":
            return loc
        if self.backend == "guardian-replace":
            return (f"spans: {loc} + Guardian Layer digit/@ spans; replacement values: Custodian Guardian "
                    f"Layer (custodian_labs SDK), local same-type surrogate as fallback")
        return f"Custodian Guardian Layer (custodian_labs SDK) + {loc} safety net (union)"

    def audit(self) -> dict:
        d = {"backend": self.backend, "local_llm_calls": self.local.n_calls,
             "local_llm_s": round(self.local.llm_seconds, 3), "local_llm_errors": self.local.errors}
        if self.guardian is not None:
            g = dict(self.guardian.stats)
            g["latency_s"] = round(g["latency_s"], 3)
            d["guardian_layer"] = {**g, **self.gstats,
                                   "disabled_reason": self.guardian.disabled_reason}
        return d

    # --- surrogates
    def surrogate_for(self, label: str, text: str, proposal: str | None = None,
                      proposal_source: str = "guardian") -> tuple[str, str]:
        k = value_key(text)
        if k in self.registry:
            s, src = self.registry[k][:2]
            return _reformat(text, s), src
        if proposal is not None and not _leaks(text, proposal):
            s, src = proposal.strip(), proposal_source
            if self.guardian is not None and self.backend != "guardian-replace":
                self.gstats["surrogates_from_guardian"] += 1
        else:
            s, src = self._email_from_name(text) or surrogate(label, text), "local"
            if self.guardian is not None and self.backend != "guardian-replace":
                self.gstats["surrogates_local_fallback"] += 1
        if k:
            self.registry[k] = (s, src)
        return s, src

    def _email_from_name(self, text: str) -> str | None:
        """'dana.whitfield@x.org' -> '<first>.<last>@example.org' built from the
        surrogate of the NAME 'Dana Whitfield' (registered here if not yet seen),
        so a caption's name and e-mail stay one consistent fake person."""
        m = re.fullmatch(r"\s*([A-Za-z]{2,})[._]([A-Za-z]{2,})@[\w.-]+\.[A-Za-z]{2,}\s*", text)
        if not m:
            return None
        name_s, _ = self.surrogate_for("NAME", f"{m[1].title()} {m[2].title()}")
        parts = re.findall(r"[A-Za-z]+", name_s.lower())
        if len(parts) != 2:
            return None
        s = f"{parts[0]}.{parts[1]}@example.org"
        return None if _leaks(f"{m[1]} {m[2]}", " ".join(parts)) else s

    # --- detection
    def _guardian_spans(self, text: str, res: dict | None):
        if res is None:
            return [], None
        sens = []
        # spans = Guardian Layer's own `sensitive_words` (NOT every rewritten word)
        words = list(dict.fromkeys(res["sensitive_words"]))
        al = _Align(text, res["out"])
        for sp in locate([(w, _guess_label(w)) for w in words], text, "guardian"):
            sp._g_out = None
            b1, b2 = al.map_range(sp.start, sp.end)
            if b2 > b1:
                sp._g_out = res["out"][b1:b2]
            sens.append(sp)
        return sens, al

    def _plausible(self, sp: Span, local_spans: list[Span]) -> bool:
        """Guardian-only spans (no local detector agrees) are kept only when
        Guardian Layer really replaced them, or when they contain digits.
        On the demo, Guardian flagged field labels and benign phrases
        ('Phone', 'Flood Clinic', 'a great day') and 'transformed' them into
        inflections ('Phones', 'Floods'); masking those destroys utility."""
        if not self.strict_guardian:
            return True
        if any(l.start < sp.end and sp.start < l.end for l in local_spans):
            return True
        if re.search(r"\d|@", sp.text):
            return True
        out = getattr(sp, "_g_out", None)
        if out is None or value_key(out) == value_key(sp.text):
            self.gstats["guardian_only_dropped_unchanged"] += 1
            return False
        if _trivial_variant(sp.text, out):
            self.gstats["guardian_only_dropped_inflection"] += 1
            return False
        return True

    # --- guardian-replace
    def _guardian_candidates(self, sp: Span, res: dict, al) -> list[str]:
        """Guardian Layer's proposals for span `sp`, best first: (1) its own
        top-1 replacement for exactly this value, (2) the value's words
        replaced one by one by its word-level replacements, (3) the aligned
        slice of its rewritten text."""
        out = []
        rep = res.get("replacements") or {}
        vk = value_key(sp.text)
        out += [v for k, v in rep.items() if value_key(k) == vk]
        words = re.findall(r"[A-Za-z]+", sp.text)
        if words and not re.search(r"\d", sp.text):
            low = {k.lower(): v for k, v in rep.items()}
            if all(w.lower() in low for w in words):
                out.append(re.sub(r"[A-Za-z]+", lambda m: low[m.group(0).lower()], sp.text))
        if al is not None:
            b1, b2 = al.map_range(sp.start, sp.end)
            if b2 > b1:
                out.append(res["out"][b1:b2])
        return list(dict.fromkeys(c.strip() for c in out if c and c.strip()))

    def _replace_value(self, label: str, text: str, cands: list[str],
                       why_none: str) -> tuple[str, str, str | None]:
        """(surrogate, source, fallback_note).  One registry per run keeps the
        same value identical on screen and in speech; a value first seen with
        the local fallback is upgraded to a later valid Guardian surrogate
        until the pipeline freezes the registry (before rendering)."""
        k = value_key(text)
        prev = self.registry.get(k)
        if prev and prev[1] == "guardian":
            return _reformat(text, prev[0]), "guardian", None
        if not (prev and self.frozen):
            reasons = []
            for c in cands:
                v, why = complete_replacement(text, c)
                if v is not None:
                    if prev:
                        self.gstats["registry_upgraded_to_guardian"] += 1
                    if k:
                        self.registry[k] = (v, "guardian")
                    return v, "guardian", None
                reasons.append(why)
                rj = self.gstats["candidates_rejected"]
                rj[why] = rj.get(why, 0) + 1
            why_none = "guardian_" + (reasons[0] if reasons else why_none)
        if prev:
            return _reformat(text, prev[0]), prev[1], prev[2] if len(prev) > 2 else None
        s = self._email_from_name(text) or surrogate(label, text)
        if k:
            self.registry[k] = (s, "local", why_none)
        return s, "local", why_none

    def _combine_replace(self, text: str, local_spans: list[Span], res: dict | None) -> list[Span]:
        ov = lambda a, b: a.start < b.end and b.start < a.end
        gsp, al = self._guardian_spans(text, res)
        self.gstats["guardian_flagged_raw"] += len(gsp)
        typed = [l for l in local_spans if l.label != "OTHER"]
        keep = []
        for g in gsp:
            g = _trim_span(g, text)
            if g is None:
                continue
            hits = [l for l in typed if ov(g, l)]
            if not hits:
                # Guardian-only: never ordinary words ('Phone', 'next week', 'VISITOR PASS')
                if re.search(r"[\d@]", g.text):
                    g.label = "ID" if _guess_label(g.text) == "NAME" else _guess_label(g.text)
                    keep.append(g)
                else:
                    self.gstats["guardian_only_dropped_no_digit"] += 1
                continue
            # overlaps a typed local span: extend only over word characters of the
            # same value (a surname next to a first name), never over a field label
            lo, hi = min(l.start for l in hits), max(l.end for l in hits)
            s, e = g.start, g.end
            if s < lo and not re.fullmatch(r"[\w.'@+-]+(?: [\w.'@+-]+)*\s?", text[s:lo]):
                s = lo
            if e > hi and not re.fullmatch(r"\s?[\w.'@+-]+(?: [\w.'@+-]+)*", text[hi:e]):
                e = hi
            if (s, e) != (g.start, g.end):
                self.gstats["guardian_extension_dropped"] += 1
            keep.append(Span(s, e, hits[0].label, text[s:e], "guardian"))
        self.gstats["local_spans"] += len(local_spans)
        self.gstats["guardian_spans"] += len(keep)
        self.gstats["guardian_only_spans"] += sum(not any(ov(g, l) for l in local_spans) for g in keep)
        if res is not None:
            self.gstats["local_only_spans(guardian_missed)"] += sum(
                not any(ov(l, g) for g in keep) for l in local_spans)
        merged = merge_spans(local_spans + keep, text)
        for sp in merged:
            lab = [l.label for l in local_spans if ov(l, sp)] or [g.label for g in keep if ov(g, sp)]
            sp.label = lab[0] if lab else _guess_label(sp.text)
            cands = self._guardian_candidates(sp, res, al) if res is not None else []
            sp.surrogate, sp.surrogate_source, sp.surrogate_note = self._replace_value(
                sp.label, sp.text, cands, "no_candidate" if res is not None else "unavailable")
            key = "surrogates_from_guardian" if sp.surrogate_source == "guardian" else "surrogates_local_fallback"
            self.gstats[key] += 1
            if sp.surrogate_note:
                fr = self.gstats["fallback_reasons"]
                fr[sp.surrogate_note] = fr.get(sp.surrogate_note, 0) + 1
        self.gstats["union_spans"] += len(merged)
        return merged

    def final_value(self, text: str) -> tuple[str, str, str | None] | None:
        """Registry value for `text` now (after possible upgrades), for the pipeline."""
        prev = self.registry.get(value_key(text))
        if not prev:
            return None
        return _reformat(text, prev[0]), prev[1], (prev[2] if len(prev) > 2 else None)

    def _combine(self, text: str, local_spans: list[Span], res: dict | None) -> list[Span]:
        if self.backend == "guardian-replace":
            return self._combine_replace(text, local_spans, res)
        if self.guardian is None:
            out = []
            for sp in local_spans:
                sp.surrogate, sp.surrogate_source = self.surrogate_for(sp.label, sp.text)
                out.append(sp)
            return out
        gsp, al = self._guardian_spans(text, res)
        ov = lambda a, b: a.start < b.end and b.start < a.end
        graw = gsp
        self.gstats["guardian_flagged_raw"] += len(graw)
        if al:
            for g in graw:   # Guardian flagged it but its rewrite left it unchanged
                if g._g_out is None or value_key(g._g_out) == value_key(g.text):
                    self.gstats["guardian_sensitive_left_verbatim"] += 1
        gsp = merge_spans([g for g in gsp if self._plausible(g, local_spans)], text)
        self.gstats["local_spans"] += len(local_spans)
        self.gstats["guardian_spans"] += len(gsp)
        self.gstats["guardian_only_spans"] += sum(not any(ov(g, l) for l in local_spans) for g in gsp)
        if res is not None:
            self.gstats["local_only_spans(guardian_missed)"] += sum(
                not any(ov(l, g) for g in gsp) for l in local_spans)
        edits = al.edits() if al else []
        self.gstats["guardian_edits"] += len(edits)
        flagged = local_spans + gsp
        used = [e for e in edits if any(f.start < e[1] and e[0] < f.end for f in flagged)]
        self.gstats["guardian_edits_used"] += len(used)
        self.gstats["guardian_edits_ignored_non_pii"] += len(edits) - len(used)
        # label: prefer the local detector's (typed) label
        merged = merge_spans(local_spans + gsp, text)
        for sp in merged:
            lab = [l.label for l in local_spans if ov(l, sp)]
            sp.label = lab[0] if lab else _guess_label(sp.text)
            proposal = None
            if al and any(sp.start < e[1] and e[0] < sp.end for e in used):
                b1, b2 = al.map_range(sp.start, sp.end)
                proposal = res["out"][b1:b2] if b2 > b1 else None
            sp.surrogate, sp.surrogate_source = self.surrogate_for(sp.label, sp.text, proposal)
        self.gstats["union_spans"] += len(merged)
        return merged

    def detect(self, text: str) -> list[Span]:
        local = self.local.detect(text)
        res = self.guardian.rewrite(text) if (self.guardian and text.strip()) else None
        return self._combine(text, local, res)

    def detect_many(self, lines: list[str]) -> list[list[Span]]:
        local = self.local.detect_many(lines)
        if self.guardian is None:
            return [self._combine(l, s, None) for l, s in zip(lines, local)]
        # one Guardian call per OCR frame: unseen lines joined by newlines, the
        # rewrite is split back per line through the alignment
        if not hasattr(self, "_gline"):
            self._gline: dict[str, dict | None] = {}
        todo = [l for l in dict.fromkeys(lines) if l.strip() and l not in self._gline]
        if todo:
            joined = "\n".join(todo)
            res = self.guardian.rewrite(joined)
            if res is None:
                for l in todo:
                    self._gline[l] = None
            else:
                al = _Align(joined, res["out"])
                pos = 0
                for l in todo:
                    b1, b2 = al.map_range(pos, pos + len(l))
                    out_l = res["out"][b1:b2].strip("\n") if b2 > b1 else l
                    sw = [w for w in res["sensitive_words"] if w in l]
                    self._gline[l] = {"out": out_l, "sensitive_words": sw,
                                      "replacements": {k: v for k, v in res["replacements"].items() if k in l}}
                    pos += len(l) + 1
        return [self._combine(l, s, self._gline.get(l)) for l, s in zip(lines, local)]
