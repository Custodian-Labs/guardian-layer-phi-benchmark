#!/usr/bin/env python3
"""C1 (redaction floor) and C2 (open Faker surrogate) corpora for the
NON-MEDICAL domain benchmarks (finance_gretel, legal_tab, enterprise_gretel).

Generalises scripts/build_c1c2_corpora.py (reuses its build_redact/build_faker)
to the setting where no Custodian transform exists yet: every gold span is a
candidate for substitution, and the open-surrogate arm must stand on its own.
Surrogates are therefore *domain-aware and format-preserving* (same type AND
same surface shape, e.g. a Luhn-valid card for a card number, a valid IBAN for
an IBAN, "12345/67" for an ECHR application number "36110/97", "Mr <name>" for
"Mr Galip Yalman", the same date format for a date). One surrogate per distinct
original string per document (consistent substitution, as a de-id tool would do).

A span is *masked* iff its surrogate differs from the original. The only spans
left unmasked are TAB `MISC` / `QUANTITY` mentions with neither digits nor number words and
TAB `DEM` mentions that are not a nationality/ethnicity/religion or a number (professions,
relatives, diagnoses) -- no well-defined same-type generator; they are kept verbatim in both C1
and C2 and are excluded from masked-span recall, exactly as spans the Custodian
transform leaves untouched are excluded in the medical study.

  data/domains/redacted/<bench>.jsonl : masked spans -> '*' x len (offsets kept)
  data/domains/faker/<bench>.jsonl    : masked spans -> surrogate (gold remapped)

If data/transformed/<bench>.jsonl exists (Custodian transform arm, see
scripts/transform_corpus.py) you can pass --masked-from custodian to use the
Custodian-masked span set instead, for a like-for-like comparison.

Deterministic (Faker + random seeded per benchmark). CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
from datetime import date, datetime, timedelta

from faker import Faker

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from build_c1c2_corpora import build_faker, build_redact  # noqa: E402

DD = os.path.join(ROOT, "data", "domains")
BENCHES = {"finance_gretel": "en_US", "legal_tab": "en_GB", "enterprise_gretel": "en_US"}
SEED = 20260926

NATIONALITIES = ["Turkish", "Russian", "Ukrainian", "Polish", "Romanian", "Bulgarian", "Georgian",
                 "Armenian", "Moldovan", "Croatian", "Italian", "French", "Greek", "Hungarian",
                 "Slovak", "Czech", "Lithuanian", "Latvian", "Serbian", "Albanian", "Kurdish",
                 "Chechen", "Azerbaijani", "Portuguese", "Spanish", "German", "Austrian", "British"]
DEMONYMS = {"Irish", "Dutch", "Swiss", "Welsh", "Scottish", "Asian", "African", "European", "Catholic",
            "Muslim", "Jewish", "Orthodox", "Protestant", "Roma", "Gypsy", "Kurd", "Arab", "Chinese"}
ORG_HEADS = ["Court", "Court of Appeal", "Prosecutor", "Prosecutor's Office", "Police", "Ministry",
             "Office", "Prison", "Hospital", "University", "Bank", "Council", "Tribunal", "Department",
             "Directorate", "Committee", "Commission", "Administration", "Authority", "Assembly",
             "Service", "Security Directorate", "Gendarmerie", "Municipality", "Institute", "Centre",
             "Party", "Association", "Union", "Company", "Agency", "Board", "Registry"]
DATE_FORMATS = ["%d %B %Y", "%B %d, %Y", "%B %Y", "%d %B", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
                "%d.%m.%Y", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y", "%b %d, %Y", "%d %b %Y", "%Y%m%d",
                "%d/%m/%y", "%m/%d/%y", "%Y"]
TITLE_RE = re.compile(r"^((?:Mr|Mrs|Ms|Miss|Dr|Judge|Prof|Mr\.|Mrs\.|Ms\.|Dr\.)\s+)", re.I)


NUMBER_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "hundred"]
_NUMW_RE = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.I)


def swap_number_words(orig: str, rng: random.Random) -> str | None:
    """'seven counts' -> 'three counts'; None if no number word present."""
    if not _NUMW_RE.search(orig):
        return None
    def rep(m):
        w = rng.choice([x for x in NUMBER_WORDS if x != m.group(0).lower()])
        return w.capitalize() if m.group(0)[0].isupper() else w
    return _NUMW_RE.sub(rep, orig)


def shape(orig: str, rng: random.Random, letters: bool | None = None) -> str:
    """Format-preserving: digit->digit, upper->upper, lower->lower, rest kept.
    By default letters are kept when the value contains digits (so 'EMP104600'
    -> 'EMP583021', 'MRN-627574' -> 'MRN-120945': the type cue survives) and
    randomised only for all-letter values or when letters=True (keys/passwords)."""
    if letters is None:
        letters = not any(c.isdigit() for c in orig)
    for _ in range(20):
        out = []
        for ch in orig:
            if ch.isdigit():
                out.append(rng.choice(string.digits))
            elif letters and ch.isalpha() and ch.isascii():
                c = rng.choice(string.ascii_lowercase)
                out.append(c.upper() if ch.isupper() else c)
            else:
                out.append(ch)
        s = "".join(out)
        if s != orig:
            return s
    return orig


def _num_sub(o: str, rng: random.Random) -> str:
    """Replace each digit run with a plausible value of the same width:
    19xx/20xx years -> year in 1960-2024, <=2 digits -> 1..12 (valid as day,
    month or hour), anything else -> random digits."""
    def rep(m):
        t = m.group(0)
        if len(t) == 4 and t[:2] in ("19", "20"):
            return str(rng.randint(1960, 2024))
        if len(t) == 1:
            return str(rng.randint(1, 9))
        if len(t) == 2:  # width-preserving and valid as day / month / hour / minute
            return f"0{rng.randint(1, 9)}" if t.startswith("0") else str(rng.randint(10, 12))
        return "".join(rng.choice(string.digits) for _ in t)
    def ordinal(m):
        n = int(m.group(1))
        suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return m.group(1) + suf
    for _ in range(10):
        s = re.sub(r"\d+", rep, o)
        s = re.sub(r"(\d+)(st|nd|rd|th)\b", ordinal, s)
        if s != o:
            return s
    return s


def fake_date(orig: str, fk: Faker, rng: random.Random) -> str:
    o = orig.strip()
    for fmt in DATE_FORMATS:
        try:
            datetime.strptime(o, fmt)
        except ValueError:
            continue
        for _ in range(10):
            d = date(1960, 1, 1) + timedelta(days=rng.randint(0, 65 * 365))
            s = d.strftime(fmt)
            if not re.search(r"\b0\d", o):  # '3 July 2002' style: no zero padding
                s = re.sub(r"\b0(\d)", r"\1", s)
            if s != o:
                return s
    if any(c.isdigit() for c in o):
        # e.g. "30th April 2022", "between 1982 and 1984", "10.30 a.m.", "0400 hours"
        return _num_sub(o, rng)
    # month names / "summer" etc.: pick another month name if it is one
    months = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]
    if o in months:
        return rng.choice([m for m in months if m != o])
    return fk.date(pattern="%d %B %Y")


MALE_TITLES = {"mr", "sir", "lord", "monsieur", "m"}
FEMALE_TITLES = {"mrs", "ms", "miss", "lady", "madam", "madame", "mme", "mlle"}
CTX_TITLE_RE = re.compile(r"\b(Mr|Mrs|Ms|Miss|Sir|Lady|Madam|Mme|Mlle)\.?\s*$")


def title_gender(title: str) -> str | None:
    t = title.strip().rstrip(".").lower()
    return "m" if t in MALE_TITLES else "f" if t in FEMALE_TITLES else None


def fake_person(orig: str, fk: Faker, gender: str | None = None) -> str:
    """`gender` ('m'/'f') comes from a title outside the span (e.g. "Mr [Yalman]");
    a title inside the span takes precedence. The surrogate first name matches it,
    so "Mr Galip Yalman" never becomes "Mr Teresa Fowler"."""
    m = TITLE_RE.match(orig)
    prefix = m.group(1) if m else ""
    gender = title_gender(prefix) or gender
    first = {"m": fk.first_name_male, "f": fk.first_name_female}.get(gender, fk.first_name)
    rest = orig[len(prefix):]
    toks = rest.split()
    if len(toks) == 1:
        # single token: initial ("S.") -> initial, else surname-like
        if re.fullmatch(r"[A-Z]\.?", toks[0]):
            return prefix + fk.random_uppercase_letter() + ("." if toks[0].endswith(".") else "")
        return prefix + fk.last_name()
    if re.fullmatch(r"(?:[A-Z]\.)+", toks[0]):  # "S. Esmer", "Z.M. Necatigil", "J. De Meyer"
        inits = "".join(f"{fk.random_uppercase_letter()}." for _ in range(toks[0].count(".")))
        return prefix + f"{inits} {fk.last_name()}"
    return prefix + f"{first()} {fk.last_name()}"


def fake_org(orig: str, fk: Faker, domain: str) -> str:
    if domain == "legal":
        heads = sorted((h for h in ORG_HEADS if orig.endswith(h) or f"{h} " in orig), key=len, reverse=True)
        if heads:
            return f"{fk.city()} {heads[0]}"
    return fk.company()


def surrogate(label: str, orig: str, fk: Faker, rng: random.Random, domain: str,
              gender: str | None = None) -> str | None:
    """Return a same-type surrogate for `orig`, or None to leave it unmasked.
    `gender` is inferred from a title just before the span (see main)."""
    L = label.lower()
    if L in ("person", "name"):
        return fake_person(orig, fk, gender)
    if L == "first_name":
        return {"m": fk.first_name_male, "f": fk.first_name_female}.get(gender, fk.first_name)()
    if L == "last_name":
        return fk.last_name()
    if L in ("company", "company_name", "org"):
        return fake_org(orig, fk, domain)
    if L in ("street_address",):
        return fk.address().replace("\n", ", ") if "," in orig else fk.street_address()
    if L in ("address",):
        return fk.address().replace("\n", ", ")
    if L in ("city",):
        return fk.city()
    if L in ("country",):
        return fk.country()
    if L in ("state",):
        return fk.state() if hasattr(fk, "state") else fk.city()
    if L in ("postcode",):
        return fk.postcode()
    if L == "loc":
        return fk.city()
    if L in ("date", "date_of_birth", "datetime", "date_time", "time"):
        return fake_date(orig, fk, rng)
    if L == "phone_number":
        return fk.phone_number()
    if L == "email":
        return fk.email()
    if L == "user_name":
        return fk.user_name()
    if L == "ipv4":
        return fk.ipv4()
    if L == "ipv6":
        return fk.ipv6()
    if L == "url":
        return fk.url()
    if L == "credit_card_number":
        return fk.credit_card_number()
    if L in ("credit_card_security_code", "cvv"):
        return fk.credit_card_security_code()
    if L == "iban":
        return fk.iban()
    if L == "bban":
        return fk.bban()
    if L in ("swift_bic_code", "swift_bic"):
        return fk.swift(length=len(orig) if len(orig) in (8, 11) else None)
    if L == "bank_routing_number":
        return fk.aba()
    if L == "ssn":
        return fk.ssn() if hasattr(fk, "ssn") else shape(orig, rng)
    if L == "passport_number":
        return shape(orig, rng)
    if L in ("local_latlng", "coordinate"):
        return f"{fk.latitude():.6f}, {fk.longitude():.6f}" if "," in orig else shape(orig, rng)
    if L == "license_plate":
        return fk.license_plate()
    if L == "dem":
        # TAB DEM mixes nationality/ethnicity/religion ("British", "Catholic"),
        # ages ("75 year-old"), professions, relatives and diagnoses. Only the
        # first two have a well-defined same-type surrogate; the rest are left
        # unmasked (excluded from masked-span recall) rather than being swapped
        # for a nationality, which would not be same-type.
        if any(c.isdigit() for c in orig):
            return _num_sub(orig, rng)
        sw = swap_number_words(orig, rng)
        if sw:
            return sw
        if orig in NATIONALITIES or orig in DEMONYMS or re.fullmatch(r"[A-Z][a-z]+(ish|ian|ese|an|ic|i|ch)", orig):
            return rng.choice([n for n in NATIONALITIES if n.lower() != orig.lower()])
        return None
    if L in ("misc", "quantity"):
        if any(c.isdigit() for c in orig):
            return _num_sub(orig, rng) if L == "quantity" else shape(orig, rng)
        return swap_number_words(orig, rng)  # None -> left unmasked
    if L in ("api_key", "password"):
        return shape(orig, rng, letters=True)
    # CODE, QUANTITY, employee_id, customer_id, medical_record_number, account_number,
    # api_key, password, account_pin, pin, device/vehicle/unique ids, national_id, tax_id,
    # driver_license_number, certificate_license_number, biometric/health-plan ids, ...
    return shape(orig, rng)


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(BENCHES))
    ap.add_argument("--masked-from", choices=["all", "custodian"], default="all",
                    help="'all': every gold span with a same-type surrogate is masked; "
                         "'custodian': only spans the Custodian transform changed "
                         "(needs data/transformed/<bench>.jsonl)")
    args = ap.parse_args()
    for sub in ("redacted", "faker"):
        os.makedirs(os.path.join(DD, sub), exist_ok=True)
    for bench in (args.only or BENCHES):
        docs = load(os.path.join(DD, "original", f"{bench}.jsonl"))
        cust = None
        if args.masked_from == "custodian":
            cust = {r["doc_id"]: r for r in load(os.path.join(ROOT, "data", "transformed", f"{bench}.jsonl"))}
        fk = Faker(BENCHES[bench]); fk.seed_instance(SEED); rng = random.Random(SEED); random.seed(SEED)
        red, fak = [], []
        n_sp = n_mask = 0
        for d in docs:
            gs = d["gold_spans"]
            domain = d.get("meta", {}).get("domain", "")
            cache, surro = {}, []
            for g in gs:
                t = CTX_TITLE_RE.search(d["text"][max(0, g["start"] - 8):g["start"]])
                gender = title_gender(t.group(1)) if t else None
                key = (g["label"], g["text"], gender)
                if key not in cache:
                    cache[key] = surrogate(g["label"], g["text"], fk, rng, domain, gender)
                surro.append(cache[key])
            flags = [s is not None and s != g["text"] for s, g in zip(surro, gs)]
            if cust is not None:
                td = cust.get(d["doc_id"])
                if not td or len(td["gold_spans"]) != len(gs):
                    continue
                flags = [f and (g["text"] != t["text"]) for f, g, t in zip(flags, gs, td["gold_spans"])]
            by_pos = {(g["start"], g["end"]): s for g, s in zip(gs, surro)}
            red.append(build_redact(d, flags))
            fr = build_faker(d, flags, fk, surrogate=lambda g: by_pos[(g["start"], g["end"])])
            fr["meta"]["domain"] = domain
            fak.append(fr)
            n_sp += len(gs); n_mask += sum(flags)
            # sanity: remapped gold offsets match surrogate text
            assert all(fr["text"][s["start"]:s["end"]] == s["text"] for s in fr["gold_spans"])
            assert len(fr["gold_spans"]) == len(gs)
        for sub, rows in (("redacted", red), ("faker", fak)):
            with open(os.path.join(DD, sub, f"{bench}.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{bench:18s} docs={len(red):4d} spans={n_sp} masked={n_mask}")
    print("wrote data/domains/{redacted,faker}/*.jsonl")


if __name__ == "__main__":
    main()
