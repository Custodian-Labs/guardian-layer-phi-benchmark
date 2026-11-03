#!/usr/bin/env python3
"""E4a: what does the commercial transform preserve? Measured from the paired
original/transformed documents only (black box: nothing is inferred about the
vendor's internals).

Unit: unique masked gold span (snap_sym re-projection, scripts/paired_core.py).
  repeated mentions  within a document, an original value masked >= 2 times:
                     consistent iff every occurrence received the same surrogate
  character class    pattern with letters -> a, digits -> 9, other characters kept;
                     exact = identical pattern (same length); collapsed = identical
                     after merging runs (aaa -> a, 999 -> 9)
  dates              format (exact pattern) and calendar validity of numeric dates
                     (d/m/y or m/d/y with a valid day for that month, any separator)
  x-masking          surrogate contains a run of >= 3 'x' that the original lacks
  verbatim           not counted here: unchanged spans are not masked (see coverage)
Writes results/audit/transform_properties.{json,md}.
"""
import calendar
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import paired_core as P  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT = os.path.join(ROOT, "results", "audit")
PHONE = {"TEL", "NUMERO_TELEFONO", "PHONE_NUMBER", "NUMERO_FAX"}
EMAIL = {"EMAIL", "CORREO_ELECTRONICO", "EMAIL_ADDRESS"}


def cls(s):
    return "".join("a" if ch.isalpha() else "9" if ch.isdigit() else ch for ch in s)


def collapsed(p):
    return re.sub(r"(a)+", "a", re.sub(r"(9)+", "9", p))


def xmasked(o, s):
    return bool(re.search(r"x{3,}", s, re.I)) and not re.search(r"x{3,}", o, re.I)


def numeric_date_valid(s):
    """None if not a purely numeric d/m/y-like date, else True/False calendar validity."""
    m = re.fullmatch(r"\s*(\d{1,4})[./-](\d{1,2})[./-](\d{1,4})\s*", s)
    if not m:
        return None
    a, b, c = (int(x) for x in m.groups())
    cands = []
    if len(m.group(1)) == 4:
        cands.append((a, b, c))           # y-m-d
    else:
        y = c if c > 99 else 2000 + c
        cands += [(y, b, a), (y, a, b)]   # d/m/y or m/d/y
    return any(1 <= mo <= 12 and 1 <= d <= calendar.monthrange(y if y > 0 else 2000, mo)[1] for y, mo, d in cands)


def main():
    os.makedirs(OUT, exist_ok=True)
    # unique masked spans with labels (one detector's rows suffice: span fields are detector-independent)
    spans = {}
    cache = {}
    for r in P.paired_rows("snap_sym", cache=cache):
        spans.setdefault((r["bench"], r["doc"], r["idx"]), r)
    labels = {}
    for b, (do, dt, _) in ((b, cache[b]) for b in P.BENCH if b in cache):
        for did, o in do.items():
            for i, g in enumerate(o["gold"]):
                labels[(b, did, i)] = g["label"]
    res, md = {}, ["# What the transform preserves (E4a)", "",
                   f"Unique masked spans (snap_sym): {len(spans):,}. Black-box measurements from paired documents.", ""]

    # repeated mentions
    groups = defaultdict(set)
    for (b, did, i), r in spans.items():
        groups[(b, did, r["orig_text"])].add(r["surr_text"])
    rep = {k: v for k, v in groups.items() if sum(1 for (b, d, i), r in spans.items() if (b, d, r["orig_text"]) == k) >= 2}
    cons = sum(len(v) == 1 for v in rep.values())
    res["repeated_mentions"] = {"values": len(rep), "consistent": cons}
    md += ["## Repeated mentions", "",
           f"Original values masked at least twice in the same document: {len(rep)}; all occurrences got the same surrogate: "
           f"{cons} ({100*cons/max(1,len(rep)):.1f}%).", ""]

    # character class by group
    def group(key):
        lab = labels.get(key, ""); gt = P.gold_type(lab)
        if lab in EMAIL:
            return "email"
        if lab in PHONE:
            return "phone"
        return {"ID": "id", "DATE": "date", "AGE": "age", "NAME": "name", "LOC": "location"}.get(gt, "other")
    by = defaultdict(list)
    for k, r in spans.items():
        by[group(k)].append(r)
    md += ["## Format: character-class pattern (letters → a, digits → 9)", "",
           "| Group | Spans | Exact pattern kept | Collapsed pattern kept | x-masked |", "|---|--:|--:|--:|--:|"]
    res["format"] = {}
    for g in ("id", "phone", "email", "date", "age", "name", "location", "other"):
        rs = by.get(g, [])
        if not rs:
            continue
        ex = sum(cls(r["orig_text"]) == cls(r["surr_text"]) for r in rs)
        co = sum(collapsed(cls(r["orig_text"])) == collapsed(cls(r["surr_text"])) for r in rs)
        xm = sum(xmasked(r["orig_text"], r["surr_text"]) for r in rs)
        res["format"][g] = {"n": len(rs), "exact": ex, "collapsed": co, "xmasked": xm}
        md.append(f"| {g} | {len(rs):,} | {100*ex/len(rs):.1f}% | {100*co/len(rs):.1f}% | {100*xm/len(rs):.1f}% |")
    idp = res["format"].get("id", {"n": 0, "exact": 0}); ph = res["format"].get("phone", {"n": 0, "exact": 0})
    n_idp = idp["n"] + ph["n"]
    md += ["", f"IDs + phones, exact pattern kept: {100*(idp['exact']+ph['exact'])/max(1,n_idp):.1f}%. "
           f"Dates: {100*res['format'].get('date',{'exact':0,'n':1})['exact']/max(1,res['format'].get('date',{'n':1})['n']):.1f}%. "
           f"E-mail x-masked: {100*res['format'].get('email',{'xmasked':0,'n':1})['xmasked']/max(1,res['format'].get('email',{'n':1})['n']):.1f}%.", ""]

    # dates
    dv = [(r["orig_text"], r["surr_text"], numeric_date_valid(r["orig_text"]), numeric_date_valid(r["surr_text"])) for r in by.get("date", [])]
    num = [x for x in dv if x[2] is not None]
    inval = [x for x in num if x[3] is False]
    nonnum = [x for x in num if x[3] is None]
    res["dates"] = {"numeric_originals": len(num), "surrogate_invalid": len(inval), "surrogate_not_numeric_date": len(nonnum),
                    "examples_invalid": [x[:2] for x in inval[:10]], "examples_not_numeric": [x[:2] for x in nonnum[:10]]}
    md += ["## Dates", "",
           f"Numeric originals (d/m/y-like): {len(num)}. Surrogate is not a valid calendar date: {len(inval)} "
           f"({100*len(inval)/max(1,len(num)):.1f}%); surrogate no longer a numeric date: {len(nonnum)} ({100*len(nonnum)/max(1,len(num)):.1f}%).",
           "Invalid examples: " + "; ".join(f"`{a}` → `{b}`" for a, b in res["dates"]["examples_invalid"][:6]),
           "Not-a-date examples: " + "; ".join(f"`{a}` → `{b}`" for a, b in res["dates"]["examples_not_numeric"][:6]), "",
           "Date-shift policy (constant offset per document, interval preservation) is not documented by the vendor; "
           "it cannot be established from 1-2 dates per document and is listed as unknown.", ""]

    # numeric-field deltas: is the "shift" just +1 on some fields? (recoverable by subtraction)
    def nums(x):
        return [int(v) for v in re.findall(r"\d+", x)]
    kinds = Counter(); kb = defaultdict(Counter); exk = defaultdict(list)
    for k, r in ((k, r) for k, r in spans.items() if group(k) == "date"):
        o, sv = nums(r["orig_text"]), nums(r["surr_text"])
        if not o or len(o) != len(sv):
            kind = "different number of numeric fields"
        else:
            dlt = [b2 - a2 for a2, b2 in zip(o, sv)]
            kind = ("numbers unchanged" if all(v == 0 for v in dlt) else
                    "every field +0 or +1" if all(v in (0, 1) for v in dlt) else "other change")
        kinds[kind] += 1; kb[k[0]][kind] += 1
        if len(exk[kind]) < 6:
            exk[kind].append((r["orig_text"], r["surr_text"]))
    nd = sum(kinds.values())
    res["date_deltas"] = {"n": nd, "kinds": dict(kinds), "by_bench": {b: dict(v) for b, v in kb.items()},
                          "examples": {k: v for k, v in exk.items()}}
    md += ["### Date surrogates: numeric-field changes", "",
           "| Change | Dates | Share |", "|---|--:|--:|"]
    md += [f"| {k} | {v} | {100*v/nd:.1f}% |" for k, v in kinds.most_common()]
    md += ["", "By benchmark (every field +0 or +1 / all dates): " + "; ".join(
        f"{b} {v.get('every field +0 or +1', 0)}/{sum(v.values())}" for b, v in kb.items()),
        "Examples (+0/+1): " + "; ".join(f"`{a}` → `{b}`" for a, b in exk.get("every field +0 or +1", [])),
        "", "**Privacy implication.** Where every numeric field moves by 0 or +1, the original date is among at most "
        "2^k candidates (k numeric fields, typically 8) obtained by subtracting 0 or 1 from each field, so these "
        "surrogates offer little protection for the date value. This is a black-box observation on these corpora; the "
        "vendor's date policy is undocumented.", ""]

    # by benchmark x-masking
    md += ["## x-masking by benchmark", "", "| Benchmark | Spans | x-masked |", "|---|--:|--:|"]
    res["xmask_by_bench"] = {}
    for b in P.BENCH:
        rs = [r for k, r in spans.items() if k[0] == b]
        if rs:
            xm = sum(xmasked(r["orig_text"], r["surr_text"]) for r in rs)
            res["xmask_by_bench"][b] = {"n": len(rs), "xmasked": xm}
            md.append(f"| {b} | {len(rs):,} | {100*xm/len(rs):.1f}% |")
    md += ["", "Locale/language consistency: LLM-judge run queued on GPU 5 (scripts/locale_consistency.py → results/locale/).",
           "Unknown vendor properties (not inferable from outputs): generator model family, date-shift policy, whether consistency "
           "is guaranteed or incidental, locale handling, training data."]
    json.dump(res, open(os.path.join(OUT, "transform_properties.json"), "w"), indent=1)
    open(os.path.join(OUT, "transform_properties.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
