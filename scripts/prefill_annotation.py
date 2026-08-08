#!/usr/bin/env python3
"""Auto-prefill the surrogate-quality annotation sheet with a heuristic FIRST
PASS, so a human rater (see data/annotation/GUIDELINES.md) only has to confirm
or correct rather than label from scratch.

Reads  data/annotation/surrogate_quality_sample.csv  (blank human columns)
Writes data/annotation/surrogate_quality_prefilled.csv with failure/valid/
type_consistent pre-populated by rule, plus blank `reviewed` and `notes`.

Design: the committed prefilled sheet IS the record of the auto guesses. A human
edits cells in place and sets reviewed=Y; `git diff` on the file then equals
exactly the cells the heuristic got wrong -> auto-vs-human agreement for free.

The heuristic is deliberately conservative: it only auto-assigns the failure
modes it can detect RELIABLY, and never guesses the ones that need human eyes.
  * failure=x_masked          : surrogate carries an x-mask run (>=4 x's) the
                                original did not -> generator leaked its mask.
                                (Reliable, mechanical.)
  * failure=truncated_garbled : ONLY when the surrogate fails its type's hard
                                format check (e.g. EMAIL without @, missing
                                digits in a phone). A *shorter* valid value is
                                NOT flagged (``Wolverhampton Codsall''->``Walsall''
                                is fine), and semantic garbling that stays
                                well-formed (``Chicago''->``Illino'') is left
                                for the human.
  * failure=salience_loss     : NEVER auto-assigned -- a semantic judgment
                                (famous/specific entity -> generic, e.g.
                                Cedars-Sinai -> Vidant). Human only.
  * failure=none              : otherwise. NOTE this is a *suggestion*: the
                                human must still scan the ``none'' rows for
                                truncation, garbling, and salience loss.
  valid            = N iff failure in {x_masked, truncated_garbled}; else Y.
  type_consistent  = N iff the surrogate clearly violates its type's format;
                     else Y (the generator is same-type by design).
CPU only, deterministic.
"""
from __future__ import annotations
import csv, os, re

D = os.path.join(os.path.dirname(__file__), os.pardir, "data", "annotation")
SRC = os.path.join(D, "surrogate_quality_sample.csv")
OUT = os.path.join(D, "surrogate_quality_prefilled.csv")

X_RUN = re.compile(r"x{4,}", re.I)
EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")      # contains an address
IPV4 = re.compile(r"\d{1,3}(\.\d{1,3}){3}")          # contains a dotted quad
HAS_DIGIT = re.compile(r"\d")
HAS_ALNUM = re.compile(r"[^\W_]", re.UNICODE)         # any letter or digit


def category(label: str) -> str:
    L = (label or "").upper()
    has = lambda *k: any(x in L for x in k)
    if has("MAIL", "CORREO"): return "email"
    if has("IP"): return "ip"
    if has("USER"): return "username"
    if has("PHONE", "TEL"): return "phone"
    if has("EDAD", "AGE", "BOD"): return "age"
    if has("FECHA", "DATE", "TIME", "DOB"): return "date"
    if has("PASSPORT", "SOCIAL", "IDCARD", "DRIVER", "LICEN", "SSN",
           "NUMBER", "ID_", "CARD", "TAX"): return "idnum"
    if has("TERRITORIO", "CALLE", "PAIS", "LOC", "GEO", "CITY", "ADDR",
           "STREET", "COUNTRY", "STATE", "ZIP", "HOSP"): return "loc"
    return "name"


def format_ok(cat: str, s: str) -> bool:
    """Hard per-type format check; True if the surrogate looks well-formed.
    Lenient by design -- only catches clearly malformed values, not short ones."""
    if not s.strip():
        return False
    if cat == "email": return bool(EMAIL.search(s))
    if cat == "ip": return bool(IPV4.search(s))
    if cat in ("phone", "date", "age"): return bool(HAS_DIGIT.search(s))
    # idnum (passports/licences may be alphanumeric), name, loc, username:
    # any alphanumeric content is well-formed enough; semantics is the human's job.
    return bool(HAS_ALNUM.search(s))


def assess(row: dict) -> tuple[str, str, str]:
    orig, sur = row["original"], row["surrogate"]
    cat = category(row["type"])
    # 1) x-mask leaked by the generator (reliable, mechanical)
    if X_RUN.search(sur) and not X_RUN.search(orig):
        return "x_masked", "N", "Y"
    # 2) hard format break (malformed / empty). A shorter valid value is NOT
    #    flagged; semantic garbling that stays well-formed is left to the human.
    if not format_ok(cat, sur):
        return "truncated_garbled", "N", "N"
    # 3) well-formed same-type surrogate -- suggestion only; the human still
    #    scans these for truncation, garbling, and salience loss.
    return "none", "Y", "Y"


def main():
    rows = list(csv.DictReader(open(SRC)))
    cols = ["id", "benchmark", "doc_id", "type", "original", "surrogate",
            "context", "failure", "valid", "type_consistent", "reviewed", "notes"]
    from collections import Counter
    fc = Counter()
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            failure, valid, tc = assess(r)
            fc[failure] += 1
            w.writerow({**{k: r.get(k, "") for k in cols[:7]},
                        "failure": failure, "valid": valid,
                        "type_consistent": tc, "reviewed": "", "notes": ""})
    n = len(rows)
    print(f"wrote {n} prefilled rows -> {OUT}")
    print("auto failure distribution:", dict(fc))
    print(f"auto valid=Y: {sum(1 for r in rows if assess(r)[1]=='Y')}/{n}")
    print("\nNEXT: a human opens the sheet, corrects wrong cells, sets reviewed=Y.")
    print("`git diff surrogate_quality_prefilled.csv` then = auto-vs-human disagreements.")


if __name__ == "__main__":
    main()
