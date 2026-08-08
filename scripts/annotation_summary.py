#!/usr/bin/env python3
"""Summarise the human-reviewed surrogate-quality annotation, so the §7
failure-rate can be folded into the paper in one command.

Usage:
  python scripts/annotation_summary.py [rater1.csv] [rater2.csv]

With one file (default: data/annotation/surrogate_quality_prefilled.csv):
  reports valid-rate, type-consistency rate, and the failure-mode distribution,
  overall and per benchmark. Warns about any rows not yet reviewed.
With two files: also reports inter-rater agreement (raw + Cohen's kappa) on the
`failure` label.
"""
from __future__ import annotations
import csv, os, sys
from collections import Counter

D = os.path.join(os.path.dirname(__file__), os.pardir, "data", "annotation")
DEFAULT = os.path.join(D, "surrogate_quality_prefilled.csv")


def load(path):
    return list(csv.DictReader(open(path)))


def summarize(rows):
    n = len(rows)
    unrev = sum(1 for r in rows if r.get("reviewed", "").strip().upper() != "Y")
    if unrev:
        print(f"  ! {unrev}/{n} rows not yet reviewed (reviewed!=Y) — numbers are provisional")
    valid = sum(1 for r in rows if r["valid"].strip().upper() == "Y")
    tc = sum(1 for r in rows if r["type_consistent"].strip().upper() == "Y")
    print(f"  valid (well-formed, real-looking PHI): {valid}/{n} = {100*valid/n:.1f}%")
    print(f"  type-consistent:                       {tc}/{n} = {100*tc/n:.1f}%")
    fc = Counter(r["failure"].strip() or "(blank)" for r in rows)
    print("  failure modes: " + ", ".join(f"{k}={v} ({100*v/n:.1f}%)" for k, v in fc.most_common()))
    print("  by benchmark (valid-rate):")
    bys = {}
    for r in rows:
        bys.setdefault(r["benchmark"], []).append(r["valid"].strip().upper() == "Y")
    for b, vs in sorted(bys.items()):
        print(f"    {b:12s} {100*sum(vs)/len(vs):5.1f}%  (n={len(vs)})")


def kappa(a, b, key="failure"):
    """Cohen's kappa on a categorical label between two aligned rater lists."""
    pairs = [(x[key].strip(), y[key].strip()) for x, y in zip(a, b)]
    n = len(pairs)
    agree = sum(1 for x, y in pairs if x == y)
    po = agree / n
    cats = set(x for x, _ in pairs) | set(y for _, y in pairs)
    ca = Counter(x for x, _ in pairs); cb = Counter(y for _, y in pairs)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return po, k, agree, n


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    f1 = args[0] if args else DEFAULT
    print(f"== {os.path.basename(f1)} ==")
    r1 = load(f1)
    summarize(r1)
    if len(args) >= 2:
        r2 = load(args[1])
        print(f"\n== inter-rater agreement ({os.path.basename(f1)} vs {os.path.basename(args[1])}) ==")
        if len(r1) != len(r2):
            print("  ! row counts differ; align the sheets first"); return
        po, k, agree, n = kappa(r1, r2)
        print(f"  failure label: raw agreement {agree}/{n} = {100*po:.1f}%, Cohen's kappa = {k:.3f}")


if __name__ == "__main__":
    main()
