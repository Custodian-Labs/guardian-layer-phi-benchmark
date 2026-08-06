#!/usr/bin/env python3
"""Draw a stratified sample of masked spans for human surrogate-quality annotation.

For each benchmark we pair the original and Custodian-transformed subsets by
doc_id and gold-span index; a span is *masked* when its surface text changed.
For each masked span we record the original value, the surrogate, its type, and
a short context window, then stratify-sample ~N across benchmarks. Output is an
annotation sheet (CSV) with blank fields for a human rater:

  valid            : is the surrogate a well-formed, same-type value? (Y/N)
  type_consistent  : does it read as the same PHI type as the original? (Y/N)
  failure          : none | truncated_garbled | salience_loss | x_masked | other
  notes            : free text

This quantifies the §7 error typology independently of any detector.
Usage: python scripts/sample_surrogate_annotation.py [N]   (default 200)
"""
from __future__ import annotations
import csv, json, os, random, sys

DL = os.path.join(os.path.dirname(__file__), os.pardir, "web", "data", "downloads")
OUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "annotation")
PAIRS = {  # benchmark -> (original file, transformed file)
    "ASQ-PHI": ("asq_phi_250.jsonl", "asq_phi_250_transformed.jsonl"),
    "MEDDOCAN": ("meddocan_250.jsonl", "meddocan_250_transformed.jsonl"),
    "MultiCoNER": ("multiconer_v2_250.jsonl", "multiconer_v2_250_transformed.jsonl"),
    "PII-en": ("pii_masking_300k_250.jsonl", "pii_masking_300k_250_transformed.jsonl"),
    "PII-nl": ("pii_masking_300k_dutch_250.jsonl", "pii_masking_300k_dutch_250_transformed.jsonl"),
    "PII-fr": ("pii_masking_300k_french_250.jsonl", "pii_masking_300k_french_250_transformed.jsonl"),
    "PII-de": ("pii_masking_300k_german_250.jsonl", "pii_masking_300k_german_250_transformed.jsonl"),
}


def load(fn):
    d = {}
    p = os.path.join(DL, fn)
    if not os.path.exists(p):
        return d
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line)
            d[r["doc_id"]] = r
    return d


def masked_spans(bench):
    ofn, tfn = PAIRS[bench]
    O, T = load(ofn), load(tfn)
    rows = []
    for did, od in O.items():
        td = T.get(did)
        if not td:
            continue
        og, tg = od.get("gold_spans", []), td.get("gold_spans", [])
        if len(og) != len(tg):
            continue
        ttext = td.get("text", "")
        for go, gt in zip(og, tg):
            if go.get("text") == gt.get("text"):
                continue  # not masked
            s, e = gt["start"], gt["end"]
            ctx = ttext[max(0, s - 40):s] + "⟦" + ttext[s:e] + "⟧" + ttext[e:e + 40]
            rows.append({
                "benchmark": bench, "doc_id": did, "type": go.get("label", ""),
                "original": go.get("text", ""), "surrogate": gt.get("text", ""),
                "context": ctx.replace("\n", " "),
            })
    return rows


def main():
    n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    random.seed(20260806)
    per = max(1, n_total // len(PAIRS))
    sample = []
    for bench in PAIRS:
        rows = masked_spans(bench)
        random.shuffle(rows)
        sample.extend(rows[:per])
    random.shuffle(sample)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "surrogate_quality_sample.csv")
    cols = ["id", "benchmark", "doc_id", "type", "original", "surrogate", "context",
            "valid", "type_consistent", "failure", "notes"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(sample, 1):
            r["id"] = i
            for c in ("valid", "type_consistent", "failure", "notes"):
                r[c] = ""
            w.writerow(r)
    # per-benchmark / per-type counts
    from collections import Counter
    bc = Counter(r["benchmark"] for r in sample)
    tc = Counter(r["type"] for r in sample)
    print(f"wrote {len(sample)} spans -> {out}")
    print("by benchmark:", dict(bc))
    print("top types:", dict(tc.most_common(8)))
    print("\nAnnotation guide:")
    print("  valid=Y if the surrogate is a well-formed, same-type value a human reads as real PHI")
    print("  failure: none | truncated_garbled | salience_loss | x_masked | other")


if __name__ == "__main__":
    main()
