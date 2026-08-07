#!/usr/bin/env python3
"""Aggregate results/c1c2/*.json into the C1 (redact) and C2 (faker) full-panel
tables: benchmark x {Presidio, OBI, Qwen-9B, Gemma-31B} masked-span recall (%).
Presidio C1/C2 come from the paper's standalone baseline runs (6 benchmarks).
Prints markdown + LaTeX rows; marks missing cells as '-'.
"""
import json, os
ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
C = os.path.join(ROOT, "results", "c1c2")
BENCH = ["asq_phi", "meddocan", "multiconer_v2", "pii_masking_300k", "pii_dutch", "pii_french", "pii_german"]
LABEL = {"asq_phi": "ASQ-PHI", "meddocan": "MEDDOCAN", "multiconer_v2": "MultiCoNER",
         "pii_masking_300k": "PII-en", "pii_dutch": "PII-nl", "pii_french": "PII-fr", "pii_german": "PII-de"}
DET = [("obi", "OBI"), ("qwen3_5_9b", "Qwen-9B"), ("gemma_4_31b", "Gemma-31B")]
# Presidio from the paper's baseline scripts (6 benchmarks; MultiCoNER n/a)
PRESIDIO = {
    "redacted": {"asq_phi": 0.0, "meddocan": 0.0, "pii_masking_300k": 1.9, "pii_dutch": 3.8, "pii_french": 2.2, "pii_german": 0.0},
    "faker":  {"asq_phi": 97.7, "meddocan": 79.7, "pii_masking_300k": 89.2, "pii_dutch": 93.0, "pii_french": 78.5, "pii_german": 78.1},
}


def val(corpus, bench, sysname):
    f = os.path.join(C, f"{corpus}_{bench}_{sysname}.json")
    if not os.path.exists(f):
        return None
    return json.load(open(f)).get("recall_pct")


def table(corpus, title):
    print(f"\n### {title} — masked-span recall (%)")
    hdr = f"{'benchmark':16s} {'Presidio':>9s} " + " ".join(f"{n:>10s}" for _, n in DET)
    print(hdr)
    for b in BENCH:
        p = PRESIDIO[corpus].get(b)
        cells = [f"{p:9.1f}" if p is not None else f"{'-':>9s}"]
        for s, _ in DET:
            v = val(corpus, b, s)
            cells.append(f"{v:10.1f}" if v is not None else f"{'-':>10s}")
        print(f"{LABEL[b]:16s} " + " ".join(cells))


table("redacted", "C1  Redaction floor")
table("faker", "C2  Open-surrogate (Faker) retention")
# completeness
missing = [f"{c}/{b}/{s}" for c in ("redacted", "faker") for b in BENCH for s, _ in DET if val(c, b, s) is None]
print(f"\nmissing cells ({len(missing)}): " + (", ".join(missing) if missing else "none — COMPLETE"))
