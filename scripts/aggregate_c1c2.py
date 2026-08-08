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
# Presidio from the paper's baseline scripts (full corpus, 6 benchmarks; MultiCoNER n/a)
PRESIDIO = {
    "redacted": {"asq_phi": 0.0, "meddocan": 0.0, "pii_masking_300k": 1.9, "pii_dutch": 3.8, "pii_french": 2.2, "pii_german": 0.0},
    "faker":  {"asq_phi": 97.7, "meddocan": 79.7, "pii_masking_300k": 89.2, "pii_dutch": 93.0, "pii_french": 78.5, "pii_german": 78.1},
}
# Presidio full-corpus masked-span counts (paper Table); used for its bootstrap CI
PRESIDIO_N = {"asq_phi": 531, "meddocan": 2925, "pii_masking_300k": 471, "pii_dutch": 373, "pii_french": 544, "pii_german": 415}


def ci_from_counts(found, n, B=5000, seed=20260808):
    if not n:
        return None
    import numpy as np
    rng = np.random.default_rng(seed)
    samp = rng.binomial(n, found / n, B) / n * 100
    lo, hi = np.percentile(samp, [2.5, 97.5])
    return round((hi - lo) / 2, 1)


def val(corpus, bench, sysname):
    f = os.path.join(C, f"{corpus}_{bench}_{sysname}.json")
    if not os.path.exists(f):
        return None
    return json.load(open(f)).get("recall_pct")


def ci_halfwidth(corpus, bench, sysname, B=5000, seed=20260808):
    """95% bootstrap-CI half-width (%) of masked-span recall, from (found, N).
    A proportion's bootstrap == resampling Binomial(N, p_hat); reproduces the
    paper's per-cell +-x. Requires numpy; returns None if unavailable/missing."""
    f = os.path.join(C, f"{corpus}_{bench}_{sysname}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f)); n, found = d.get("masked", 0), d.get("found", 0)
    if not n:
        return None
    import numpy as np
    rng = np.random.default_rng(seed)
    samp = rng.binomial(n, found / n, B) / n * 100
    lo, hi = np.percentile(samp, [2.5, 97.5])
    return round((hi - lo) / 2, 1)


def table(corpus, title, ci=False):
    print(f"\n### {title} — masked-span recall (%)" + ("  [value ±95% bootstrap-CI half-width]" if ci else ""))
    hdr = f"{'benchmark':16s} {'Presidio':>11s} " + " ".join(f"{n:>12s}" for _, n in DET)
    print(hdr)
    for b in BENCH:
        p = PRESIDIO[corpus].get(b)
        if p is None:
            cells = [f"{'-':>11s}"]
        elif ci:
            ph = ci_from_counts(round(p / 100 * PRESIDIO_N[b]), PRESIDIO_N[b])
            cells = [f"{p:>5.1f}±{ph:<4.1f}"]
        else:
            cells = [f"{p:11.1f}"]
        for s, _ in DET:
            v = val(corpus, b, s)
            if v is None:
                cells.append(f"{'-':>12s}")
            elif ci:
                h = ci_halfwidth(corpus, b, s)
                cells.append(f"{v:>6.1f}±{h:<5.1f}")
            else:
                cells.append(f"{v:12.1f}")
        print(f"{LABEL[b]:16s} " + " ".join(cells))


def latex_rows(corpus):
    """Emit the paper's table body rows (benchmark & N & \\vc{recall}{ci} x4)."""
    NROW = {"asq_phi": "267", "meddocan": "1{,}419", "multiconer_v2": "8",
            "pii_masking_300k": "210", "pii_dutch": "195", "pii_french": "220", "pii_german": "204"}
    TEX = {"asq_phi": "ASQ-PHI", "meddocan": "MEDDOCAN", "multiconer_v2": "MultiCoNER",
           "pii_masking_300k": "PII en", "pii_dutch": "PII nl", "pii_french": "PII fr", "pii_german": "PII de"}
    print(f"\n% ---- {corpus} \\vc rows ----")
    for b in BENCH:
        p = PRESIDIO[corpus].get(b)
        pc = "--" if p is None else f"\\vc{{{p:.1f}}}{{{ci_from_counts(round(p/100*PRESIDIO_N[b]), PRESIDIO_N[b]):.1f}}}"
        cells = [pc]
        for s, _ in DET:
            v = val(corpus, b, s); h = ci_halfwidth(corpus, b, s)
            cells.append(f"\\vc{{{v:.1f}}}{{{h:.1f}}}")
        print(f"{TEX[b]} & \\nk{{{NROW[b]}}} & " + " & ".join(cells) + "\\\\")


import sys
CI = "--ci" in sys.argv
if "--latex" in sys.argv:
    latex_rows("redacted"); latex_rows("faker")
else:
    table("redacted", "C1  Redaction floor", ci=CI)
    table("faker", "C2  Open-surrogate (Faker) retention", ci=CI)
# completeness
missing = [f"{c}/{b}/{s}" for c in ("redacted", "faker") for b in BENCH for s, _ in DET if val(c, b, s) is None]
print(f"\nmissing cells ({len(missing)}): " + (", ".join(missing) if missing else "none — COMPLETE"))
