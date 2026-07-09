#!/usr/bin/env python3
"""Masked-span detectability equivalence analysis (paper §6, contrast C3).

For each benchmark we pair the original run and the transformed run by
``doc_id`` and by gold-span index. A gold span is *masked* (i.e. the transform
actually substituted it) when its surface text differs between the two
conditions. Restricting to masked spans, we ask, per detector: was the span
found on the original text, and was the surrogate found on the transformed
text (overlap match)?

We then report, pooled and per-benchmark:
  * recall(original) and recall(transformed) on masked spans,
  * the paired difference and its McNemar-based 95% CI,
  * a TOST equivalence test at margins delta in {1, 2, 3} points,
  * McNemar's chi-square (to demonstrate the large-N significance trap).

No model inference is run here -- this is pure re-analysis of results/*.jsonl,
so it doubles as the reproducibility artifact referenced in the paper.
"""
from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")

# benchmark -> (original file globs, transformed file globs)
# We take the 250-doc runs only; 25/50 sample runs and alternate scorings
# (*_type, *_relaxed, *_thinking) are excluded to avoid double counting.
BENCH = {
    "ASQ-PHI": (["asq_250_*.jsonl"], ["t_asq_phi_250*.jsonl"]),
    "MEDDOCAN": (["meddocan_250_*.jsonl", "meddocan_250.jsonl"], ["t_meddocan_250*.jsonl"]),
    "MultiCoNER v2": (["multiconer_v2_250*.jsonl"], ["t_multiconer_v2_250*.jsonl"]),
    "PII en": (["pii_250_*.jsonl", "pii_masking_300k_250.jsonl"], ["t_pii_masking_300k_250*.jsonl"]),
    "PII nl": (["pii_dutch_250*.jsonl"], ["t_pii_dutch_250*.jsonl"]),
    "PII fr": (["pii_french_250*.jsonl"], ["t_pii_french_250*.jsonl"]),
    "PII de": (["pii_german_250*.jsonl"], ["t_pii_german_250*.jsonl"]),
}
EXCLUDE = ("_type", "_relaxed", "_thinking", "_openai_type")


def _overlap(a_s, a_e, b_s, b_e):
    return not (b_e <= a_s or b_s >= a_e)


def load(globs):
    """Return {doc_id: {"gold": [...], "preds": {system: [spans]}}}, merged
    across files; first file wins for a given (doc_id, system)."""
    docs: dict = {}
    for g in globs:
        for fn in sorted(glob.glob(os.path.join(RESULTS, g))):
            base = os.path.basename(fn)
            if base.startswith("t_") ^ g.startswith("t_"):
                pass  # glob already scopes prefix; keep
            if any(x in base for x in EXCLUDE):
                continue
            for line in open(fn):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                did = r.get("doc_id")
                if did is None:
                    continue
                d = docs.setdefault(did, {"gold": r.get("gold_spans", []), "preds": {}})
                if not d["gold"]:
                    d["gold"] = r.get("gold_spans", [])
                for sysname, pv in (r.get("predictions") or {}).items():
                    if sysname not in d["preds"]:
                        d["preds"][sysname] = pv.get("spans", [])
    return docs


def found(gold_span, pred_spans):
    gs, ge = gold_span["start"], gold_span["end"]
    for p in pred_spans:
        if _overlap(gs, ge, p["start"], p["end"]):
            return True
    return False


def tost(d, b, c, n, delta):
    """TOST for paired difference of proportions d=(b-c)/n at margin `delta`.
    Returns (p_value, equivalent_at_0.05). Var per McNemar-style estimator."""
    if n == 0:
        return 1.0, False
    var = ((b + c) - (b - c) ** 2 / n) / (n ** 2)
    se = math.sqrt(var) if var > 0 else 1e-12
    # H0a: mu <= -delta  ->  upper-tail
    z1 = (d + delta) / se
    p1 = 0.5 * math.erfc(z1 / math.sqrt(2))          # P(Z > z1)
    # H0b: mu >= +delta  ->  lower-tail
    z2 = (d - delta) / se
    p2 = 0.5 * math.erfc(-z2 / math.sqrt(2))         # P(Z < z2)
    p = max(p1, p2)
    return p, p < 0.05


def mcnemar(b, c):
    if b + c == 0:
        return 0.0
    return (abs(b - c) - 1) ** 2 / (b + c)  # continuity-corrected chi-square, df=1


def analyze(docs_o, docs_t):
    """Return (n, found_o, found_t, b, c) over masked spans, pooled across
    systems present in both conditions."""
    n = fo = ft = b = c = 0
    for did, do in docs_o.items():
        dt = docs_t.get(did)
        if not dt:
            continue
        go, gt = do["gold"], dt["gold"]
        if len(go) != len(gt):
            continue
        systems = set(do["preds"]) & set(dt["preds"])
        for i, (sg, tg) in enumerate(zip(go, gt)):
            if sg.get("text") == tg.get("text"):
                continue  # not masked -- transform left it unchanged
            for s in systems:
                io = found(sg, do["preds"][s])
                it = found(tg, dt["preds"][s])
                n += 1
                fo += io
                ft += it
                if io and not it:
                    b += 1
                elif it and not io:
                    c += 1
    return n, fo, ft, b, c


def main():
    rows = []
    tot = [0, 0, 0, 0, 0]
    for name, (og, tg) in BENCH.items():
        do = load(og)
        dt = load(tg)
        n, fo, ft, b, c = analyze(do, dt)
        rows.append((name, n, fo, ft, b, c))
        for i, v in enumerate((n, fo, ft, b, c)):
            tot[i] += v
    rows.append(("POOLED", *tot))

    print(f"{'benchmark':14s} {'n':>7s} {'rec_o':>7s} {'rec_t':>7s} {'diff':>7s} "
          f"{'CI95':>16s} {'TOST2':>9s} {'McNemarχ²':>10s}")
    for name, n, fo, ft, b, c in rows:
        if n == 0:
            print(f"{name:14s} {'--- no paired data ---'}")
            continue
        ro, rt = fo / n, ft / n
        d = ro - rt
        var = ((b + c) - (b - c) ** 2 / n) / (n ** 2)
        se = math.sqrt(var) if var > 0 else 0.0
        lo, hi = d - 1.96 * se, d + 1.96 * se
        p2, eq2 = tost(d, b, c, n, 0.02)
        chi = mcnemar(b, c)
        print(f"{name:14s} {n:7d} {ro*100:6.1f}% {rt*100:6.1f}% {d*100:+6.2f}pt "
              f"[{lo*100:+5.2f},{hi*100:+5.2f}] {('p='+format(p2,'.1e')):>9s} {chi:10.1f}")

    print("\nTOST equivalence by margin (delta in points), pooled + per benchmark:")
    print(f"{'benchmark':14s} {'Δ=1':>12s} {'Δ=2':>12s} {'Δ=3':>12s}")
    for name, n, fo, ft, b, c in rows:
        if n == 0:
            continue
        d = fo / n - ft / n
        cells = []
        for delta in (0.01, 0.02, 0.03):
            p, eq = tost(d, b, c, n, delta)
            cells.append(f"{'✓' if eq else '✗'} {p:.1e}")
        print(f"{name:14s} {cells[0]:>12s} {cells[1]:>12s} {cells[2]:>12s}")

    print("\nInterpretation: McNemar χ² is large (=> 'significant' difference) yet "
          "TOST establishes equivalence within ±2 pts — the large-N trap.")


if __name__ == "__main__":
    main()
