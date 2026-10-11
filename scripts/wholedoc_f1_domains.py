#!/usr/bin/env python3
"""Whole-document span F1 on the non-medical domains: original vs Custodian
transform vs redaction, under exact-boundary ("type") and overlap ("relaxed")
matching -- the paper's whole-document view, next to the masked-span recall in
scripts/analyze_domains.py.

Exact-boundary F1 penalises surrogates whose remapped gold span is a few
characters off (transform_corpus.py aligns with difflib, and the Custodian
transform also rewrites non-PII text), so the overlap column separates
boundary noise from genuinely missed spans.

  PYTHONPATH=. python scripts/wholedoc_f1_domains.py  -> results/domains/wholedoc_f1.{md,json}
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from benchmarks.base import PHISpan  # noqa: E402
from evaluation.metrics import aggregate, score_document  # noqa: E402
from systems.base import PredictedSpan  # noqa: E402

BENCHES = [("finance_gretel", "Finance"), ("legal_tab", "Legal"), ("enterprise_gretel", "Enterprise")]
DETECTORS = [("presidio", "Presidio"), ("obi", "OBI deid_roberta"), ("qwen3_5_4b", "Qwen3.5-4B"),
             ("qwen3_5_9b", "Qwen3.5-9B"), ("qwen3_5_35b", "Qwen3.5-35B-A3B"), ("llama3_1_8b", "Llama-3.1-8B"),
             ("llama3_3_70b", "Llama-3.3-70B"), ("gemma_e4b", "Gemma-4-E4B"), ("gemma_4_31b", "Gemma-4-31B"),
             ("deepseek_v2_lite", "DeepSeek-V2-Lite")]
CORPUS = {"original": "data/domains/original/{b}.jsonl", "custodian": "data/transformed/{b}.jsonl",
          "redacted": "data/domains/redacted/{b}.jsonl"}
PRED = "results/domains/preds/{arm}/{b}__{det}.jsonl"


def jl(p):
    with open(os.path.join(ROOT, p)) as f:
        return [json.loads(l) for l in f if l.strip()]


def f1(bench: str, det: str, arm: str, mode: str) -> float | None:
    pp = PRED.format(arm=arm, b=bench, det=det)
    if not os.path.exists(os.path.join(ROOT, pp)):
        return None
    P = {}
    for r in jl(pp):
        if not r.get("error"):
            P.setdefault(r["doc_id"], r)
    sc = []
    for d in jl(CORPUS[arm].format(b=bench)):
        if d["doc_id"] not in P:
            continue
        gold = [PHISpan(s["start"], s["end"], s["label"], s["text"]) for s in d["gold_spans"]]
        pred = [PredictedSpan(s["start"], s["end"], s["label"], s["text"]) for s in P[d["doc_id"]]["spans"]]
        sc.append(score_document(gold, pred, mode=mode))
    return aggregate(det, bench, sc, mode).f1 if sc else None


def main() -> None:
    res = {}
    md = ["# Whole-document span F1: original vs Custodian transform vs redaction", "",
          "Mean over detectors; exact = span-exact, label-agnostic (`type`); overlap = any overlap (`relaxed`).", "",
          "| Domain | Exact: original | Custodian | Δ | Redacted | Overlap: original | Custodian | Δ | Redacted |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for b, bn in BENCHES:
        for det, _ in DETECTORS:
            for arm in CORPUS:
                for mode in ("type", "relaxed"):
                    res[f"{b}|{det}|{arm}|{mode}"] = f1(b, det, arm, mode)
        cells = []
        for mode in ("type", "relaxed"):
            m = {}
            for arm in CORPUS:
                v = [res[f"{b}|{d}|{arm}|{mode}"] for d, _ in DETECTORS]
                v = [x for x in v if x is not None]
                m[arm] = sum(v) / len(v)
            cells += [f"{m['original']:.3f}", f"{m['custodian']:.3f}", f"{m['custodian'] - m['original']:+.3f}",
                      f"{m['redacted']:.3f}"]
        md.append(f"| {bn} | " + " | ".join(cells) + " |")
    md += ["", "## Per detector (exact-boundary F1; overlap Δ in the last column)", "",
           "| Domain | Detector | Original | Custodian | Δ exact | Redacted | Δ overlap |",
           "|---|---|--:|--:|--:|--:|--:|"]
    for b, bn in BENCHES:
        for det, dn in sorted(DETECTORS, key=lambda x: -(res[f"{b}|{x[0]}|original|type"] or 0)):
            o, c, r = (res[f"{b}|{det}|{a}|type"] for a in ("original", "custodian", "redacted"))
            oo, co = res[f"{b}|{det}|original|relaxed"], res[f"{b}|{det}|custodian|relaxed"]
            if None in (o, c, r, oo, co):
                continue
            md.append(f"| {bn} | {dn} | {o:.3f} | {c:.3f} | {c - o:+.3f} | {r:.3f} | {co - oo:+.3f} |")
    out = os.path.join(ROOT, "results", "domains")
    with open(os.path.join(out, "wholedoc_f1.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(out, "wholedoc_f1.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\n".join(md[:9]))


if __name__ == "__main__":
    main()
