#!/usr/bin/env python3
"""GPT-5 (fresh runs) vs the stored panel on the same stratified subset.

Inputs: results/openai_audit/retest_gpt-5.jsonl (scripts/retest_openai.py --model gpt-5 --runs 1),
        stored predictions results/*.jsonl (paired_core / analyze_equivalence loaders).
On the documents that GPT-5 ran on (both conditions), reports for GPT-5, the stored
OpenAI detector (`openai_gpt`, gpt-4o-mini per results/openai_audit/model_check.json),
Gemma 4 31B and the rest of the panel:
  1. whole-document span F1 on the original documents (exact boundaries, label-agnostic:
     the paper's "type" mode) and with any-overlap matching;
  2. masked-span recall original -> transformed (snap_sym, overlap) with a document-cluster
     bootstrap, restricted to documents every compared detector covers.
Writes results/openai_audit/gpt5_comparison.{json,md}.
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import paired_core as P  # noqa: E402
from benchmarks.base import PHISpan  # noqa: E402
from evaluation.metrics import aggregate, score_document  # noqa: E402
from systems.base import PredictedSpan  # noqa: E402

OUT = os.path.join(ROOT, "results", "openai_audit")
NAMES = {"gpt-5": "GPT-5 (fresh run)", "openai_gpt": "stored OpenAI detector (gpt-4o-mini)", "gemma_4_31b": "Gemma 4 31B",
         "qwen3.5_35b_a3b": "Qwen 3.5 35B-A3B", "llama3.3_70b": "Llama 3.3 70B", "gemma_4_e4b": "Gemma 4 E4B",
         "qwen3.5_9b": "Qwen 3.5 9B", "presidio": "Presidio", "deid_roberta_i2b2": "OBI RoBERTa"}


def main():
    g5 = [json.loads(l) for l in open(os.path.join(OUT, "retest_gpt-5.jsonl"))]
    g5p = {(r["bench"], r["doc_id"], r["cond"]): r["spans"] for r in g5 if r["run"] == 0}
    docs = {(b, d) for (b, d, c) in g5p if (b, d, "original") in g5p and (b, d, "transformed") in g5p}
    cache = {}
    for b, (og, tg) in P.BENCH.items():
        cache[b] = (P.load(og), P.load(tg))
    md = ["# GPT-5 (fresh runs) vs the stored panel, same documents", "",
          f"Documents: {len(docs)} ({', '.join(f'{b} {sum(1 for x in docs if x[0]==b)}' for b in P.BENCH)}). "
          "GPT-5 run with the paper's prompt and parser at its default temperature (it rejects temperature 0).", ""]
    res = {"docs": len(docs)}

    # 1. whole-document F1 on original documents
    md += ["## Whole-document detection on the original documents", "",
           "| Detector | Docs | P | R | F1 (exact boundary) | F1 (any overlap) |", "|---|--:|--:|--:|--:|--:|"]
    res["wholedoc"] = {}
    for det in NAMES:
        sc_t, sc_r, n = [], [], 0
        for (b, d) in sorted(docs):
            o = cache[b][0].get(d)
            if not o:
                continue
            preds = g5p[(b, d, "original")] if det == "gpt-5" else o["preds"].get(det)
            if preds is None:
                continue
            gold = [PHISpan(start=g["start"], end=g["end"], label=g["label"], text=g.get("text", "")) for g in o["gold"]]
            pr = [PredictedSpan(start=p["start"], end=p["end"], label=str(p.get("label", "")), text="") for p in preds]
            sc_t.append(score_document(gold, pr, mode="type")); sc_r.append(score_document(gold, pr, mode="relaxed")); n += 1
        if not n:
            continue
        at, ar = aggregate(det, "subset", sc_t, "type"), aggregate(det, "subset", sc_r, "relaxed")
        res["wholedoc"][det] = {"docs": n, "p": at.precision, "r": at.recall, "f1": at.f1, "f1_overlap": ar.f1}
        md.append(f"| {NAMES[det]} | {n} | {at.precision:.3f} | {at.recall:.3f} | {at.f1:.3f} | {ar.f1:.3f} |")

    # 2. masked-span change on documents covered by GPT-5, the stored OpenAI detector and Gemma 4 31B
    rows = [r for r in P.paired_rows("snap_sym") if (r["bench"], r["doc"]) in docs]
    have = defaultdict(set)
    for r in rows:
        have[(r["bench"], r["doc"])].add(r["det"])
    common = {k for k, v in have.items() if {"openai_gpt", "gemma_4_31b"} <= v}
    texts = {b: (P._texts(P.BENCH[b][0]), P._texts(P.BENCH[b][1])) for b in P.BENCH}
    g5rows, seen = [], set()
    for r in rows:
        k = (r["bench"], r["doc"], r["idx"])
        if (r["bench"], r["doc"]) not in common or k in seen:
            continue
        seen.add(k)
        b, d, i = k
        go, gt = cache[b][0][d]["gold"][i], cache[b][1][d]["gold"][i]
        ts, te = P.snap_sym(texts[b][0][d], go["start"], go["end"], texts[b][1][d], gt["start"], gt["end"])
        g5rows.append({"bench": b, "doc": d, "det": "gpt-5",
                       "o": {"overlap": P.found(go["start"], go["end"], go["label"], g5p[(b, d, "original")], "overlap")},
                       "t": {"overlap": P.found(ts, te, gt["label"], g5p[(b, d, "transformed")], "overlap")}})
    md += ["", "## Masked-span recall, original → transformed (snap_sym, overlap)", "",
           f"Documents covered by GPT-5, the stored OpenAI detector and Gemma 4 31B: {len(common)}.", "",
           "| Detector | Spans | Recall orig → transf. | Diff | 90% CI (doc cluster) |", "|---|--:|--|--:|--|"]
    res["masked"] = {}
    for det in ("gpt-5", "openai_gpt", "gemma_4_31b"):
        sub = g5rows if det == "gpt-5" else [r for r in rows if r["det"] == det and (r["bench"], r["doc"]) in common]
        s = P.cluster_stats(P.per_doc_arrays(sub, "overlap"), B=3000)
        res["masked"][det] = s
        md.append(f"| {NAMES[det]} | {s['n']} | {s['rec_o']:.1f} → {s['rec_t']:.1f} | {s['diff']:+.2f} | "
                  f"[{s['ci90'][0]:+.2f}, {s['ci90'][1]:+.2f}] |")
    cost = sum(r["usage"]["in"] * 1.25 + r["usage"]["out"] * 10 for r in g5) / 1e6
    md += ["", f"GPT-5 calls: {len(g5)}; estimated cost ${cost:.2f}."]
    json.dump(res, open(os.path.join(OUT, "gpt5_comparison.json"), "w"), indent=1)
    open(os.path.join(OUT, "gpt5_comparison.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
