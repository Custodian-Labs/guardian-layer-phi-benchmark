#!/usr/bin/env python3
"""Per-benchmark detection results of the 11-detector panel on the ORIGINAL documents.

For each detector x benchmark: span F1 with exact boundaries (label-agnostic, the
"type" mode of evaluation/metrics.py, as in the paper's whole-document view) and
leakage = share of gold PHI spans not overlapped by any prediction.
Reads results/<bench>_250*.jsonl through scripts/analyze_equivalence.load.
Writes results/per_benchmark_original.{md,json}.
Usage: python scripts/per_benchmark_table.py
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
from analyze_equivalence import BENCH, load  # noqa: E402
from benchmarks.base import PHISpan  # noqa: E402
from evaluation.metrics import aggregate, score_document  # noqa: E402
from systems.base import PredictedSpan  # noqa: E402

DET = [("gemma_4_31b", "Gemma 4 31B"), ("llama3.3_70b", "Llama 3.3 70B"), ("qwen3.5_35b_a3b", "Qwen 3.5 35B-A3B"),
       ("openai_gpt", "GPT-4o-mini"), ("gemma_4_e4b", "Gemma 4 E4B"), ("qwen3.5_9b", "Qwen 3.5 9B"),
       ("qwen3.5_4b", "Qwen 3.5 4B"), ("presidio", "Presidio"), ("deepseek_v2_lite", "DeepSeek V2-Lite"),
       ("llama3.1_8b", "Llama 3.1 8B"), ("deid_roberta_i2b2", "OBI RoBERTa")]
SHORT = {"ASQ-PHI": "ASQ-PHI", "MEDDOCAN": "MEDDOCAN", "MultiCoNER v2": "MultiCoNER", "PII en": "PII-en",
         "PII nl": "PII-nl", "PII fr": "PII-fr", "PII de": "PII-de"}


def overlaps(g, p):
    return not (p["end"] <= g["start"] or p["start"] >= g["end"])


def main():
    res = {}
    for bench, (og, _) in BENCH.items():
        docs = load(og)
        for det, _ in DET:
            scores, n_gold, missed, n_docs = [], 0, 0, 0
            for d in docs.values():
                if det not in d["preds"]:
                    continue
                n_docs += 1
                gold = [PHISpan(start=g["start"], end=g["end"], label=g["label"], text=g.get("text", "")) for g in d["gold"]]
                pred = [PredictedSpan(start=p["start"], end=p["end"], label=str(p.get("label", "")), text="") for p in d["preds"][det]]
                scores.append(score_document(gold, pred, mode="type"))
                n_gold += len(d["gold"])
                missed += sum(not any(overlaps(g, p) for p in d["preds"][det]) for g in d["gold"])
            if n_docs:
                a = aggregate(det, bench, scores, "type")
                res[f"{det}|{bench}"] = {"docs": n_docs, "p": a.precision, "r": a.recall, "f1": a.f1,
                                         "leak": 100 * missed / max(1, n_gold)}
    head = "| Detector | " + " | ".join(SHORT[b] for b in BENCH) + " |"
    sep = "|---|" + "--:|" * len(BENCH)
    md = ["**Span F1** (exact boundaries, label-agnostic) on the original documents, 250 per benchmark:", "", head, sep]
    for det, name in DET:
        md.append(f"| {name} | " + " | ".join(f"{res[f'{det}|{b}']['f1']:.3f}" if f"{det}|{b}" in res else "–" for b in BENCH) + " |")
    md += ["", "**Leakage** (% of gold PHI spans not overlapped by any prediction) on the original documents:", "", head, sep]
    for det, name in DET:
        md.append(f"| {name} | " + " | ".join(f"{res[f'{det}|{b}']['leak']:.1f}" if f"{det}|{b}" in res else "–" for b in BENCH) + " |")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "per_benchmark_original.json"), "w"), indent=1)
    open(os.path.join(ROOT, "results", "per_benchmark_original.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
