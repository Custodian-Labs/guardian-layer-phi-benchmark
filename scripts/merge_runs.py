"""Merge multiple per-system JSONL runs into a single combined results file.

Each `results/<name>.jsonl` produced by run_benchmark.py contains one row per
document with the per-system predictions for the systems run in that
invocation. When we want a side-by-side comparison across systems that were
run separately (e.g. some on CPU, some on GPU, some via API), we need to
merge the rows by doc_id.

Usage
-----
python scripts/merge_runs.py --out results/llm_50_merged \
    results/llm_50_presidio.jsonl \
    results/llm_50_obi.jsonl \
    results/llm_50_openai.jsonl \
    results/llm_50_gemma.jsonl \
    results/llm_50_qwen.jsonl \
    results/llm_50_moonlight.jsonl \
    results/llm_50_deepseek.jsonl

Writes:
    results/llm_50_merged.jsonl          # merged per-doc rows
    results/llm_50_merged.summary.json   # aggregate metrics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output prefix (no extension)")
    p.add_argument("inputs", nargs="+", help="Per-system JSONL files to merge")
    p.add_argument("--mode", default=None, help="Override mode in summary; otherwise inferred")
    p.add_argument("--benchmark", default=None, help="Override benchmark; otherwise inferred from rows")
    args = p.parse_args()

    by_doc: dict[str, dict] = {}
    for path in args.inputs:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                key = row["doc_id"]
                if key not in by_doc:
                    by_doc[key] = {
                        "doc_id": key,
                        "gold_n": row.get("gold_n", 0),
                        "text": row.get("text", ""),
                        "gold_spans": row.get("gold_spans", []),
                        "predictions": {},
                    }
                by_doc[key]["predictions"].update(row.get("predictions", {}))

    out_jsonl = Path(args.out + ".jsonl")
    with out_jsonl.open("w") as f:
        for doc_id in sorted(by_doc):
            f.write(json.dumps(by_doc[doc_id], ensure_ascii=False) + "\n")

    # Re-aggregate per-system summaries from merged predictions.
    per_system: dict[str, list[dict]] = {}
    for doc in by_doc.values():
        for sys_name, p in doc["predictions"].items():
            per_system.setdefault(sys_name, []).append(p["score"])

    benchmark = args.benchmark or _infer_benchmark(args.inputs)
    mode = args.mode or _infer_mode(args.inputs)

    summary = []
    for sys_name, scores in per_system.items():
        tp = sum(s["tp"] for s in scores)
        fp = sum(s["fp"] for s in scores)
        fn = sum(s["fn"] for s in scores)
        cl = sum(s["gold_chars_leaked"] for s in scores)
        ct = sum(s["gold_chars_total"] for s in scores)
        n_docs = len(scores)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        summary.append({
            "system": sys_name,
            "benchmark": benchmark,
            "mode": mode,
            "n_docs": n_docs,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "leakage_rate": 1 - rec,
            "char_leakage_rate": cl / ct if ct else 0.0,
        })
    summary.sort(key=lambda r: -r["f1"])

    summary_path = Path(args.out + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"merged {len(by_doc)} docs from {len(args.inputs)} runs")
    print(f"  -> {out_jsonl}")
    print(f"  -> {summary_path}")
    print()
    print(f"{'system':<25}{'P':>8}{'R':>8}{'F1':>8}{'leak':>8}{'n':>6}")
    print('-' * 65)
    for r in summary:
        print(f"{r['system']:<25}{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}{r['leakage_rate']:>8.3f}{r['n_docs']:>6d}")
    return 0


def _infer_benchmark(inputs: list[str]) -> str:
    names = [Path(p).stem for p in inputs]
    for prefix in ("meddocan", "asq_phi"):
        if any(prefix in n for n in names):
            return prefix
    return "unknown"


def _infer_mode(inputs: list[str]) -> str:
    return "type"  # safe default; runner records mode per row if needed


if __name__ == "__main__":
    raise SystemExit(main())
