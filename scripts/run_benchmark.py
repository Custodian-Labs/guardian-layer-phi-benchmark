"""Main entrypoint.

Examples
--------
# Just the cheapest baseline against ASQ-PHI
python scripts/run_benchmark.py --benchmark asq_phi --systems presidio

# Custodian vs every LLM, ASQ-PHI test split
python scripts/run_benchmark.py --benchmark asq_phi \
    --systems custodian openai deepseek kimi qwen gemma

# Spanish track, transformer + LLMs
python scripts/run_benchmark.py --benchmark meddocan --split test \
    --systems presidio openai qwen
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import REGISTRY as BENCH_REGISTRY  # noqa: E402
from evaluation.metrics import aggregate, score_document  # noqa: E402


def _build_system(name: str, language: str):
    """Lazy import so a single missing dependency doesn't crash the whole CLI."""
    if name == "custodian":
        from systems.custodian import Custodian, CustodianConfig
        import os
        return Custodian(CustodianConfig(
            compliance_mode=os.environ.get("CUSTODIAN_COMPLIANCE_MODE", "PROPRIETARY"),  # type: ignore[arg-type]
            masking_type=os.environ.get("CUSTODIAN_MASKING_TYPE", "transform"),          # type: ignore[arg-type]
            domain=os.environ.get("CUSTODIAN_DOMAIN", "General"),
        ))
    if name == "custodian_all":
        from systems.custodian import build_variants
        return build_variants()  # returns a list — handled below
    if name.startswith("custodian:"):
        # Syntax: custodian:MASKED:redact  or  custodian:PROPRIETARY:transform
        from systems.custodian import Custodian, CustodianConfig
        _, mode, masking = name.split(":")
        return Custodian(CustodianConfig(
            compliance_mode=mode,  # type: ignore[arg-type]
            masking_type=masking,  # type: ignore[arg-type]
        ))
    if name == "presidio":
        from systems.presidio import Presidio
        return Presidio(language=language)
    if name == "obi":
        from systems.obi_deid import OBIDeID
        return OBIDeID()
    if name == "philter":
        from systems.philter import Philter
        return Philter()
    if name == "jsl":
        from systems.johnsnow import JohnSnowLabs
        return JohnSnowLabs()
    if name == "openai":
        from systems.llm_openai_compatible import make_llm, OPENAI
        return make_llm(OPENAI)
    if name == "deepseek":
        from systems.llm_openai_compatible import make_llm, DEEPSEEK
        return make_llm(DEEPSEEK)
    if name == "kimi":
        from systems.llm_openai_compatible import make_llm, KIMI
        return make_llm(KIMI)
    if name == "qwen":
        from systems.llm_openai_compatible import make_llm, QWEN
        return make_llm(QWEN)
    if name == "gemma":
        from systems.llm_gemma import GemmaAIStudio
        return GemmaAIStudio()
    if name == "gemma_e4b":
        from systems.llm_local_hf import LocalHFLLM, GEMMA_E4B
        return LocalHFLLM(GEMMA_E4B)
    if name == "qwen3_35b":
        from systems.llm_local_hf import LocalHFLLM, QWEN3_35B_A3B
        return LocalHFLLM(QWEN3_35B_A3B)
    if name == "moonlight":
        from systems.llm_local_hf import LocalHFLLM, MOONLIGHT
        return LocalHFLLM(MOONLIGHT)
    if name == "deepseek_v2_lite":
        from systems.llm_local_hf import LocalHFLLM, DEEPSEEK_V2_LITE
        return LocalHFLLM(DEEPSEEK_V2_LITE)
    if name == "kimi_vl":
        from systems.llm_local_hf import LocalHFLLM, KIMI_VL_A3B
        return LocalHFLLM(KIMI_VL_A3B)
    if name == "qwen3_5_4b":
        from systems.llm_local_hf import LocalHFLLM, QWEN3_5_4B
        return LocalHFLLM(QWEN3_5_4B)
    if name == "qwen3_5_4b_thinking":
        from systems.llm_local_hf import LocalHFLLM, QWEN3_5_4B_THINKING
        return LocalHFLLM(QWEN3_5_4B_THINKING)
    if name == "qwen3_5_9b":
        from systems.llm_local_hf import LocalHFLLM, QWEN3_5_9B
        return LocalHFLLM(QWEN3_5_9B)
    if name == "qwen3_5_35b":
        from systems.llm_local_hf import LocalHFLLM, QWEN3_5_35B_A3B
        return LocalHFLLM(QWEN3_5_35B_A3B)
    if name == "gemma_4_31b":
        from systems.llm_local_hf import LocalHFLLM, GEMMA_4_31B
        return LocalHFLLM(GEMMA_4_31B)
    if name == "llama3_1_8b":
        from systems.llm_local_hf import LocalHFLLM, LLAMA_3_1_8B
        return LocalHFLLM(LLAMA_3_1_8B)
    if name == "llama3_3_70b":
        from systems.llm_local_hf import LocalHFLLM, LLAMA_3_3_70B
        return LocalHFLLM(LLAMA_3_3_70B)
    raise ValueError(f"unknown system: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=list(BENCH_REGISTRY))
    parser.add_argument(
        "--systems", nargs="+", required=True,
        help="custodian presidio obi philter jsl openai deepseek kimi qwen gemma",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", default=None,
                        help="Defaults to data/<benchmark>/ under repo root.")
    parser.add_argument("--mode", default="type", choices=["strict", "type", "relaxed"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of docs (sanity-check runs).")
    parser.add_argument("--out", default=None,
                        help="Output JSONL path; defaults to results/<bench>_<ts>.jsonl")
    parser.add_argument("--include-text", action="store_true",
                        help="Persist doc.text and gold_spans in per-doc JSONL "
                             "(needed for the web drilldown). Only safe for "
                             "synthetic / open-license corpora.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    bench_cls = BENCH_REGISTRY[args.benchmark]
    data_root = Path(args.data_root) if args.data_root else ROOT / "data" / args.benchmark
    bench = (
        bench_cls(root=data_root, split=args.split)
        if "split" in bench_cls.__init__.__code__.co_varnames
        else bench_cls(root=data_root)
    )

    ts = int(time.time())
    out_path = Path(args.out) if args.out else ROOT / "results" / f"{args.benchmark}_{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[run] benchmark={args.benchmark} split={args.split} language={bench.language}")

    # Build systems up front so credential errors surface immediately.
    systems = []
    for s in args.systems:
        try:
            built = _build_system(s, bench.language)
            if isinstance(built, list):
                systems.extend(built)
                print(f"[ok]  system '{s}' expanded to {len(built)} variants")
            else:
                systems.append(built)
                print(f"[ok]  system '{s}' ready")
        except Exception as e:
            print(f"[skip] system '{s}' failed to init: {e}")

    if not systems:
        print("no systems initialized; aborting", file=sys.stderr)
        return 1

    # Stream documents through each system.
    per_system_scores = {sys.name: [] for sys in systems}
    docs = list(bench)
    if args.limit:
        docs = docs[:args.limit]
    print(f"[run] {len(docs)} documents")

    with out_path.open("w") as fh:
        for doc in tqdm(docs):
            row = {"doc_id": doc.doc_id, "gold_n": len(doc.gold_spans), "predictions": {}}
            if args.include_text:
                row["text"] = doc.text
                row["gold_spans"] = [s.__dict__ for s in doc.gold_spans]
            for sys_ in systems:
                try:
                    pred = sys_.predict(doc.text)
                except Exception as e:
                    print(f"[err] {sys_.name} on {doc.doc_id}: {e}", file=sys.stderr)
                    pred = None
                if pred is None:
                    continue
                score = score_document(doc.gold_spans, pred.spans, mode=args.mode)
                per_system_scores[sys_.name].append(score)
                row["predictions"][sys_.name] = {
                    "spans": [s.__dict__ for s in pred.spans],
                    "score": score.__dict__,
                }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n=== Aggregate ===")
    summary = []
    for sys_ in systems:
        agg = aggregate(sys_.name, args.benchmark, per_system_scores[sys_.name], args.mode)
        summary.append(agg.to_dict())
        print(
            f"{sys_.name:<20}  P={agg.precision:.3f}  R={agg.recall:.3f}  "
            f"F1={agg.f1:.3f}  leak={agg.leakage_rate:.3f}  "
            f"char_leak={agg.char_leakage_rate:.3f}  (n={agg.n_docs})"
        )

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDetails: {out_path}\nSummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
