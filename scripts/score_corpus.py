#!/usr/bin/env python3
"""Score a redact/faker corpus with any detector, on the masked spans.

Loads the ORIGINAL subset (web/data/downloads/<orig>.jsonl) and a C1/C2 corpus
(data/redacted/<bench>.jsonl or data/faker/<bench>.jsonl), pairs by doc_id +
gold-span index, marks a span *masked* when its text differs between the two,
runs the detector on the corpus text, and reports masked-span recall (overlap).

Usage:
  PYTHONPATH=. python scripts/score_corpus.py --system obi --corpus redacted --benchmark asq_phi
"""
from __future__ import annotations
import argparse, json, os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
DL = os.path.join(ROOT, "web", "data", "downloads")

ORIG = {
    "asq_phi": ("asq_phi_250.jsonl", "en"), "meddocan": ("meddocan_250.jsonl", "es"),
    "multiconer_v2": ("multiconer_v2_250.jsonl", "en"), "pii_masking_300k": ("pii_masking_300k_250.jsonl", "en"),
    "pii_dutch": ("pii_masking_300k_dutch_250.jsonl", "nl"), "pii_french": ("pii_masking_300k_french_250.jsonl", "fr"),
    "pii_german": ("pii_masking_300k_german_250.jsonl", "de"),
}


def load(path):
    d = {}
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line); d[r["doc_id"]] = r
    return d


def overlap(a, b):
    return not (a[1] <= b[0] or a[0] >= b[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--corpus", choices=["redacted", "faker"])
    ap.add_argument("--benchmark", choices=list(ORIG))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="score both corpora x all benchmarks, loading the model once")
    args = ap.parse_args()

    LOCAL = {"qwen3_5_9b": "QWEN3_5_9B", "gemma_4_31b": "GEMMA_4_31B",
             "deepseek_v2_lite": "DEEPSEEK_V2_LITE", "llama3_3_70b": "LLAMA_3_3_70B"}
    if args.system in LOCAL:
        import systems.llm_local_hf as M
        sys_obj = M.LocalHFLLM(getattr(M, LOCAL[args.system]))
    else:
        try:
            from dotenv import load_dotenv; load_dotenv(os.path.join(ROOT, '.env'))
        except Exception:
            pass
        _lang = ORIG[args.benchmark][1] if args.benchmark else "en"
        from run_benchmark import _build_system
        sys_obj = _build_system(args.system, _lang)

    def score_one(corpus, bench):
        C = load(os.path.join(ROOT, "data", corpus, f"{bench}.jsonl"))
        n = found = 0
        docs = list(C.items())
        if args.limit:
            docs = docs[:args.limit]
        t0 = time.time()
        for k, (did, cd) in enumerate(docs):
            cg = cd.get("gold_spans", [])
            masked_idx = [i for i, g in enumerate(cg) if g.get("masked")]
            if not masked_idx:
                continue
            preds = sys_obj.predict(cd["text"]).spans
            pspans = [(p.start, p.end) for p in preds]
            for i in masked_idx:
                g = cg[i]; n += 1
                found += any(overlap((g["start"], g["end"]), ps) for ps in pspans)
            if (k + 1) % 50 == 0:
                print(f"  {bench}/{args.system}/{corpus}: {k+1}/{len(docs)} docs, {time.time()-t0:.0f}s", flush=True)
        rec = 100 * found / n if n else 0.0
        outdir = os.path.join(ROOT, "results", "c1c2"); os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, f"{corpus}_{bench}_{args.system}.json")
        json.dump({"system": args.system, "corpus": corpus, "benchmark": bench,
                   "masked": n, "found": found, "recall_pct": round(rec, 1)}, open(out, "w"), indent=2)
        print(f"RESULT {args.system} {corpus} {bench}: recall {rec:.1f}% ({found}/{n}) -> {out}", flush=True)

    if args.all:
        targets = [(c, b) for c in ("redacted", "faker") for b in ORIG]
    else:
        targets = [(args.corpus, args.benchmark)]
    for corpus, bench in targets:
        score_one(corpus, bench)
    if hasattr(sys_obj, "close"):
        try: sys_obj.close()
        except Exception: pass


if __name__ == "__main__":
    main()
