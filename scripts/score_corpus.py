#!/usr/bin/env python3
"""Score a redact/faker corpus with any detector, on the masked spans.

Loads the ORIGINAL subset (web/data/downloads/<orig>.jsonl) and a C1/C2 corpus
(data/redacted/<bench>.jsonl or data/faker/<bench>.jsonl), pairs by doc_id +
gold-span index, marks a span *masked* when its text differs between the two,
runs the detector on the corpus text, and reports masked-span recall (overlap).

Usage:
  PYTHONPATH=. python scripts/score_corpus.py --system obi --corpus redacted --benchmark asq_phi
  # C1 with a detector told what the mask token means (local LLMs only):
  PYTHONPATH=. python scripts/score_corpus.py --system qwen3_5_9b --all --corpora redacted --prompt mask_aware
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
# Non-medical domains (scripts/build_domain_corpora.py + build_domain_c1c2.py).
# Corpora live under data/domains/<corpus>/; not part of --all (use
# scripts/run_domain_eval.py for those, which also keeps per-doc predictions).
DOMAIN = {"finance_gretel": "en", "legal_tab": "en", "enterprise_gretel": "en"}


# Appended to the detector prompt for --prompt mask_aware. Redaction writes '*' x len
# over each masked value, so a detector that knows the convention can find every one.
MASK_AWARE_NOTE = """

Note: this text has already been partly redacted. Every run of asterisks
(for example *****) is a placeholder where a PHI value was removed. Report each
such run as a span (label OTHER, text = the asterisks), in addition to any PHI
that is still present."""


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
    ap.add_argument("--corpus", choices=["redacted", "faker", "original"])
    ap.add_argument("--benchmark", choices=list(ORIG) + list(DOMAIN))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="score both corpora x all benchmarks, loading the model once")
    ap.add_argument("--exclude", default="", help="comma-separated benchmarks to skip in --all")
    ap.add_argument("--force", action="store_true", help="recompute even if a result json exists")
    ap.add_argument("--corpora", default="redacted,faker", help="corpora scored by --all")
    ap.add_argument("--save-preds", action="store_true",
                    help="also write per-span outcomes to results/c1c2/preds/<tag>.jsonl (gold-span index, found)")
    ap.add_argument("--prompt", choices=["default", "mask_aware"], default="default",
                    help="mask_aware: the LLM prompt says runs of '*' are removed PHI (C1 for a mask-adapted detector)")
    args = ap.parse_args()

    LOCAL = {"qwen3_5_9b": "QWEN3_5_9B", "gemma_4_31b": "GEMMA_4_31B",
             "deepseek_v2_lite": "DEEPSEEK_V2_LITE", "llama3_3_70b": "LLAMA_3_3_70B"}
    if args.system in LOCAL:
        import dataclasses
        import systems.llm_local_hf as M
        cfg = getattr(M, LOCAL[args.system])
        if args.prompt == "mask_aware":
            cfg = dataclasses.replace(cfg, system_prompt_override=M.SYSTEM_PROMPT + MASK_AWARE_NOTE)
        sys_obj = M.LocalHFLLM(cfg)
    else:
        if args.prompt != "default":
            ap.error("--prompt mask_aware is only implemented for local LLM detectors")
        try:
            from dotenv import load_dotenv; load_dotenv(os.path.join(ROOT, '.env'))
        except Exception:
            pass
        _lang = (ORIG[args.benchmark][1] if args.benchmark in ORIG else DOMAIN[args.benchmark]) if args.benchmark else "en"
        from run_benchmark import _build_system
        sys_obj = _build_system(args.system, _lang)

    def score_one(corpus, bench):
        tag = f"{corpus}_{bench}_{args.system}" + ("" if args.prompt == "default" else f"_{args.prompt}")
        _out = os.path.join(ROOT, "results", "c1c2", f"{tag}.json")
        if os.path.exists(_out) and not args.force:
            print(f"SKIP {args.system} {corpus} {bench} (already exists -> skip)", flush=True); return
        C = load(os.path.join(ROOT, "data", *(("domains",) if bench in DOMAIN else ()), corpus, f"{bench}.jsonl"))
        n = found = 0
        per_doc = []
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
            hits = []
            for i in masked_idx:
                g = cg[i]; n += 1
                hits.append(int(any(overlap((g["start"], g["end"]), ps) for ps in pspans)))
            found += sum(hits)
            per_doc.append({"doc_id": did, "masked_idx": masked_idx, "found": hits})
            if (k + 1) % 50 == 0:
                print(f"  {bench}/{args.system}/{corpus}: {k+1}/{len(docs)} docs, {time.time()-t0:.0f}s", flush=True)
        rec = 100 * found / n if n else 0.0
        outdir = os.path.join(ROOT, "results", "c1c2"); os.makedirs(outdir, exist_ok=True)
        out = _out
        json.dump({"system": args.system, "prompt": args.prompt, "corpus": corpus, "benchmark": bench,
                   "masked": n, "found": found, "recall_pct": round(rec, 1)}, open(out, "w"), indent=2)
        if args.save_preds:
            os.makedirs(os.path.join(outdir, "preds"), exist_ok=True)
            with open(os.path.join(outdir, "preds", f"{tag}.jsonl"), "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in per_doc)
        print(f"RESULT {args.system} {corpus} {bench}: recall {rec:.1f}% ({found}/{n}) -> {out}", flush=True)

    if args.all:
        _skip = set(x for x in args.exclude.split(",") if x)
        targets = [(c, b) for c in args.corpora.split(",") for b in ORIG if b not in _skip]
    else:
        targets = [(args.corpus, args.benchmark)]
    for corpus, bench in targets:
        score_one(corpus, bench)
    if hasattr(sys_obj, "close"):
        try: sys_obj.close()
        except Exception: pass


if __name__ == "__main__":
    main()
