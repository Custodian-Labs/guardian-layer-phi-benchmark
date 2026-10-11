#!/usr/bin/env python3
"""Run one detector over the non-medical domain benchmarks in every condition,
loading the model ONCE, and persist per-document predictions so that the paired
(original vs surrogate vs redacted) analysis in scripts/analyze_domains.py can
be done at the span level.

Conditions (corpora):
  original   data/domains/original/<bench>.jsonl   (scripts/build_domain_corpora.py)
  faker      data/domains/faker/<bench>.jsonl      (C2 open surrogate, scripts/build_domain_c1c2.py)
  redacted   data/domains/redacted/<bench>.jsonl   (C1 '*****' floor)
  custodian  data/transformed/<bench>.jsonl        (Custodian transform; scripts/transform_corpus.py,
                                                    needs a valid CUSTODIAN_SDK_API_KEY)

Output: results/domains/preds/<corpus>/<bench>__<system>.jsonl, one row per doc
  {doc_id, n_gold, spans:[{start,end,label,text}], error}
Resumable: doc_ids already present are skipped.

Usage:
  PYTHONPATH=. python scripts/run_domain_eval.py --system presidio
  PYTHONPATH=. python scripts/run_domain_eval.py --system qwen3_5_9b --batch-size 8 --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))

BENCHES = ["finance_gretel", "legal_tab", "enterprise_gretel"]
LANG = {"finance_gretel": "en", "legal_tab": "en", "enterprise_gretel": "en"}
CORPORA = ["original", "faker", "redacted", "custodian"]


def corpus_path(corpus: str, bench: str) -> str:
    if corpus == "custodian":
        return os.path.join(ROOT, "data", "transformed", f"{bench}.jsonl")
    return os.path.join(ROOT, "data", "domains", corpus, f"{bench}.jsonl")


def out_path(corpus: str, bench: str, system: str) -> str:
    return os.path.join(ROOT, "results", "domains", "preds", corpus, f"{bench}__{system}.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, help="any name accepted by scripts/run_benchmark.py")
    ap.add_argument("--benchmarks", nargs="+", default=BENCHES, choices=BENCHES)
    ap.add_argument("--corpora", nargs="+", default=["original", "faker", "redacted"], choices=CORPORA)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1, help="batched generation for local LLMs")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv; load_dotenv(os.path.join(ROOT, ".env"))
    except Exception:
        pass
    from run_benchmark import _build_system
    det = _build_system(args.system, LANG[args.benchmarks[0]])
    print(f"[init] system={args.system} -> {det.name}", flush=True)

    n_failed = 0
    for bench in args.benchmarks:
        for corpus in args.corpora:
            src = corpus_path(corpus, bench)
            if not os.path.exists(src):
                print(f"[skip] {corpus}/{bench}: {src} missing", flush=True)
                continue
            rows = [json.loads(l) for l in open(src) if l.strip()]
            if args.limit:
                rows = rows[: args.limit]
            op = out_path(corpus, bench, args.system)
            os.makedirs(os.path.dirname(op), exist_ok=True)
            done = set()
            if os.path.exists(op):
                # rows with an error (legacy: OOM docs were written as empty preds) are redone
                done = {r["doc_id"] for r in map(json.loads, filter(str.strip, open(op))) if not r.get("error")}
            todo = [r for r in rows if r["doc_id"] not in done]
            print(f"[run] {args.system} {corpus}/{bench}: {len(todo)}/{len(rows)} to do", flush=True)
            t0 = time.time()
            bs = max(1, args.batch_size)
            with open(op, "a") as fh:
                for k in range(0, len(todo), bs):
                    chunk = todo[k:k + bs]
                    texts = [r["text"] for r in chunk]
                    errs = [None] * len(chunk)
                    try:
                        if bs > 1 and hasattr(det, "predict_batch"):
                            preds = det.predict_batch(texts)
                        else:
                            preds = [det.predict(t) for t in texts]
                    except Exception as e:  # fall back to per-doc so one bad doc can't sink a batch
                        preds = []
                        for i, t in enumerate(texts):
                            try:
                                preds.append(det.predict(t))
                            except Exception as e2:
                                preds.append(None); errs[i] = f"{type(e2).__name__}: {e2}"[:300]
                        print(f"[warn] batch failed ({type(e).__name__}: {str(e)[:120]}); per-doc fallback", flush=True)
                    for r, p, err in zip(chunk, preds, errs):
                        if err is not None:
                            # never record a failed doc as an empty prediction: it would count as a
                            # miss and be skipped on resume. Leave it for the next (re)try instead.
                            n_failed += 1
                            continue
                        spans = [
                            {"start": s.start, "end": s.end, "label": s.label, "text": s.text} for s in p.spans]
                        fh.write(json.dumps({"doc_id": r["doc_id"], "n_gold": len(r["gold_spans"]),
                                             "spans": spans, "error": None,
                                             # raw LLM completion (debugging parse failures); None for non-LLMs
                                             "raw": (p.raw[:4000] if p is not None and isinstance(p.raw, str) else None)},
                                            ensure_ascii=False) + "\n")
                    fh.flush()
                    n = min(k + bs, len(todo))
                    if n % 25 < bs or n == len(todo):
                        el = time.time() - t0
                        print(f"  {args.system} {corpus}/{bench}: {n}/{len(todo)} {el:.0f}s "
                              f"({el/max(1,n):.2f}s/doc)", flush=True)
    if hasattr(det, "close"):
        try:
            det.close()
        except Exception:
            pass
    if n_failed:
        # non-zero exit -> a job runner can retry; resume redoes only these docs
        print(f"[fail] {n_failed} docs failed and were not recorded", flush=True)
        raise SystemExit(3)
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
