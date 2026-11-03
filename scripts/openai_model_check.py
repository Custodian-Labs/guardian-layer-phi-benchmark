#!/usr/bin/env python3
"""Which OpenAI model produced the stored `openai_gpt` predictions? (reproducibility audit D3)

Re-runs the first N ASQ-PHI original documents with the paper's prompt and parser
(systems/llm_openai_compatible: SYSTEM_PROMPT, json_object, _parse_spans) under
  gpt-4o-mini  temperature=0 (exactly the stored code path)
  gpt-5        default temperature (gpt-5 rejects temperature=0, which the code always sends)
and compares predicted span sets with the stored ones. Logs token usage and an
estimated cost; stops if the estimate exceeds --budget USD.
"""
import argparse, json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"), override=True)
from openai import OpenAI
from systems.llm_openai_compatible import SYSTEM_PROMPT, _parse_spans

PRICE = {"gpt-5": (1.25, 10.0), "gpt-4o-mini": (0.15, 0.60)}  # USD per 1M input/output tokens (list prices, approximate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budget", type=float, default=2.0)
    a = ap.parse_args()
    client = OpenAI()
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", "asq_250_openai.jsonl"))][: a.n]
    out = open(os.path.join(ROOT, "results", "openai_audit", "model_check.jsonl"), "w")
    cost = 0.0
    summary = {}
    for model in ("gpt-4o-mini", "gpt-5"):
        same = jac = 0.0
        for r in rows:
            kw = dict(model=model, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["text"]}],
                      response_format={"type": "json_object"})
            if model != "gpt-5":
                kw["temperature"] = 0
            resp = client.chat.completions.create(**kw)
            u = resp.usage
            cost += (u.prompt_tokens * PRICE[model][0] + u.completion_tokens * PRICE[model][1]) / 1e6
            new = {(s.start, s.end) for s in _parse_spans(resp.choices[0].message.content or "", r["text"])}
            old = {(s["start"], s["end"]) for s in r["predictions"]["openai_gpt"]["spans"]}
            same += new == old; jac += len(new & old) / max(1, len(new | old))
            out.write(json.dumps({"model": resp.model, "doc_id": r["doc_id"], "new": sorted(new), "stored": sorted(old),
                                  "usage": {"in": u.prompt_tokens, "out": u.completion_tokens}}) + "\n"); out.flush()
            if cost > a.budget:
                print(f"budget reached (${cost:.2f}); stopping"); break
        n = len(rows)
        summary[model] = {"identical_span_sets": int(same), "n": n, "mean_jaccard": round(jac / n, 3)}
        print(f"{model}: identical span sets {int(same)}/{n}, mean Jaccard {jac/n:.2f}  (cumulative est. cost ${cost:.2f})", flush=True)
    json.dump({"summary": summary, "est_cost_usd": round(cost, 3)},
              open(os.path.join(ROOT, "results", "openai_audit", "model_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
