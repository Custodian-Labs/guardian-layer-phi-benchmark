#!/usr/bin/env python3
"""E2 test-retest for the OpenAI detector (the paper's `openai_gpt`, which the
model check in results/openai_audit/ attributes to gpt-4o-mini, temperature 0).

For a stratified subset (first --per documents per benchmark that contain at least
one masked span), runs the paper's prompt/parser --runs times on the original and
on the transformed text, and stores every run's spans:
  results/openai_audit/retest_<model>.jsonl  {bench, doc_id, cond, run, model, spans, usage}
Resumable; stops when the estimated cost exceeds --budget USD.
Analysis: python scripts/retest_openai.py --analyze
  within-condition flip rate (masked span x run pairs whose found/missed differs),
  stored-vs-rerun agreement, and the paired original-vs-transformed change in the
  same subset, with document-cluster bootstrap CIs (scripts/paired_core.py).
"""
import argparse, itertools, json, os, random, sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(ROOT, "results", "openai_audit")
PRICE = {"gpt-5": (1.25, 10.0), "gpt-4o-mini": (0.15, 0.60)}


def subset(per):
    import paired_core as P
    keys = defaultdict(set)
    for r in P.paired_rows("snap_sym"):
        if r["det"] == "openai_gpt":
            keys[(r["bench"], r["doc"])].add(r["idx"])
    by = defaultdict(list)
    for (b, d) in sorted(keys):
        by[b].append(d)
    return {b: v[:per] for b, v in by.items()}, keys


def run(a):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"), override=True)
    from openai import OpenAI
    from systems.llm_openai_compatible import SYSTEM_PROMPT, _parse_spans
    import paired_core as P
    sub, _ = subset(a.per)
    texts = {b: (P._texts(P.BENCH[b][0]), P._texts(P.BENCH[b][1])) for b in sub}
    path = os.path.join(OUT, f"retest_{a.model}.jsonl")
    done = set()
    cost = 0.0
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done.add((r["bench"], r["doc_id"], r["cond"], r["run"]))
            cost += (r["usage"]["in"] * PRICE[a.model][0] + r["usage"]["out"] * PRICE[a.model][1]) / 1e6
    client = OpenAI()
    todo = [(run_i, b, d, cond) for run_i in range(a.runs) for b, docs in sub.items() for d in docs
            for cond in ("original", "transformed") if (b, d, cond, run_i) not in done]
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock, stop = threading.Lock(), threading.Event()

    def call(job):
        run_i, b, d, cond = job
        if stop.is_set():
            return None
        T = texts[b][0] if cond == "original" else texts[b][1]
        kw = dict(model=a.model, response_format={"type": "json_object"},
                  messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": T[d]}])
        if a.model != "gpt-5":
            kw["temperature"] = 0
        resp = client.chat.completions.create(**kw)
        spans = [{"start": s.start, "end": s.end, "label": s.label} for s in _parse_spans(resp.choices[0].message.content or "", T[d])]
        return job, resp.model, spans, resp.usage

    with open(path, "a") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(call, j) for j in todo]):
            out = fut.result()
            if out is None:
                continue
            (run_i, b, d, cond), model, spans, u = out
            with lock:
                cost += (u.prompt_tokens * PRICE[a.model][0] + u.completion_tokens * PRICE[a.model][1]) / 1e6
                f.write(json.dumps({"bench": b, "doc_id": d, "cond": cond, "run": run_i, "model": model,
                                    "spans": spans, "usage": {"in": u.prompt_tokens, "out": u.completion_tokens}}) + "\n")
                f.flush()
                if cost > a.budget and not stop.is_set():
                    print(f"budget reached (${cost:.2f}); finishing in-flight calls", flush=True)
                    stop.set()
    print(f"done, est. cost ${cost:.2f}", flush=True)

def analyze(a):
    import paired_core as P
    sub, _ = subset(a.per)
    rows = [json.loads(l) for l in open(os.path.join(OUT, f"retest_{a.model}.jsonl"))]
    preds = {(r["bench"], r["doc_id"], r["cond"], r["run"]): r["spans"] for r in rows}
    runs = sorted({r["run"] for r in rows})
    want = {(b, d) for b, v in sub.items() for d in v}
    pr = [r for r in P.paired_rows("snap_sym") if r["det"] == "openai_gpt" and (r["bench"], r["doc"]) in want]
    cache, texts = {}, {}
    for b, (og, tg) in P.BENCH.items():
        if b in sub:
            cache[b] = (P.load(og), P.load(tg))
            texts[b] = (P._texts(og), P._texts(tg))
    units = []  # per masked span: stored o/t found, and per run o/t found
    for r in pr:
        b, d, i = r["bench"], r["doc"], r["idx"]
        go, gt = cache[b][0][d]["gold"][i], cache[b][1][d]["gold"][i]
        ts, te = P.snap_sym(texts[b][0][d], go["start"], go["end"], texts[b][1][d], gt["start"], gt["end"])
        u = {"bench": b, "doc": d, "o_stored": r["o"]["overlap"], "t_stored": r["t"]["overlap"], "o": {}, "t": {}}
        for k in runs:
            if (b, d, "original", k) in preds and (b, d, "transformed", k) in preds:
                u["o"][k] = P.found(go["start"], go["end"], go["label"], preds[(b, d, "original", k)], "overlap")
                u["t"][k] = P.found(ts, te, gt["label"], preds[(b, d, "transformed", k)], "overlap")
        if u["o"]:
            units.append(u)
    n = len(units)
    pairs = list(itertools.combinations(runs, 2))
    flip_o = sum(u["o"][x] != u["o"][y] for u in units for x, y in pairs if x in u["o"] and y in u["o"])
    flip_t = sum(u["t"][x] != u["t"][y] for u in units for x, y in pairs if x in u["t"] and y in u["t"])
    npair = sum(1 for u in units for x, y in pairs if x in u["o"] and y in u["o"])
    stored_agree_o = sum(u["o"].get(0) == u["o_stored"] for u in units if 0 in u["o"])
    # paired change in the subset: stored, and per run
    def change(sel_o, sel_t):
        per_doc = defaultdict(lambda: [0, 0, 0])
        for u in units:
            v = per_doc[(u["bench"], u["doc"])]; v[0] += 1; v[1] += sel_o(u); v[2] += sel_t(u)
        strata = defaultdict(list)
        for (b, _), v in per_doc.items():
            strata[b].append(v)
        import numpy as np
        return P.cluster_stats({b: tuple(np.array(x) for x in zip(*v)) for b, v in strata.items()}, B=2000)
    res = {"model": a.model, "n_masked_spans": n, "runs": runs, "docs": len({(u['bench'], u['doc']) for u in units}),
           "within_condition_flip_rate_original": 100 * flip_o / max(1, npair),
           "within_condition_flip_rate_transformed": 100 * flip_t / max(1, npair),
           "rerun0_vs_stored_agreement_original": 100 * stored_agree_o / max(1, n),
           "paired_change_stored": change(lambda u: u["o_stored"], lambda u: u["t_stored"]),
           "paired_change_by_run": {k: change(lambda u, k=k: u["o"].get(k, False), lambda u, k=k: u["t"].get(k, False)) for k in runs}}
    # between-condition discordance per run, to compare with within-condition flips
    res["between_condition_discordance_by_run"] = {k: 100 * sum(u["o"].get(k) != u["t"].get(k) for u in units) / n for k in runs}
    json.dump(res, open(os.path.join(OUT, f"retest_{a.model}_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("paired_change")}, indent=1))
    for k, v in res["paired_change_by_run"].items():
        print(f"run {k}: {v['rec_o']:.1f} -> {v['rec_t']:.1f} ({v['diff']:+.2f}, 90% CI [{v['ci90'][0]:+.2f}, {v['ci90'][1]:+.2f}])")
    s = res["paired_change_stored"]
    print(f"stored: {s['rec_o']:.1f} -> {s['rec_t']:.1f} ({s['diff']:+.2f}, 90% CI [{s['ci90'][0]:+.2f}, {s['ci90'][1]:+.2f}])")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--per", type=int, default=30, help="documents per benchmark")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=1, help="concurrent API calls")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze(a) if a.analyze else run(a)
