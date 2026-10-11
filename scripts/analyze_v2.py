#!/usr/bin/env python3
"""All numbers for the arXiv v2 revision, computed with the corrected
(word-boundary-snapped) gold-span re-projection.

Medical (7 benchmarks, 11 detectors) and non-medical (finance / legal /
enterprise, 10 detectors) masked-span analyses, cluster bootstrap, posterior
ROPE probabilities, per-detector retention, whole-document F1, lost-population
statistics. Writes results/v2_numbers.json. Pure re-analysis; no inference.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

import analyze_robustness as R  # noqa: E402
from analyze_equivalence import BENCH, found, load  # noqa: E402
from analyze_reprojection import cuts_word, exact, snap, snapped_load, texts  # noqa: E402
import analyze_domains as AD  # noqa: E402
from benchmarks.base import PHISpan  # noqa: E402
from evaluation.metrics import aggregate, score_document  # noqa: E402
from systems.base import PredictedSpan  # noqa: E402

B = 5000
OUT = {}


def boot_stats(per_doc, rng):
    """per_doc: dict stratum -> (n[], fo[], ft[]) arrays. Returns point, 95/90 CI,
    posterior P(|d|<m) via Bayesian bootstrap, min equivalence margin."""
    tot_n = sum(v[0].sum() for v in per_doc.values())
    d = 100 * sum((v[1] - v[2]).sum() for v in per_doc.values()) / tot_n
    cb_n = np.zeros(B); cb_d = np.zeros(B); bb_n = np.zeros(B); bb_d = np.zeros(B)
    for n, fo, ft in per_doc.values():
        k = len(n)
        idx = rng.integers(0, k, size=(B, k))
        cb_n += n[idx].sum(1); cb_d += (fo - ft)[idx].sum(1)
        w = rng.dirichlet(np.ones(k), size=B)
        bb_n += (w @ n) * k; bb_d += (w @ (fo - ft)) * k
    cb = 100 * cb_d / cb_n; bb = 100 * bb_d / bb_n
    lo95, hi95 = np.percentile(cb, [2.5, 97.5]); lo90, hi90 = np.percentile(cb, [5, 95])
    return {"n": int(tot_n), "diff": d, "ci95": [lo95, hi95], "ci90": [lo90, hi90],
            "min_margin": max(abs(lo90), abs(hi90)),
            "p_rope": {m: float(np.mean(np.abs(bb) < m)) for m in (1, 2, 3)}}


def arrays(rows_by_doc, sysfilter=None):
    n, fo, ft = [], [], []
    for rows in rows_by_doc.values():
        m = [(io, it) for (_, s, mk, _, io, it) in rows if mk and (sysfilter is None or s == sysfilter)]
        if m:
            n.append(len(m)); fo.append(sum(a for a, _ in m)); ft.append(sum(b for _, b in m))
    return np.array(n), np.array(fo), np.array(ft)


# ------------------------------------------------------------------ medical
def medical():
    R.load = snapped_load
    data = R.collect()
    rng = np.random.default_rng(20261011)
    med = {"bench": {}, "detector": {}}
    uniq = {}
    for name, docs in data.items():
        uniq[name] = sum(len({r[0] for r in rows if r[2]}) for rows in docs.values())
        n, fo, ft = arrays(docs)
        s = boot_stats({name: (n, fo, ft)}, rng)
        s.update(uniq=uniq[name], docs=int(len(n)), rec_o=100 * fo.sum() / n.sum(), rec_t=100 * ft.sum() / n.sum())
        med["bench"][name] = s
    s = boot_stats({k: arrays(v) for k, v in data.items()}, rng)
    n_all = sum(med["bench"][k]["n"] for k in data)
    fo_all = sum(med["bench"][k]["rec_o"] * med["bench"][k]["n"] for k in data) / n_all
    ft_all = sum(med["bench"][k]["rec_t"] * med["bench"][k]["n"] for k in data) / n_all
    s.update(uniq=sum(uniq.values()), docs=sum(med["bench"][k]["docs"] for k in data), rec_o=fo_all, rec_t=ft_all)
    med["pooled"] = s

    # per detector (overlap + boundary-exact on masked spans)
    systems = sorted({r[1] for docs in data.values() for rows in docs.values() for r in rows})
    ex = defaultdict(lambda: [0, 0, 0])
    for name, (og, tg) in BENCH.items():
        do, dt = load(og), snapped_load(tg)
        for did, d0 in do.items():
            d1 = dt.get(did)
            if not d1 or len(d0["gold"]) != len(d1["gold"]):
                continue
            for sg, tg_ in zip(d0["gold"], d1["gold"]):
                if sg.get("text") == tg_.get("text"):
                    continue
                for sname in set(d0["preds"]) & set(d1["preds"]):
                    e = ex[sname]; e[0] += 1
                    e[1] += exact(sg, d0["preds"][sname]); e[2] += exact(tg_, d1["preds"][sname])
    for sname in systems:
        st = boot_stats({k: arrays(v, sname) for k, v in data.items() if arrays(v, sname)[0].size}, rng)
        n = st["n"]
        fo = sum(arrays(v, sname)[1].sum() for v in data.values()); ft = sum(arrays(v, sname)[2].sum() for v in data.values())
        e = ex[sname]
        st.update(rec_o=100 * fo / n, rec_t=100 * ft / n,
                  ex_o=100 * e[1] / e[0], ex_t=100 * e[2] / e[0])
        med["detector"][sname] = st

    # lost population
    lost = Counter(); both = 0; same_len = 0; lost_n = 0
    for name, (og, tg) in BENCH.items():
        do, dt = load(og), snapped_load(tg)
        for did, d0 in do.items():
            d1 = dt.get(did)
            if not d1 or len(d0["gold"]) != len(d1["gold"]):
                continue
            for sg, tg_ in zip(d0["gold"], d1["gold"]):
                if sg.get("text") == tg_.get("text"):
                    continue
                cat = R.category(sg.get("label", ""))
                for sname in set(d0["preds"]) & set(d1["preds"]):
                    io, it = found(sg, d0["preds"][sname]), found(tg_, d1["preds"][sname])
                    if io and it:
                        both += 1
                    elif io and not it:
                        lost_n += 1; lost[cat] += 1
                        same_len += len(sg["text"]) == len(tg_["text"])
    med["lost"] = {"both": both, "lost": lost_n, "same_len_pct": 100 * same_len / lost_n,
                   "by_type_pct": {k: 100 * v / lost_n for k, v in lost.most_common()}}

    # whole-document F1 (boundary-exact, label-agnostic) and leakage, orig vs snapped transformed
    wd = defaultdict(dict)
    for name, (og, tg) in BENCH.items():
        do, dt = load(og), snapped_load(tg)
        for sname in systems:
            res = {}
            for cond, D in (("o", do), ("t", dt)):
                sc = []
                for did in do:
                    if did not in dt or sname not in do[did]["preds"] or sname not in dt[did]["preds"]:
                        continue
                    d = D[did]
                    gold = [PHISpan(g["start"], g["end"], g.get("label", ""), g.get("text", "")) for g in d["gold"]]
                    pr = [PredictedSpan(p["start"], p["end"], p.get("label", ""), p.get("text", "")) for p in d["preds"][sname]]
                    sc.append(score_document(gold, pr, mode="type"))
                if sc:
                    res[cond] = aggregate(sname, name, sc, "type")
            if len(res) == 2:
                wd[sname][name] = {"f1_o": res["o"].f1, "f1_t": res["t"].f1,
                                   "leak_o": res["o"].leakage_rate, "leak_t": res["t"].leakage_rate}
    med["wholedoc"] = wd
    OUT["medical"] = med


# ------------------------------------------------------------------ domains
def domains():
    rng = np.random.default_rng(20261012)
    dom = {"custodian": {}, "custodian_family": {}, "faker": {}, "redacted": {}, "cut_rate": {}}
    pooled = {"custodian": {}, "faker": {}, "redacted": {}}
    fam_pool = defaultdict(dict)
    for bench, bname in AD.BENCHES:
        O = {d["doc_id"]: d for d in AD.jl(os.path.join(ROOT, "data", "domains", "original", f"{bench}.jsonl"))}
        for arm in ("custodian", "faker", "redacted"):
            C = {d["doc_id"]: d for d in AD.corpus(bench, arm)}
            if arm == "custodian":
                cut = tot = 0
                for d in C.values():
                    new = []
                    for g in d["gold_spans"]:
                        tot += 1; cut += cuts_word(d["text"], g["start"], g["end"])
                        s, e = snap(d["text"], g["start"], g["end"])
                        new.append({**g, "start": s, "end": e, "text": d["text"][s:e]})
                    d["gold_spans"] = new
                dom["cut_rate"][bname] = 100 * cut / tot
            per_doc = defaultdict(lambda: [0, 0, 0]); per_fam = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
            for det, dname, fam in AD.DETECTORS:
                Po, Pa = AD.preds("original", bench, det), AD.preds(arm, bench, det)
                if not Po or not Pa or not (set(O) <= set(Po) and set(C) <= set(Pa)):
                    continue
                for did, cd in C.items():
                    od = O.get(did)
                    if od is None or len(od["gold_spans"]) != len(cd["gold_spans"]):
                        continue
                    for g, a in zip(od["gold_spans"], cd["gold_spans"]):
                        masked = a["masked"] if "masked" in a else (g["text"] != a["text"])
                        if not masked:
                            continue
                        x, y = AD.hit(g, Po[did]["spans"]), AD.hit(a, Pa[did]["spans"])
                        for acc in (per_doc[did], per_fam[fam][did]):
                            acc[0] += 1; acc[1] += x; acc[2] += y
            arr = lambda dd: tuple(np.array([v[i] for v in dd.values()]) for i in range(3))
            st = boot_stats({bench: arr(per_doc)}, rng)
            a = arr(per_doc)
            st.update(rec_o=100 * a[1].sum() / a[0].sum(), rec_t=100 * a[2].sum() / a[0].sum())
            dom[arm][bname] = st
            pooled[arm][bench] = arr(per_doc)
            if arm == "custodian":
                for fam, dd in per_fam.items():
                    fam_pool[fam][bench] = arr(dd)
                    f = arr(dd); sf = boot_stats({bench: f}, rng)
                    sf.update(rec_o=100 * f[1].sum() / f[0].sum(), rec_t=100 * f[2].sum() / f[0].sum())
                    dom["custodian_family"][f"{bname}/{fam}"] = sf
    for arm, strata in pooled.items():
        st = boot_stats(strata, rng)
        n = sum(v[0].sum() for v in strata.values())
        st.update(rec_o=100 * sum(v[1].sum() for v in strata.values()) / n,
                  rec_t=100 * sum(v[2].sum() for v in strata.values()) / n)
        dom[arm]["All"] = st
    for fam, strata in fam_pool.items():
        st = boot_stats(strata, rng)
        n = sum(v[0].sum() for v in strata.values())
        st.update(rec_o=100 * sum(v[1].sum() for v in strata.values()) / n,
                  rec_t=100 * sum(v[2].sum() for v in strata.values()) / n)
        dom["custodian_family"][f"All/{fam}"] = st
    OUT["domains"] = dom


def main():
    medical()
    domains()
    path = os.path.join(ROOT, "results", "v2_numbers.json")
    json.dump(OUT, open(path, "w"), indent=1, default=float)
    m = OUT["medical"]
    p = m["pooled"]
    print(f"MED pooled n={p['n']} uniq={p['uniq']} {p['rec_o']:.1f}->{p['rec_t']:.1f} d={p['diff']:+.2f} "
          f"ci95={np.round(p['ci95'],2)} ci90={np.round(p['ci90'],2)} minΔ={p['min_margin']:.2f} P={p['p_rope']}")
    for k, s in m["bench"].items():
        print(f"  {k:14s} uniq={s['uniq']:5d} n={s['n']:6d} {s['rec_o']:.1f}->{s['rec_t']:.1f} d={s['diff']:+.2f} "
              f"ci95=[{s['ci95'][0]:+.2f},{s['ci95'][1]:+.2f}] minΔ={s['min_margin']:.2f} P2={s['p_rope'][2]:.3f} P3={s['p_rope'][3]:.3f}")
    print("detectors:")
    for k, s in sorted(m["detector"].items(), key=lambda x: -x[1]["rec_o"]):
        print(f"  {k:20s} ov {s['rec_o']:.1f}->{s['rec_t']:.1f} d={s['diff']:+.2f} ci95=[{s['ci95'][0]:+.2f},{s['ci95'][1]:+.2f}] "
              f"ret={100*s['rec_t']/s['rec_o']:.1f} | exact {s['ex_o']:.1f}->{s['ex_t']:.1f}")
    print("lost:", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in m["lost"].items() if k != "by_type_pct"},
          {k: round(v, 1) for k, v in m["lost"]["by_type_pct"].items()})
    print("wholedoc (mean over benchmarks):")
    for sname, bb in m["wholedoc"].items():
        f = lambda key: np.mean([v[key] for v in bb.values()])
        print(f"  {sname:20s} F1 {f('f1_o'):.3f}->{f('f1_t'):.3f} ({f('f1_t')-f('f1_o'):+.3f}) "
              f"leak {f('leak_o'):.3f}->{f('leak_t'):.3f}  per-bench dF1 "
              + " ".join(f"{b[:6]}:{v['f1_t']-v['f1_o']:+.3f}" for b, v in bb.items()))
    d = OUT["domains"]
    print("DOMAINS cut-word rate (custodian re-projection):", {k: round(v, 1) for k, v in d["cut_rate"].items()})
    for arm in ("custodian", "faker", "redacted"):
        for k, s in d[arm].items():
            print(f"  {arm:9s} {k:10s} n={s['n']:6d} {s['rec_o']:.1f}->{s['rec_t']:.1f} d={s['diff']:+.2f} "
                  f"ci95=[{s['ci95'][0]:+.2f},{s['ci95'][1]:+.2f}] minΔ={s['min_margin']:.2f} P2={s['p_rope'][2]:.3f}")
    for k, s in d["custodian_family"].items():
        print(f"  fam {k:22s} n={s['n']:6d} {s['rec_o']:.1f}->{s['rec_t']:.1f} d={s['diff']:+.2f} "
              f"ci95=[{s['ci95'][0]:+.2f},{s['ci95'][1]:+.2f}] minΔ={s['min_margin']:.2f}")


if __name__ == "__main__":
    main()
