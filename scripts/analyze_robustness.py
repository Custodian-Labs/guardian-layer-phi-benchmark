#!/usr/bin/env python3
"""Robustness analyses for the masked-span equivalence result (paper v2).

Pure re-analysis of results/*.jsonl (no inference). Reuses the pairing and
masked-span definition of scripts/analyze_equivalence.py, and adds:

  R1  unit-of-analysis counts: unique masked spans, documents, outcomes
  R2  document-level cluster bootstrap (resample documents within benchmark),
      90% CI -> TOST verdict, smallest equivalence margin, design effect;
      plus a detector-equal-weighted variant
  R3  span selection: original-document recall on masked vs unmasked gold
      spans, raw and type-standardised, plus a type-reweighted estimate of the
      masked-span change projected onto the full gold population
  R4  matched C2 comparison: commercial vs Faker surrogates on the identical
      120-doc subsample / masked-span set / four detectors
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_equivalence import BENCH, found, load  # noqa: E402
from build_c1c2_corpora import category  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
B = 5000
SEED = 20261007
C1C2_KEY = {"ASQ-PHI": "asq_phi", "MEDDOCAN": "meddocan", "MultiCoNER v2": "multiconer_v2",
            "PII en": "pii_masking_300k", "PII nl": "pii_dutch", "PII fr": "pii_french",
            "PII de": "pii_german"}
C2_DET = {"deid_roberta_i2b2": "obi", "qwen3.5_9b": "qwen3_5_9b", "gemma_4_31b": "gemma_4_31b",
          "presidio": None}


def collect():
    """Per benchmark: list of documents, each a list of outcome rows
    (span_idx, system, masked, label, io, it). it is None for unmasked spans."""
    out = {}
    for name, (og, tg) in BENCH.items():
        do, dt = load(og), load(tg)
        docs = {}
        for did, d0 in do.items():
            d1 = dt.get(did)
            if not d1 or len(d0["gold"]) != len(d1["gold"]):
                continue
            systems = sorted(set(d0["preds"]) & set(d1["preds"]))
            rows = []
            for i, (sg, tg_) in enumerate(zip(d0["gold"], d1["gold"])):
                masked = sg.get("text") != tg_.get("text")
                cat = category(sg.get("label", ""))
                for s in systems:
                    io = found(sg, d0["preds"][s])
                    it = found(tg_, d1["preds"][s]) if masked else None
                    rows.append((i, s, masked, cat, io, it))
            docs[did] = rows
        out[name] = docs
    return out


# ---------------------------------------------------------------- R1 / R2
def doc_arrays(docs):
    """Per-document sums over masked outcomes: n, found_o, found_t."""
    ids, n, fo, ft = [], [], [], []
    for did, rows in docs.items():
        m = [(io, it) for (_, _, mk, _, io, it) in rows if mk]
        if not m:
            continue
        ids.append(did)
        n.append(len(m)); fo.append(sum(a for a, _ in m)); ft.append(sum(b for _, b in m))
    return ids, np.array(n), np.array(fo), np.array(ft)


def naive_se(docs):
    b = c = n = 0
    for rows in docs.values():
        for (_, _, mk, _, io, it) in rows:
            if mk:
                n += 1; b += io and not it; c += it and not io
    return np.sqrt(((b + c) - (b - c) ** 2 / n) / n ** 2) if n else np.nan


def boot_diffs(per_bench, rng):
    """Stratified cluster bootstrap: resample documents within each benchmark.
    Returns {bench: diffs[B]} and pooled diffs[B] (ratio estimator, pts)."""
    res, pool_n, pool_d = {}, np.zeros(B), np.zeros(B)
    for name, (n, fo, ft) in per_bench.items():
        k = len(n)
        idx = rng.integers(0, k, size=(B, k))
        sn, sd = n[idx].sum(1), (fo - ft)[idx].sum(1)
        res[name] = 100 * sd / sn
        pool_n += sn; pool_d += sd
    res["POOLED"] = 100 * pool_d / pool_n
    return res


def detector_weighted(docs_by_bench, rng):
    """Equal weight per detector: mean over detectors of per-detector paired
    change, pooled over benchmarks, with the same document resampling."""
    systems = sorted({s for docs in docs_by_bench.values() for rows in docs.values()
                      for (_, s, mk, _, _, _) in rows if mk})
    # per bench: doc x system arrays of (n, fo-ft)
    mats = {}
    for name, docs in docs_by_bench.items():
        ids = [d for d, rows in docs.items() if any(r[2] for r in rows)]
        N = np.zeros((len(ids), len(systems))); D = np.zeros_like(N)
        for a, did in enumerate(ids):
            for (_, s, mk, _, io, it) in docs[did]:
                if mk:
                    j = systems.index(s); N[a, j] += 1; D[a, j] += io - it
        mats[name] = (N, D)
    def stat(Ns, Ds):
        n = sum(Ns); d = sum(Ds)
        with np.errstate(invalid="ignore", divide="ignore"):
            return 100 * np.nanmean(d / n, axis=-1)
    point = stat([m[0].sum(0) for m in mats.values()], [m[1].sum(0) for m in mats.values()])
    bs_n, bs_d = [], []
    for N, D in mats.values():
        idx = rng.integers(0, N.shape[0], size=(B, N.shape[0]))
        bs_n.append(N[idx].sum(1)); bs_d.append(D[idx].sum(1))
    return point, stat(bs_n, bs_d)


def r1_r2(data):
    print("=" * 100)
    print("R1  Unit of analysis")
    print(f"{'benchmark':14s} {'docs':>5s} {'docs_w_mask':>11s} {'uniq_spans':>10s} "
          f"{'outcomes':>9s} {'det/span':>8s}")
    tot = defaultdict(int)
    per_bench = {}
    for name, docs in data.items():
        ids, n, fo, ft = doc_arrays(docs)
        uniq = sum(len({r[0] for r in rows if r[2]}) for rows in docs.values())
        per_bench[name] = (n, fo, ft)
        print(f"{name:14s} {len(docs):5d} {len(ids):11d} {uniq:10d} {n.sum():9d} {n.sum()/uniq:8.2f}")
        tot["docs"] += len(docs); tot["dm"] += len(ids); tot["u"] += uniq; tot["o"] += n.sum()
    print(f"{'POOLED':14s} {tot['docs']:5d} {tot['dm']:11d} {tot['u']:10d} {tot['o']:9d} "
          f"{tot['o']/tot['u']:8.2f}")

    rng = np.random.default_rng(SEED)
    bs = boot_diffs(per_bench, rng)
    print("\nR2  Document-level cluster bootstrap (B=%d, docs resampled within benchmark)" % B)
    print(f"{'benchmark':14s} {'diff':>7s} {'naive95':>16s} {'cluster95':>16s} {'cluster90':>16s} "
          f"{'DEFF':>5s} {'minΔ':>5s} {'Δ=2':>4s} {'Δ=3':>4s}")
    naive = {n: naive_se(d) for n, d in data.items()}
    allm = {k: v for k, v in data.items()}
    n_all = sum(v[0].sum() for v in per_bench.values())
    b_all = c_all = 0
    for docs in allm.values():
        for rows in docs.values():
            for (_, _, mk, _, io, it) in rows:
                if mk:
                    b_all += io and not it; c_all += it and not io
    naive["POOLED"] = np.sqrt(((b_all + c_all) - (b_all - c_all) ** 2 / n_all) / n_all ** 2)
    summary = {}
    for name in list(per_bench) + ["POOLED"]:
        if name == "POOLED":
            n = sum(v[0].sum() for v in per_bench.values())
            d = 100 * sum((v[1] - v[2]).sum() for v in per_bench.values()) / n
        else:
            n, fo, ft = per_bench[name]; d = 100 * (fo - ft).sum() / n.sum()
        s = bs[name]
        lo95, hi95 = np.percentile(s, [2.5, 97.5]); lo90, hi90 = np.percentile(s, [5, 95])
        se_n = 100 * naive[name]
        deff = (s.std() / se_n) ** 2
        mind = max(abs(lo90), abs(hi90))
        summary[name] = dict(diff=d, lo90=lo90, hi90=hi90, lo95=lo95, hi95=hi95, deff=deff, mind=mind)
        print(f"{name:14s} {d:+6.2f} [{d-1.96*se_n:+5.2f},{d+1.96*se_n:+5.2f}] "
              f"[{lo95:+5.2f},{hi95:+5.2f}] [{lo90:+5.2f},{hi90:+5.2f}] {deff:5.1f} {mind:5.2f} "
              f"{'✓' if mind < 2 else '✗':>4s} {'✓' if mind < 3 else '✗':>4s}")

    pt, bsd = detector_weighted(data, np.random.default_rng(SEED + 1))
    lo, hi = np.percentile(bsd, [5, 95])
    print(f"\nDetector-equal-weighted pooled change: {pt:+.2f} pts, cluster 90% CI [{lo:+.2f},{hi:+.2f}]")
    return summary


# ---------------------------------------------------------------- R3
def r3(data):
    print("\n" + "=" * 100)
    print("R3  Span selection: ORIGINAL-document recall, masked vs unmasked gold spans (overlap)")
    print(f"{'benchmark':14s} {'n_mask':>7s} {'rec_mask':>8s} {'n_unm':>7s} {'rec_unm':>8s} "
          f"{'gap':>6s} {'gap_typeStd':>11s}")
    pool = defaultdict(lambda: [0, 0, 0, 0])  # (bench,cat) -> n_m, f_m, n_u, f_u
    diff_cells = defaultdict(lambda: [0, 0])  # (bench,cat) -> n_masked outcomes, sum(io-it)
    gold_cells = defaultdict(int)            # (bench,cat) -> all gold outcomes
    tot = [0, 0, 0, 0]
    for name, docs in data.items():
        cell = defaultdict(lambda: [0, 0, 0, 0])
        for rows in docs.values():
            for (_, s, mk, cat, io, it) in rows:
                c = cell[cat]
                if mk:
                    c[0] += 1; c[1] += io
                    diff_cells[(name, cat)][0] += 1; diff_cells[(name, cat)][1] += io - it
                else:
                    c[2] += 1; c[3] += io
                gold_cells[(name, cat)] += 1
        nm = sum(c[0] for c in cell.values()); fm = sum(c[1] for c in cell.values())
        nu = sum(c[2] for c in cell.values()); fu = sum(c[3] for c in cell.values())
        # type-standardised: weight per-type recalls by the masked-span type mix
        w = {k: c[0] for k, c in cell.items() if c[0] and c[2]}
        W = sum(w.values())
        std_gap = (sum(w[k] * (cell[k][1] / cell[k][0] - cell[k][3] / cell[k][2]) for k in w) / W * 100
                   if W else float("nan"))
        rm = 100 * fm / nm if nm else float("nan"); ru = 100 * fu / nu if nu else float("nan")
        print(f"{name:14s} {nm:7d} {rm:7.1f}% {nu:7d} {ru:7.1f}% {rm-ru:+6.1f} {std_gap:+11.1f}")
        for k, c in cell.items():
            for j in range(4):
                pool[(name, k)][j] += c[j]
        tot = [a + b for a, b in zip(tot, (nm, fm, nu, fu))]
    print(f"{'POOLED':14s} {tot[0]:7d} {100*tot[1]/tot[0]:7.1f}% {tot[2]:7d} {100*tot[3]/tot[2]:7.1f}% "
          f"{100*(tot[1]/tot[0]-tot[3]/tot[2]):+6.1f}")
    w = {k: c[0] for k, c in pool.items() if c[0] and c[2]}
    W = sum(w.values())
    print("pooled type×benchmark-standardised gap: %+.1f pts" %
          (100 * sum(w[k] * (pool[k][1] / pool[k][0] - pool[k][3] / pool[k][2]) for k in w) / W))

    # Projected change if every gold span had been masked: reweight per-(bench,type)
    # masked-span change to the full gold composition (cells with masked data only).
    num = den = 0.0
    for k, g in gold_cells.items():
        if diff_cells[k][0]:
            num += g * diff_cells[k][1] / diff_cells[k][0]; den += g
    print("masked-span change reweighted to full gold type×benchmark mix: %+.2f pts "
          "(covers %.1f%% of gold outcomes)" % (100 * num / den, 100 * den / sum(gold_cells.values())))

    print("\n    per-type (pooled over benchmarks) original recall: masked vs unmasked")
    bytype = defaultdict(lambda: [0, 0, 0, 0])
    for (bn, cat), c in pool.items():
        for j in range(4):
            bytype[cat][j] += c[j]
    for cat, c in sorted(bytype.items(), key=lambda x: -x[1][0]):
        if c[0] and c[2]:
            print(f"    {cat:9s} masked {100*c[1]/c[0]:5.1f}% (n={c[0]:6d})  "
                  f"unmasked {100*c[3]/c[2]:5.1f}% (n={c[2]:6d})")


# ---------------------------------------------------------------- R4
def r4(data):
    print("\n" + "=" * 100)
    print("R4  Matched C2: same docs / masked spans / detectors — original vs commercial vs Faker (%)")
    print(f"{'benchmark':14s} {'detector':18s} {'N':>5s} {'orig':>6s} {'commercial':>10s} "
          f"{'Faker':>6s} {'F-C':>6s}")
    agg = defaultdict(lambda: [0, 0, 0, 0, 0])
    for name, docs in data.items():
        key = C1C2_KEY[name]
        fpath = os.path.join(ROOT, "data", "faker", f"{key}.jsonl")
        order = [json.loads(l)["doc_id"] for l in open(fpath) if l.strip()]
        for sysname, short in C2_DET.items():
            use = order if sysname == "presidio" else order[:120]
            if short is None:
                fj = None
                pres = {"asq_phi": 97.7, "meddocan": 79.7, "pii_masking_300k": 89.2,
                        "pii_dutch": 93.0, "pii_french": 78.5, "pii_german": 78.1}
                if key not in pres:
                    continue
                fk = pres[key]
            else:
                fj = json.load(open(os.path.join(ROOT, "results", "c1c2", f"faker_{key}_{short}.json")))
                fk = fj["recall_pct"]
            n = fo = ft = 0
            for did in use:
                for (_, s, mk, _, io, it) in docs.get(did, []):
                    if mk and s == sysname:
                        n += 1; fo += io; ft += it
            if not n:
                continue
            flag = "" if fj is None or fj["masked"] == n else f"  (C2 N={fj['masked']})"
            print(f"{name:14s} {sysname:18s} {n:5d} {100*fo/n:6.1f} {100*ft/n:10.1f} {fk:6.1f} "
                  f"{fk-100*ft/n:+6.1f}{flag}")
            a = agg[sysname]; a[0] += n; a[1] += fo; a[2] += ft; a[3] += fk * n / 100; a[4] += 1
    print("\n    N-weighted over benchmarks:")
    for s, a in agg.items():
        print(f"    {s:18s} orig {100*a[1]/a[0]:5.1f}  commercial {100*a[2]/a[0]:5.1f}  "
              f"Faker {100*a[3]/a[0]:5.1f}")


def main():
    data = collect()
    r1_r2(data)
    r3(data)
    r4(data)


if __name__ == "__main__":
    main()
