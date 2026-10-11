#!/usr/bin/env python3
"""Posterior probability that the masked-span recall change lies in a ROPE
(+-1, +-2, +-3 pts): stratified Bayesian bootstrap over documents (Rubin 1981),
i.e. Dirichlet(1,...,1) weights on documents within each benchmark.
Run with --snap to use the word-boundary-snapped re-projection."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import analyze_robustness as R

B = 20000
if "--snap" in sys.argv:
    from analyze_reprojection import snapped_load
    R.load = snapped_load
data = R.collect()
rng = np.random.default_rng(20261007)
per = {k: R.doc_arrays(v)[1:] for k, v in data.items()}
pool_n = np.zeros(B); pool_d = np.zeros(B)
print(f"{'benchmark':14s} {'mean':>6s} {'95% CrI':>16s} {'P(|Δ|<1)':>9s} {'P(|Δ|<2)':>9s} {'P(|Δ|<3)':>9s}")
rows = {}
for name, (n, fo, ft) in per.items():
    w = rng.dirichlet(np.ones(len(n)), size=B)
    sn, sd = w @ n, w @ (fo - ft)
    rows[name] = 100 * sd / sn
    # pooled: weight each benchmark by its observed total so strata keep their size
    pool_n += sn * n.sum() / sn.mean(); pool_d += sd * n.sum() / sn.mean()
rows["POOLED"] = 100 * pool_d / pool_n
for name, s in rows.items():
    lo, hi = np.percentile(s, [2.5, 97.5])
    p = [np.mean(np.abs(s) < m) for m in (1, 2, 3)]
    print(f"{name:14s} {s.mean():+6.2f} [{lo:+6.2f},{hi:+6.2f}] {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f}")
