#!/usr/bin/env python3
"""E3 (matching / re-projection robustness) and E6 (equivalence-margin analysis).

Pure re-analysis of results/*.jsonl through scripts/paired_core.py; no inference.

E3  paired masked-span change under 2 re-projections x 4 matching criteria,
    pooled and per benchmark, with stratified document-level cluster bootstrap,
    TOST at 1/2/3 points and the smallest passing margin; type-aware vs
    type-agnostic; mapping-quality statistics and examples.
E6  smallest passing margin and posterior P(|diff| < m) per benchmark and per
    detector; margin curve (fraction of benchmarks / detectors equivalent at m);
    second-pass miss-rate framing.

Writes results/robustness/{robustness.json, tables.md, margin_curve.png, margin_curve.pdf}.
Usage: python scripts/analyze_robustness.py [--B 5000]
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import paired_core as P  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT = os.path.join(ROOT, "results", "robustness")
BENCHES = list(P.BENCH)
DET_NAME = {"presidio": "Presidio", "deid_roberta_i2b2": "OBI RoBERTa", "openai_gpt": "OpenAI GPT*",
            "gemma_4_e4b": "Gemma 4 E4B", "gemma_4_31b": "Gemma 4 31B", "qwen3.5_4b": "Qwen 3.5 4B",
            "qwen3.5_9b": "Qwen 3.5 9B", "qwen3.5_35b_a3b": "Qwen 3.5 35B-A3B", "deepseek_v2_lite": "DeepSeek V2-Lite",
            "llama3.1_8b": "Llama 3.1 8B", "llama3.3_70b": "Llama 3.3 70B"}
MARGINS = [round(0.25 * k, 2) for k in range(1, 25)]  # 0.25 .. 6.0


def fmt(s):
    lo, hi = s["ci90"]
    return f"{s['rec_o']:.1f} → {s['rec_t']:.1f} | {s['diff']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {s['min_margin']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=5000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cache, rows = {}, {}
    for rp in P.REPROJ:
        P.SKIPPED.clear()
        rows[rp] = list(P.paired_rows(rp, cache=cache))
        print(f"[{rp}] {len(rows[rp])} outcomes; skipped {dict(P.SKIPPED)}", flush=True)
    res = {"meta": {"B": args.B, "seed": 0, "skipped_snap_sym": {f"{k[0]}|{k[1]}": v for k, v in P.SKIPPED.items()}}}
    md = ["# Robustness of the paired masked-span change (E3, E6)", "",
          "diff = recall(original) − recall(transformed), points (positive = loss). 90% CI: stratified "
          f"document-level cluster bootstrap (B = {args.B}). Smallest margin = smallest m with the 90% CI inside (−m, m) "
          "(TOST, α = 0.05). `difflib` reproduces the paper; `snap_sym` is the corrected re-projection "
          "(see scripts/paired_core.py).", ""]

    # ---------------- E3: re-projection x criterion, pooled + per benchmark
    md += ["## E3. Re-projection × matching criterion (pooled, all detectors)", "",
           "| Re-projection | Criterion | Spans × det. | Docs | Recall orig → transf. | Diff | 90% CI | Smallest margin | ±2 TOST |",
           "|---|---|--:|--:|--|--:|--|--:|:-:|"]
    e3 = {}
    for rp in P.REPROJ:
        for c in P.CRITERIA:
            s = P.cluster_stats(P.per_doc_arrays(rows[rp], c), B=args.B)
            e3[f"{rp}|{c}|pooled"] = s
            lo, hi = s["ci90"]
            md.append(f"| {rp} | {c} | {s['n']:,} | {s['docs']:,} | {s['rec_o']:.1f} → {s['rec_t']:.1f} | {s['diff']:+.2f} | "
                      f"[{lo:+.2f}, {hi:+.2f}] | {s['min_margin']:.2f} | {'✓' if s['tost'][2] else '✗'} |")
            for b in BENCHES:
                pd = {k: v for k, v in P.per_doc_arrays(rows[rp], c).items() if k == b}
                if pd:
                    e3[f"{rp}|{c}|{b}"] = P.cluster_stats(pd, B=args.B)
    md += ["", "### Per benchmark: smallest passing margin (points) by re-projection × criterion", "",
           "| Benchmark | " + " | ".join(f"{rp} {c}" for rp in P.REPROJ for c in P.CRITERIA) + " |",
           "|---|" + "--:|" * (len(P.REPROJ) * len(P.CRITERIA))]
    for b in BENCHES:
        md.append(f"| {b} | " + " | ".join(
            f"{e3[f'{rp}|{c}|{b}']['diff']:+.2f} ({e3[f'{rp}|{c}|{b}']['min_margin']:.2f})" if f"{rp}|{c}|{b}" in e3 else "–"
            for rp in P.REPROJ for c in P.CRITERIA) + " |")
    md += ["", "Cells: diff (smallest passing margin).", ""]
    res["e3"] = e3

    # type-aware vs agnostic by gold type (snap_sym)
    md += ["### Exact boundary: type-agnostic vs type-aware, by gold type (snap_sym)", "",
           "| Gold type | Spans × det. | exact: orig → transf. (diff) | exact_type: orig → transf. (diff) |", "|---|--:|--|--|"]
    bytype = {}
    for t in P.TYPES:
        sub = [r for r in rows["snap_sym"] if r["gtype"] == t]
        if len(sub) < 50:
            continue
        cells = []
        for c in ("exact", "exact_type"):
            n = len(sub); o = sum(r["o"][c] for r in sub); tt = sum(r["t"][c] for r in sub)
            cells.append((100 * o / n, 100 * tt / n))
            bytype[f"{t}|{c}"] = {"n": n, "rec_o": 100 * o / n, "rec_t": 100 * tt / n}
        md.append(f"| {t} | {len(sub):,} | {cells[0][0]:.1f} → {cells[0][1]:.1f} ({cells[0][0]-cells[0][1]:+.1f}) | "
                  f"{cells[1][0]:.1f} → {cells[1][1]:.1f} ({cells[1][0]-cells[1][1]:+.1f}) |")
    res["by_type"] = bytype

    # mapping quality
    uniq = {rp: {(r["bench"], r["doc"], r["idx"]): (r["orig_text"], r["surr_text"]) for r in rows[rp]} for rp in P.REPROJ}
    only_snap = sorted(set(uniq["snap_sym"]) - set(uniq["difflib"]))
    only_diff = sorted(set(uniq["difflib"]) - set(uniq["snap_sym"]))
    changed = [k for k in set(uniq["difflib"]) & set(uniq["snap_sym"]) if uniq["difflib"][k][1] != uniq["snap_sym"][k][1]]
    res["mapping"] = {"masked_difflib": len(uniq["difflib"]), "masked_snap_sym": len(uniq["snap_sym"]),
                      "masked_only_after_snapping": len(only_snap), "masked_only_under_difflib": len(only_diff),
                      "surrogate_boundary_changed_by_snapping": len(changed),
                      "by_bench_changed": Counter(k[0] for k in changed),
                      "examples_changed": [(k[0], uniq["difflib"][k][1], uniq["snap_sym"][k][1]) for k in changed[:12]],
                      "examples_only_snap": [(k[0], uniq["snap_sym"][k][0], uniq["snap_sym"][k][1]) for k in only_snap[:12]]}
    m = res["mapping"]
    md += ["", "### Mapping quality (unique masked spans)", "",
           f"- Masked under difflib: {m['masked_difflib']:,}; under snap_sym: {m['masked_snap_sym']:,}.",
           f"- Re-projected surrogate boundary moved by snapping (difflib cut inside a word): {m['surrogate_boundary_changed_by_snapping']:,} "
           f"({100*m['surrogate_boundary_changed_by_snapping']/m['masked_difflib']:.1f}%); by benchmark {dict(m['by_bench_changed'])}.",
           f"- Replacements difflib missed (masked only after snapping): {m['masked_only_after_snapping']}; "
           f"masked only under difflib (gold text ≠ document, excluded from snap_sym): {m['masked_only_under_difflib']}.",
           "- Examples, difflib → snapped surrogate: " + "; ".join(f"`{a}` → `{b}`" for _, a, b in m["examples_changed"][:8]),
           "- Examples of replacements difflib missed, original → surrogate: " + "; ".join(f"`{a}` → `{b}`" for _, a, b in m["examples_only_snap"][:8]),
           "", "Snapping is a heuristic correction; it is not the transform's own replacement offsets.", ""]

    # ---------------- E6: margins per benchmark and per detector
    md += ["## E6. Equivalence margins (overlap criterion)", ""]
    e6 = {}
    for rp in P.REPROJ:
        md += [f"### {rp}: per benchmark", "", "| Benchmark | Spans × det. | Recall orig → transf. | Diff | 90% CI | Smallest margin | P(<1) | P(<2) | P(<3) |",
               "|---|--:|--|--:|--|--:|--:|--:|--:|"]
        for b in BENCHES + ["POOLED"]:
            s = e3[f"{rp}|overlap|{b if b != 'POOLED' else 'pooled'}"]
            e6[f"{rp}|bench|{b}"] = s
            lo, hi = s["ci90"]
            md.append(f"| {b} | {s['n']:,} | {s['rec_o']:.1f} → {s['rec_t']:.1f} | {s['diff']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                      f"{s['min_margin']:.2f} | {s['p_rope'][1]:.2f} | {s['p_rope'][2]:.2f} | {s['p_rope'][3]:.2f} |")
        md += ["", f"### {rp}: per detector (pooled over benchmarks)", "",
               "| Detector | Spans × det. | Recall orig → transf. | Diff | 90% CI | Smallest margin | P(<2) |", "|---|--:|--|--:|--|--:|--:|"]
        for det in sorted({r["det"] for r in rows[rp]}):
            s = P.cluster_stats(P.per_doc_arrays(rows[rp], "overlap", keep=lambda r, d=det: r["det"] == d), B=args.B)
            e6[f"{rp}|det|{det}"] = s
            lo, hi = s["ci90"]
            md.append(f"| {DET_NAME.get(det, det)} | {s['n']:,} | {s['rec_o']:.1f} → {s['rec_t']:.1f} | {s['diff']:+.2f} | "
                      f"[{lo:+.2f}, {hi:+.2f}] | {s['min_margin']:.2f} | {s['p_rope'][2]:.2f} |")
        md.append("")
    res["e6"] = e6
    md += ["\\* `openai_gpt`: the paper reports GPT-5; which OpenAI model produced these runs is under audit "
           "(see docs/reproducibility_audit.md).", ""]

    # margin curve
    curve = {}
    for rp in P.REPROJ:
        bm = [e6[f"{rp}|bench|{b}"]["min_margin"] for b in BENCHES]
        dm = [v["min_margin"] for k, v in e6.items() if k.startswith(f"{rp}|det|")]
        curve[rp] = {"margins": MARGINS, "pooled_pass": [e6[f"{rp}|bench|POOLED"]["min_margin"] < m for m in MARGINS],
                     "frac_bench": [sum(x < m for x in bm) / len(bm) for m in MARGINS],
                     "frac_det": [sum(x < m for x in dm) / len(dm) for m in MARGINS]}
    res["margin_curve"] = curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        c = curve["snap_sym"]
        ax.step(MARGINS, [100 * x for x in c["frac_bench"]], where="post", label="benchmarks (7)", color="#0f766e")
        ax.step(MARGINS, [100 * x for x in c["frac_det"]], where="post", label="detectors (11)", color="#7c3aed", linestyle="--")
        ax.axvline(e6["snap_sym|bench|POOLED"]["min_margin"], color="#475569", lw=1, linestyle=":", label="pooled")
        ax.axvline(2, color="#b91c1c", lw=0.8, alpha=.6)
        ax.set_xlabel("equivalence margin m (recall points)"); ax.set_ylabel("% equivalent (TOST, 90% CI)")
        ax.set_ylim(0, 102); ax.set_xlim(0, 6); ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT, f"margin_curve.{ext}"), dpi=200)
        md += ["![margin curve](margin_curve.png)", "",
               "Margin curve (snap_sym, overlap): share of benchmarks / detectors whose 90% cluster-bootstrap CI lies inside (−m, m).", ""]
    except ImportError:
        md += ["(matplotlib not available: margin curve not drawn)", ""]

    # second-pass framing
    s = e6["snap_sym|bench|POOLED"]
    md += ["## Second-pass detector framing", "",
           f"On the spans the transform replaced, a detector run after substitution misses {100-s['rec_t']:.1f}% of them, "
           f"against {100-s['rec_o']:.1f}% of the original values ({s['diff']:+.2f} points, 90% CI [{s['ci90'][0]:+.2f}, {s['ci90'][1]:+.2f}]; "
           f"relative increase in misses {100*s['diff']/(100-s['rec_o']):.1f}%). This concerns masked-span detectability only; it is not overall PHI "
           "exposure, which also depends on the transform's coverage (spans it did not replace stay verbatim).", ""]

    json.dump(res, open(os.path.join(OUT, "robustness.json"), "w"), indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
