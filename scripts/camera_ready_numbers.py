#!/usr/bin/env python3
"""All numbers and LaTeX tables for the ML4H 2026 camera-ready, from stored results.

Primary analysis: corrected re-projection (snap_sym,
scripts/paired_core.py), label-agnostic overlap matching, stratified document-level
cluster bootstrap, Bayesian-bootstrap ROPE. Change metric used throughout the paper:
    Delta = recall(transformed) - recall(original), in points (negative = loss).

Reads: results/*.jsonl (predictions), results/robustness/robustness.json,
results/c1c2/*.json, results/c1c2/counterfactual.json, results/annotation/human_v2.json,
results/decisions/medical_summary.json, results/openai_audit/*.json,
results/audit/transform_properties.json, results/locale/summary.json.
Writes: paper/ml4h/camera_ready/numbers.json and paper/ml4h/camera_ready/tables/*.tex
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import paired_core as P  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
R = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "ml4h", "camera_ready")
TAB = os.path.join(OUT, "tables")
BENCHES = list(P.BENCH)
TNAME = {"NAME": "Name", "DATE": "Date", "LOC": "Location", "ID": "Identifier", "CONTACT": "Contact", "AGE": "Age", "SEX": "Sex", "OTHER": "Other"}
BSHORT = {"ASQ-PHI": "ASQ-PHI", "MEDDOCAN": "MEDDOCAN", "MultiCoNER v2": "MultiCoNER", "PII en": "PII-en",
          "PII nl": "PII-nl", "PII fr": "PII-fr", "PII de": "PII-de"}
DET = [("gemma_4_31b", "Gemma 4 31B"), ("llama3.3_70b", "Llama 3.3 70B"), ("qwen3.5_35b_a3b", "Qwen 3.5 35B-A3B"),
       ("openai_gpt", "GPT-4o-mini"), ("gemma_4_e4b", "Gemma 4 E4B"), ("qwen3.5_9b", "Qwen 3.5 9B"),
       ("qwen3.5_4b", "Qwen 3.5 4B"), ("presidio", "Presidio"), ("deepseek_v2_lite", "DeepSeek V2-Lite"),
       ("llama3.1_8b", "Llama 3.1 8B"), ("deid_roberta_i2b2", "OBI RoBERTa")]
B = 5000


def j(path):
    return json.load(open(os.path.join(R, path)))


def f1(x, d=1):
    return f"{x:.{d}f}"


def sg(x, d=2):
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.{d}f}"


def num(n):
    return f"{n:,}".replace(",", "{,}")


def pr(s, m):
    """posterior P(|Delta| < m); keys are ints in memory, strings after a JSON round trip."""
    v = s["p_rope"].get(m, s["p_rope"].get(str(m)))
    return "$>$0.99" if v >= 0.995 else f"{v:.2f}"


def flip(s):
    """paired_core reports diff = orig - transf; the paper uses transf - orig."""
    lo, hi = s["ci90"]
    lo95, hi95 = s["ci95"]
    return {**s, "delta": -s["diff"], "ci90_d": [-hi, -lo], "ci95_d": [-hi95, -lo95]}


def main():
    os.makedirs(TAB, exist_ok=True)
    N = {}
    cache = {}
    rows = list(P.paired_rows("snap_sym", cache=cache))
    texts = {b: (P._texts(P.BENCH[b][0]), P._texts(P.BENCH[b][1])) for b in BENCHES}

    # ---------------- primary: pooled + per benchmark (overlap)
    per = P.per_doc_arrays(rows, "overlap")
    pooled = flip(P.cluster_stats(per, B=B))
    N["pooled"] = pooled
    N["unique_spans"] = len({(r["bench"], r["doc"], r["idx"]) for r in rows})
    N["docs"] = len({(r["bench"], r["doc"]) for r in rows})
    N["bench"] = {b: flip(P.cluster_stats({b: per[b]}, B=B)) for b in BENCHES}
    N["det"] = {d: flip(P.cluster_stats(P.per_doc_arrays(rows, "overlap", keep=lambda r, d=d: r["det"] == d), B=B))
                for d, _ in DET}
    N["criteria"] = {c: flip(P.cluster_stats(P.per_doc_arrays(rows, c), B=B)) for c in P.CRITERIA}
    rows_d = list(P.paired_rows("difflib", cache=cache))
    N["paper_pipeline"] = flip(P.cluster_stats(P.per_doc_arrays(rows_d, "overlap"), B=B))
    N["paper_pipeline"]["unique_spans"] = len({(r["bench"], r["doc"], r["idx"]) for r in rows_d})

    # ---------------- coverage, whole-document F1 (exact) and leakage (overlap), span selection
    cov = {}
    wd = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0, 0]))  # det -> bench -> [tp_o, fp_o, fn_o, tp_t, fp_t, fn_t]
    leak = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))       # det -> bench -> [n_gold, missed_o, missed_t]
    sel = defaultdict(lambda: [0, 0, 0, 0])                          # bench -> masked n, found; unmasked n, found (original docs)
    seltype = defaultdict(lambda: [0, 0, 0, 0])
    for b in BENCHES:
        do, dt, _ = cache[b]
        ot, tt = texts[b]
        g_all = g_mask = 0
        for did, o in do.items():
            t = dt.get(did)
            if not t or len(o["gold"]) != len(t["gold"]) or did not in ot or did not in tt:
                continue
            gold_t = []
            masked = []
            for go, gt in zip(o["gold"], t["gold"]):
                s, e = P.snap_sym(ot[did], go["start"], go["end"], tt[did], gt["start"], gt["end"])
                gold_t.append((s, e, gt["label"]))
                ok = ot[did][go["start"]:go["end"]] == go["text"]
                masked.append(ok and go["text"] != tt[did][s:e])
                g_all += ok; g_mask += masked[-1]
            for det in set(o["preds"]) & set(t["preds"]):
                po, pt = o["preds"][det], t["preds"][det]
                for cond, gold, preds in (("o", [(g["start"], g["end"], g["label"]) for g in o["gold"]], po),
                                          ("t", gold_t, pt)):
                    used = set(); tp = 0
                    for (s, e, lab) in gold:
                        hit = next((k for k, p in enumerate(preds) if k not in used and p["start"] == s and p["end"] == e), None)
                        if hit is not None:
                            used.add(hit); tp += 1
                    off = 0 if cond == "o" else 3
                    wd[det][b][off] += tp; wd[det][b][off + 1] += len(preds) - len(used); wd[det][b][off + 2] += len(gold) - tp
                lk = leak[det][b]
                for k, ((s, e, lab), go) in enumerate(zip(gold_t, o["gold"])):
                    lk[0] += 1
                    lk[1] += not P.found(go["start"], go["end"], go["label"], po, "overlap")
                    lk[2] += not P.found(s, e, lab, pt, "overlap")
                    fo = P.found(go["start"], go["end"], go["label"], po, "overlap")
                    a = sel[b]; ty = seltype[P.gold_type(go["label"])]
                    if masked[k]:
                        a[0] += 1; a[1] += fo; ty[0] += 1; ty[1] += fo
                    else:
                        a[2] += 1; a[3] += fo; ty[2] += 1; ty[3] += fo
        cov[b] = 100 * g_mask / g_all
    N["coverage"] = cov

    def prf(tp, fp, fn):
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        return 2 * p * r / max(1e-9, p + r)
    N["wholedoc"] = {}
    for d, _ in DET:
        f1o = [prf(*wd[d][b][:3]) for b in BENCHES if b in wd[d]]
        f1t = [prf(*wd[d][b][3:]) for b in BENCHES if b in wd[d]]
        lo = [100 * leak[d][b][1] / leak[d][b][0] for b in BENCHES if leak[d][b][0]]
        lt = [100 * leak[d][b][2] / leak[d][b][0] for b in BENCHES if leak[d][b][0]]
        N["wholedoc"][d] = {"f1_o": float(np.mean(f1o)), "f1_t": float(np.mean(f1t)),
                            "leak_o": float(np.mean(lo)), "leak_t": float(np.mean(lt)), "n_bench": len(f1o)}
    sm = [sum(v[i] for v in sel.values()) for i in range(4)]
    N["span_selection"] = {"masked": 100 * sm[1] / sm[0], "unmasked": 100 * sm[3] / sm[2],
                           "by_type": {t: {"masked": 100 * v[1] / v[0] if v[0] else None,
                                           "unmasked": 100 * v[3] / v[2] if v[2] else None, "n_masked": v[0], "n_unmasked": v[2]}
                                       for t, v in seltype.items()}}

    # ---------------- other experiments (already computed)
    N["c1c2"] = {}
    for f in os.listdir(os.path.join(R, "c1c2")):
        if f.endswith(".json") and not f.startswith("counterfactual"):
            x = json.load(open(os.path.join(R, "c1c2", f)))
            N["c1c2"][f[:-5]] = x
    N["counterfactual"] = j("c1c2/counterfactual.json") if os.path.exists(os.path.join(R, "c1c2/counterfactual.json")) else None
    N["human"] = j("annotation/human_v2.json")
    N["downstream"] = j("decisions/medical_summary.json")
    N["gpt5"] = j("openai_audit/gpt5_comparison.json")
    N["openai_model_check"] = j("openai_audit/model_check.json")
    N["retest"] = j("openai_audit/retest_gpt-4o-mini_summary.json")
    N["transform"] = j("audit/transform_properties.json")
    N["locale"] = j("locale/summary.json")
    json.dump(N, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=float)
    write_tables(N)
    print(f"pooled: {pooled['rec_o']:.1f} -> {pooled['rec_t']:.1f}, Delta {pooled['delta']:+.2f}, "
          f"90% CI [{pooled['ci90_d'][0]:+.2f}, {pooled['ci90_d'][1]:+.2f}], n={pooled['n']}, spans={N['unique_spans']}, docs={N['docs']}")


def write_tables(N):
    # T1: per benchmark
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Benchmark & Cov. & Outc. & Recall O$\rightarrow$T & $\Delta$ [90\% CI] & $m^\ast$ & $P(|\Delta|{<}2)$\\", r"\midrule"]
    for b in BENCHES:
        s = N["bench"][b]
        L.append(f"{BSHORT[b]} & {f1(N['coverage'][b])} & {num(s['n'])} & {f1(s['rec_o'])}$\\rightarrow${f1(s['rec_t'])} & "
                 f"{sg(s['delta'])} [{sg(s['ci90_d'][0])}, {sg(s['ci90_d'][1])}] & {s['min_margin']:.2f} & {pr(s, 2)}\\\\")
    s = N["pooled"]
    L += [r"\midrule", f"\\textbf{{Pooled}} & & \\textbf{{{num(s['n'])}}} & \\textbf{{{f1(s['rec_o'])}$\\rightarrow${f1(s['rec_t'])}}} & "
                         f"\\textbf{{{sg(s['delta'])}}} [{sg(s['ci90_d'][0])}, {sg(s['ci90_d'][1])}] & \\textbf{{{s['min_margin']:.2f}}} & {pr(s, 2)}\\\\",
          r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "t_bench.tex"), "w").write("\n".join(L) + "\n")

    # T2: per detector
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{Whole document} & \multicolumn{4}{c}{Masked spans}\\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}",
         r"Detector & F1 O$\rightarrow$T & Leak O$\rightarrow$T & Recall O$\rightarrow$T & $\Delta$ & 90\% CI & $m^\ast$\\", r"\midrule"]
    for d, name in DET:
        w = N["wholedoc"][d]; s = N["det"][d]
        star = r"$^\dagger$" if w["n_bench"] < 7 or s["n"] < max(x["n"] for x in N["det"].values()) else ""
        L.append(f"{name}{star} & .{round(1000*w['f1_o']):03d}$\\rightarrow$.{round(1000*w['f1_t']):03d} & "
                 f"{f1(w['leak_o'])}$\\rightarrow${f1(w['leak_t'])} & {f1(s['rec_o'])}$\\rightarrow${f1(s['rec_t'])} & {sg(s['delta'])} & "
                 f"[{sg(s['ci90_d'][0])}, {sg(s['ci90_d'][1])}] & {s['min_margin']:.2f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "t_det.tex"), "w").write("\n".join(L) + "\n")

    # T3: matching criteria
    L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Criterion & O$\rightarrow$T & $\Delta$ [90\% CI] & $m^\ast$\\", r"\midrule"]
    names = {"overlap": "Overlap (primary)", "iou50": r"IoU $\geq$ 0.5", "exact": "Exact", "exact_type": "Exact + type"}
    for c in P.CRITERIA:
        s = N["criteria"][c]
        L.append(f"{names[c]} & {f1(s['rec_o'])}$\\rightarrow${f1(s['rec_t'])} & {sg(s['delta'])} [{sg(s['ci90_d'][0])}, {sg(s['ci90_d'][1])}] & {s['min_margin']:.2f}\\\\")
    s = N["paper_pipeline"]
    L += [r"\midrule", f"Overlap, uncorrected & {f1(s['rec_o'])}$\\rightarrow${f1(s['rec_t'])} & {sg(s['delta'])} [{sg(s['ci90_d'][0])}, {sg(s['ci90_d'][1])}] & {s['min_margin']:.2f}\\\\",
          r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "t_criteria.tex"), "w").write("\n".join(L) + "\n")

    # T4: C1 (redaction, incl. mask-aware prompt) and C2 (Faker), 120-doc subsample, read fresh
    C = {f[:-5]: json.load(open(os.path.join(R, "c1c2", f))) for f in os.listdir(os.path.join(R, "c1c2"))
         if f.endswith(".json") and not f.startswith("counterfactual")}
    key = {"ASQ-PHI": "asq_phi", "MEDDOCAN": "meddocan", "MultiCoNER v2": "multiconer_v2", "PII en": "pii_masking_300k",
           "PII nl": "pii_dutch", "PII fr": "pii_french", "PII de": "pii_german"}
    dets = [("presidio", "Presidio"), ("obi", "OBI"), ("qwen3_5_9b", "Qwen-9B"), ("gemma_4_31b", "Gemma-31B")]
    L = [r"\begin{tabular}{lr" + "r" * 5 + "r" * 4 + "}", r"\toprule",
         r" & & \multicolumn{5}{c}{C1: redaction (\tp{*****})} & \multicolumn{4}{c}{C2: Faker surrogates}\\",
         r"\cmidrule(lr){3-7}\cmidrule(lr){8-11}",
         r"Benchmark & $N$ & Presidio & OBI & Qwen-9B & Gemma-31B & Qwen-9B$^{\ddagger}$ & Presidio & OBI & Qwen-9B & Gemma-31B\\", r"\midrule"]
    means = defaultdict(list)
    for b in BENCHES:
        k = key[b]
        n = C.get(f"redacted_{k}_obi", {}).get("masked", 0)
        cells = []
        for corpus, dl in (("redacted", dets), ("redacted_ma", [("qwen3_5_9b", "")]), ("faker", dets)):
            for d, _ in dl:
                name = f"redacted_{k}_{d}_mask_aware" if corpus == "redacted_ma" else f"{corpus}_{k}_{d}"
                v = C.get(name, {}).get("recall_pct")
                cells.append("--" if v is None else f1(v))
                if v is not None and b != "MultiCoNER v2":
                    means[(corpus, d)].append(v)
        L.append(f"{BSHORT[b]} & {num(n)} & " + " & ".join(cells) + "\\\\")
    mrow = [f1(np.mean(means[(c, d)])) for c, dl in (("redacted", dets), ("redacted_ma", [("qwen3_5_9b", "")]), ("faker", dets)) for d, _ in dl]
    L += [r"\midrule", "Mean (excl.\\ MultiCoNER) & & " + " & ".join(mrow) + "\\\\", r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "t_c1c2.tex"), "w").write("\n".join(L) + "\n")

    # T5: human labels linked to outcomes + counterfactual
    H = N["human"]; lk = H["linkage"]
    cats = [("well_formed", "Well-formed"), ("x_masked", "x-masked"), ("malformed_truncated", "Malformed / truncated"),
            ("context_inconsistent", "Context-inconsistent"), ("type_inconsistent", "Type-inconsistent"),
            ("locale_inconsistent", "Locale-inconsistent"), ("disagreement", "Rater disagreement")]
    L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Consensus label & Spans & Found on O & Lost after T\\", r"\midrule"]
    for c, name in cats:
        v = lk.get(f"audit200|{c}")
        if v:
            L.append(f"{name} & {v['spans']} & {v['found_orig']} & {v['lost']} ({f1(v['rate'])}\\%)\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "t_human.tex"), "w").write("\n".join(L) + "\n")
    cf = N["counterfactual"]
    L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Detector & Lost (T) & Found (Faker) & Kept by T\\", r"\midrule"]
    for k, name in (("presidio", "Presidio"), ("obi", "OBI RoBERTa"), ("qwen3_5_9b", "Qwen 3.5 9B"), ("all", "\\textbf{All}")):
        v = cf[k]
        L.append(f"{name} & {v['lost_c']} & {v['lost_c_found_f']} ({round(100*v['lost_c_found_f']/v['lost_c'])}\\%) & "
                 f"{v['lost_f_found_c']}/{v['lost_f']} ({round(100*v['lost_f_found_c']/v['lost_f'])}\\%)\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "t_counterfactual.tex"), "w").write("\n".join(L) + "\n")

    # T6: downstream clinical tasks
    D = N["downstream"]
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Model & Text & ICD-10 acc.\ $\Delta$ [95\% CI] & AUROC $\Delta$ & Sex $\Delta$ & Age group $\Delta$\\", r"\midrule"]
    mname = {"qwen3.5_4b": "Qwen 3.5 4B", "qwen3.5_9b": "Qwen 3.5 9B", "gemma_4_e4b": "Gemma 4 E4B"}
    aname = {"custodian": "Guardian Layer", "faker": "Faker", "redacted": "Redacted"}
    for m in ("qwen3.5_4b", "qwen3.5_9b", "gemma_4_e4b"):
        for i, a in enumerate(("custodian", "faker", "redacted")):
            v = D[f"{m}|{a}"]; ao, aa, ci = v["icd_acc"]; uo, ua, _ = v["icd_auroc"]
            L.append(f"{mname[m] if i == 0 else ''} & {aname[a]} & {sg(aa-ao,1)} [{sg(ci[0],1)}, {sg(ci[1],1)}] & {sg(ua-uo,1)} & "
                     f"{sg(v['sex_delta'],1)} & {sg(v['age_delta'],1)}\\\\")
        L.append(r"\addlinespace" if m != "gemma_4_e4b" else "")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "t_downstream.tex"), "w").write("\n".join(x for x in L if x) + "\n")

    # T7: GPT-5 subset
    G = N["gpt5"]
    L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Detector & F1 (exact) & F1 (overlap) & Masked $\Delta$ [90\% CI]\\", r"\midrule"]
    gname = {"gpt-5": "GPT-5", "gemma_4_31b": "Gemma 4 31B", "llama3.3_70b": "Llama 3.3 70B$^\\dagger$",
             "openai_gpt": "GPT-4o-mini", "qwen3.5_35b_a3b": "Qwen 3.5 35B-A3B"}
    for d in ("gpt-5", "llama3.3_70b", "gemma_4_31b", "openai_gpt", "qwen3.5_35b_a3b"):
        w = G["wholedoc"][d]; ms = G["masked"].get(d)
        mtxt = "--" if not ms else f"{sg(-ms['diff'])} [{sg(-ms['ci90'][1])}, {sg(-ms['ci90'][0])}]"
        L.append(f"{gname[d]} & .{round(1000*w['f1']):03d} & .{round(1000*w['f1_overlap']):03d} & {mtxt}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "t_gpt5.tex"), "w").write("\n".join(L) + "\n")

    # T8: span selection by type
    S = N["span_selection"]
    L = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Gold type & Masked outc. & Recall & Unmasked outc. & Recall\\", r"\midrule"]
    for t in ("NAME", "DATE", "LOC", "ID", "CONTACT", "AGE", "SEX", "OTHER"):
        v = S["by_type"].get(t)
        if v and v["n_masked"] and v["n_unmasked"]:
            L.append(f"{TNAME[t]} & {num(v['n_masked'])} & {f1(v['masked'])} & {num(v['n_unmasked'])} & {f1(v['unmasked'])}\\\\")
    L += [r"\midrule", f"All & & {f1(S['masked'])} & & {f1(S['unmasked'])}\\\\", r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "t_selection.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    if "--tables-only" in sys.argv:
        write_tables(json.load(open(os.path.join(OUT, "numbers.json"))))
    else:
        main()
