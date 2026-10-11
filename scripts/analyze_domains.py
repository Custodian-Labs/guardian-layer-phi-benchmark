#!/usr/bin/env python3
"""Per-domain analysis for the non-medical benchmarks (finance / legal / enterprise).

Reads per-doc predictions written by scripts/run_domain_eval.py
(results/domains/preds/<corpus>/<bench>__<system>.jsonl) and the corpora in
data/domains/, and reports:

  1. Original-document detection: span P / R / F1 (mode `type` = exact
     boundaries, label-agnostic -- the repo default), overlap ("relaxed") F1,
     leakage = 1 - recall(type), char-leakage.
  2. Masked-span recall (overlap match, as in scripts/score_corpus.py and
     scripts/analyze_equivalence.py) on the SAME masked spans in three
     conditions: original, C2 Faker surrogate, C1 redaction (and the Custodian
     transform if results exist).
  3. Paired equivalence tests, original vs surrogate and original vs redaction,
     using exactly the estimator of scripts/analyze_equivalence.py (paired
     difference of proportions, McNemar-style variance, 95% CI, TOST at +-2 pt,
     McNemar chi2). Pooled per domain over detectors, per detector, and per
     detector family. As a robustness check we add a document-cluster bootstrap
     95% CI of the difference (spans within a doc are not independent).

Usage:
  PYTHONPATH=. python scripts/analyze_domains.py            # writes results/domains/summary.{json,md}
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))

from analyze_equivalence import mcnemar, tost  # noqa: E402  (same method as the medical study)
from benchmarks.base import PHISpan  # noqa: E402
from evaluation.metrics import aggregate, score_document  # noqa: E402
from systems.base import PredictedSpan  # noqa: E402

BENCHES = [("finance_gretel", "Finance"), ("legal_tab", "Legal"), ("enterprise_gretel", "Enterprise")]
DETECTORS = [  # (run key, display, family)
    ("presidio", "Presidio", "rule"),
    ("obi", "OBI deid_roberta", "fine-tuned"),
    ("qwen3_5_9b", "Qwen3.5-9B", "LLM"),
    ("llama3_1_8b", "Llama-3.1-8B", "LLM"),
    ("gemma_e4b", "Gemma-4-E4B", "LLM"),
    ("gemma_4_31b", "Gemma-4-31B", "LLM"),
    ("llama3_3_70b", "Llama-3.3-70B", "LLM"),
    ("qwen3_5_4b", "Qwen3.5-4B", "LLM"),
    ("qwen3_5_35b", "Qwen3.5-35B-A3B", "LLM"),
    ("deepseek_v2_lite", "DeepSeek-V2-Lite", "LLM"),
    ("openai", "GPT-5", "LLM"),
]
ARMS = ["faker", "redacted", "custodian"]
PRED = os.path.join(ROOT, "results", "domains", "preds")
OUT = os.path.join(ROOT, "results", "domains")


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else None


def corpus(bench, arm):
    if arm == "custodian":
        return jl(os.path.join(ROOT, "data", "transformed", f"{bench}.jsonl"))
    return jl(os.path.join(ROOT, "data", "domains", arm, f"{bench}.jsonl"))


def preds(arm, bench, det):
    rows = jl(os.path.join(PRED, arm, f"{bench}__{det}.jsonl"))
    # failed docs (legacy error rows) are not predictions; first clean row per doc wins
    out = {}
    for r in rows or []:
        if not r.get("error"):
            out.setdefault(r["doc_id"], r)
    return out or None


def hit(g, spans):
    return any(not (p["end"] <= g["start"] or p["start"] >= g["end"]) for p in spans)


def stats(pairs):
    """pairs: list of (doc_key, found_orig, found_arm). Same estimator as analyze_equivalence."""
    n = len(pairs)
    if n == 0:
        return None
    fo = sum(a for _, a, _ in pairs); fa = sum(b for _, _, b in pairs)
    b = sum(1 for _, x, y in pairs if x and not y); c = sum(1 for _, x, y in pairs if y and not x)
    d = (fo - fa) / n
    var = ((b + c) - (b - c) ** 2 / n) / n ** 2
    se = math.sqrt(var) if var > 0 else 0.0
    p2, eq2 = tost(d, b, c, n, 0.02)
    # One-sided non-inferiority at the same margin: H0 arm <= orig - 2pt.
    # (The two TOST halves; this is the half that matters for "surrogates do not
    # make PHI harder to detect". d = orig - arm, so H0: d >= +0.02.)
    z_ni = (0.02 - d) / (se if se > 0 else 1e-12)
    p_ni = 0.5 * math.erfc(z_ni / math.sqrt(2))
    return {"n": n, "noninf2_p": p_ni, "noninf_2pt": bool(p_ni < 0.05), "rec_orig": 100 * fo / n, "rec_arm": 100 * fa / n, "diff_pt": -100 * d,  # arm - orig
            "ci95": [-100 * (d + 1.96 * se), -100 * (d - 1.96 * se)], "tost2_p": p2, "equiv_2pt": bool(eq2),
            "mcnemar_chi2": mcnemar(b, c), "b_lost": b, "c_gained": c}


def cluster_boot(pairs, B=2000, seed=20260926):
    """Document-cluster bootstrap 95% CI of (arm - orig) recall difference, in points."""
    by = {}
    for did, x, y in pairs:
        by.setdefault(did, []).append((x, y))
    docs = list(by)
    if not docs:
        return None
    rng = random.Random(seed); ds = []
    for _ in range(B):
        sx = sy = n = 0
        for d in (rng.choice(docs) for _ in docs):
            for x, y in by[d]:
                sx += x; sy += y; n += 1
        ds.append(100 * (sy - sx) / n if n else 0.0)
    ds.sort()
    return [ds[int(0.025 * B)], ds[int(0.975 * B) - 1]]


def main():
    summary = {"original": [], "masked": [], "pooled": []}
    md = []
    orig_docs = {b: jl(os.path.join(ROOT, "data", "domains", "original", f"{b}.jsonl")) for b, _ in BENCHES}

    # ---------------------------------------------------------- 1. original detection
    md.append("## 1. Detection on original documents (span-level, N=250 docs/domain)\n")
    md.append("| Domain | Detector | Family | P | R | F1 (exact) | F1 (overlap) | Leakage | Char-leak |")
    md.append("|---|---|---|--:|--:|--:|--:|--:|--:|")
    for bench, bname in BENCHES:
        for det, dname, fam in DETECTORS:
            P = preds("original", bench, det)
            if not P:
                continue
            docs = [d for d in orig_docs[bench] if d["doc_id"] in P]
            dname_ = dname if len(docs) == len(orig_docs[bench]) else f"{dname} (partial {len(docs)})"
            res = {}
            for mode in ("type", "relaxed"):
                sc = []
                for d in docs:
                    gold = [PHISpan(s["start"], s["end"], s["label"], s["text"]) for s in d["gold_spans"]]
                    pr = [PredictedSpan(s["start"], s["end"], s["label"], s["text"]) for s in P[d["doc_id"]]["spans"]]
                    sc.append(score_document(gold, pr, mode=mode))
                res[mode] = aggregate(det, bench, sc, mode)
            t, r = res["type"], res["relaxed"]
            summary["original"].append({"benchmark": bench, "detector": det, "family": fam, "n_docs": t.n_docs,
                                        "precision": t.precision, "recall": t.recall, "f1": t.f1,
                                        "f1_overlap": r.f1, "recall_overlap": r.recall,
                                        "leakage": t.leakage_rate, "char_leakage": t.char_leakage_rate})
            md.append(f"| {bname} | {dname_} | {fam} | {t.precision:.3f} | {t.recall:.3f} | {t.f1:.3f} | "
                      f"{r.f1:.3f} | {t.leakage_rate:.3f} | {t.char_leakage_rate:.3f} |")

    # ---------------------------------------------------------- 2/3. masked-span recall + equivalence
    pools = {}  # (scope, arm) -> list of pairs
    incomplete = set()  # (detector, bench) excluded from sections 2-4 for incomplete coverage
    per = []
    by_label = {}  # (bench, arm, family, label) -> pairs
    for bench, bname in BENCHES:
        O = {d["doc_id"]: d for d in orig_docs[bench]}
        for arm in ARMS:
            C = corpus(bench, arm)
            if C is None:
                continue
            C = {d["doc_id"]: d for d in C}
            for det, dname, fam in DETECTORS:
                Po, Pa = preds("original", bench, det), preds(arm, bench, det)
                if not Po or not Pa:
                    continue
                if not (set(O) <= set(Po) and set(C) <= set(Pa)):
                    # still running / partially failed: leave it out rather than pool a subset
                    incomplete.add((det, bench))
                    continue
                pairs = []
                for did, cd in C.items():
                    od = O.get(did)
                    if od is None or did not in Po or did not in Pa:
                        continue
                    go, ga = od["gold_spans"], cd["gold_spans"]
                    if len(go) != len(ga):
                        continue
                    for g, a in zip(go, ga):
                        masked = a["masked"] if "masked" in a else (g["text"] != a["text"])
                        if not masked:
                            continue
                        pr = (f"{bench}/{did}", hit(g, Po[did]["spans"]), hit(a, Pa[did]["spans"]))
                        pairs.append(pr)
                        by_label.setdefault((bench, arm, fam, g["label"]), []).append(pr)
                s = stats(pairs)
                if not s:
                    continue
                s.update({"benchmark": bench, "arm": arm, "detector": det, "family": fam,
                          "n_docs": len({p[0] for p in pairs})})
                per.append(s)
                for scope in (("domain", bench), ("domain_family", bench, fam), ("all_family", fam), ("all",)):
                    pools.setdefault((scope, arm), []).extend(pairs)
    summary["masked"] = per
    summary["excluded_incomplete"] = sorted(map(list, incomplete))
    if incomplete:
        md.append("\n> Excluded from sections 2-4 (predictions do not yet cover every document): "
                  + ", ".join(f"{d} on {b}" for d, b in sorted(incomplete)) + "\n")

    def row(label, s, boot=None):
        ci = f"[{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]"
        bt = f"[{boot[0]:+.2f}, {boot[1]:+.2f}]" if boot else "-"
        eq = "yes" if s["equiv_2pt"] else "**no**"
        ni = "yes" if s["noninf_2pt"] else "**no**"
        return (f"| {label} | {s['n']} | {s['rec_orig']:.1f} | {s['rec_arm']:.1f} | {s['diff_pt']:+.2f} | {ci} | {bt} | "
                f"{s['tost2_p']:.1e} | {eq} | {s['noninf2_p']:.1e} | {ni} |")

    hdr = ("| Scope | N masked | Recall orig (%) | Recall arm (%) | Δ (pt) | 95% CI (McNemar) | 95% CI (doc bootstrap) | "
           "TOST ±2pt p | Equivalent? | Non-inf. −2pt p | Non-inferior? |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    arm_title = {"faker": "C2 open surrogate (Faker, format-preserving)", "redacted": "C1 redaction floor (`*****`)",
                 "custodian": "Custodian Guardian Layer transform"}
    md.append("\n## 2. Masked-span recall: original vs arm, per detector\n")
    md.append("Overlap match on the same masked gold spans. Δ = arm − original.\n")
    dn = {d: n for d, n, _ in DETECTORS}
    for arm in ARMS:
        rows = [s for s in per if s["arm"] == arm]
        if not rows:
            continue
        md.append(f"\n### {arm_title[arm]}\n")
        md.append(hdr)
        for bench, bname in BENCHES:
            for det, _, _ in DETECTORS:
                for s in rows:
                    if s["benchmark"] == bench and s["detector"] == det:
                        md.append(row(f"{bname} · {dn[det]}", s))

    md.append("\n## 3. Pooled equivalence (spans pooled over detectors, as in the medical study)\n")
    bn = dict(BENCHES)
    for arm in ARMS:
        if not any(k[1] == arm for k in pools):
            continue
        md.append(f"\n### {arm_title[arm]}\n")
        md.append(hdr)
        order = [("domain", b) for b, _ in BENCHES] + \
                [("domain_family", b, f) for b, _ in BENCHES for f in ("rule", "fine-tuned", "LLM")] + \
                [("all_family", f) for f in ("rule", "fine-tuned", "LLM")] + [("all",)]
        for scope in order:
            pairs = pools.get((scope, arm))
            if not pairs:
                continue
            s = stats(pairs); boot = cluster_boot(pairs)
            if scope[0] == "domain":
                lab = f"**{bn[scope[1]]}** (all detectors)"
            elif scope[0] == "domain_family":
                lab = f"{bn[scope[1]]} · {scope[2]}"
            elif scope[0] == "all_family":
                lab = f"All 3 domains · {scope[1]}"
            else:
                lab = "**All 3 domains · all detectors**"
            s.update({"scope": list(scope), "arm": arm, "boot_ci95": boot})
            summary["pooled"].append(s)
            md.append(row(lab, s, boot))

    # ---------------------------------------------------------- 4. per-label diagnostic (C2)
    md.append("\n## 4. Where do surrogate and original differ? Per-label masked-span recall, C2 Faker\n")
    md.append("Pooled over detectors of the given family; labels with N >= 30 (pooled over families) shown, "
              "sorted by |Δ| of the all-family pool.\n")
    summary["per_label"] = []
    for bench, bname in BENCHES:
        labs = {}
        for (b, arm, fam, lab), prs in by_label.items():
            if b == bench and arm == "faker":
                labs.setdefault(lab, {})[fam] = prs
        if not labs:
            continue
        md.append(f"\n### {bname}\n")
        md.append("| Label | N (all) | orig → faker, all (%) | Δ all | rule Δ | fine-tuned Δ | LLM Δ |")
        md.append("|---|--:|--:|--:|--:|--:|--:|")
        rows_ = []
        for lab, fams in labs.items():
            allp = [x for v in fams.values() for x in v]
            st = stats(allp)
            if st["n"] < 30:
                continue
            fd = {f: stats(fams[f]) for f in fams}
            rows_.append((abs(st["diff_pt"]), lab, st, fd))
            summary["per_label"].append({"benchmark": bench, "label": lab, "all": st,
                                         **{f: v for f, v in fd.items()}})
        for _, lab, st, fd in sorted(rows_, reverse=True):
            cell = lambda f: f"{fd[f]['diff_pt']:+.1f}" if f in fd else "-"
            md.append(f"| {lab} | {st['n']} | {st['rec_orig']:.1f} → {st['rec_arm']:.1f} | {st['diff_pt']:+.1f} | "
                      f"{cell('rule')} | {cell('fine-tuned')} | {cell('LLM')} |")

    os.makedirs(OUT, exist_ok=True)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {OUT}/summary.json and summary.md")


if __name__ == "__main__":
    main()
