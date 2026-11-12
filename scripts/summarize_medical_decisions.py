#!/usr/bin/env python3
"""Compact summary of the clinical decision study (results/decisions/*__medical.jsonl).

Per model x arm, paired with the same model on the original text:
  ICD-10 (CodiEsp, 8 chapters, MEDDOCAN): macro accuracy and macro AUROC (P(yes)) original vs arm,
      with the change and a document-bootstrap 95% CI (resampling documents, all chapters together);
  MEDDOCAN sex and age-group accuracy change (gold from MEDDOCAN annotation);
  ASQ-PHI: mean argmax agreement with the original over its 4 questions (no gold labels).
Writes results/decisions/medical_summary.{md,json}.
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
import decision_eval as D  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT = os.path.join(ROOT, "results", "decisions")
CH = [f"icd_{c}" for c in D.ICD_CHAPTERS]


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def stats(pairs):
    """pairs: list of (orig_row, arm_row) for MEDDOCAN. Returns macro acc and AUROC for both."""
    acc_o, acc_a, au_o, au_a = [], [], [], []
    for c in CH:
        lab = [o["gold"][c] == "yes" for o, _ in pairs]
        po = [o["probs"][c][0] for o, _ in pairs]; pa = [a["probs"][c][0] for _, a in pairs]
        acc_o.append(sum((p > .5) == l for p, l in zip(po, lab)) / len(lab))
        acc_a.append(sum((p > .5) == l for p, l in zip(pa, lab)) / len(lab))
        au_o.append(auroc(po, lab)); au_a.append(auroc(pa, lab))
    m = lambda x: 100 * sum(x) / len(x)
    return m(acc_o), m(acc_a), m(au_o), m(au_a)


def main():
    md = ["# Clinical decision study: ASQ-PHI and MEDDOCAN (CodiEsp ICD-10)", "",
          "Paired with the same model on the original text. ICD: macro over 8 chapters (CodiEsp gold, threshold P(yes) > 0.5 "
          "for accuracy); 95% CI from 1,000 document bootstraps. Sex / age group: MEDDOCAN gold.", "",
          "| Model | Arm | ICD acc orig → arm (Δ, 95% CI) | ICD AUROC orig → arm (Δ, 95% CI) | Sex acc Δ | Age-group acc Δ | ASQ-PHI agreement |",
          "|---|---|--|--|--:|--:|--:|"]
    res = {}
    for fn in sorted(f for f in os.listdir(OUT) if f.endswith("__medical.jsonl")):
        model = fn.replace("local__", "").replace("__medical.jsonl", "")
        rows = [json.loads(l) for l in open(os.path.join(OUT, fn))]
        idx = {(r["bench"], r["arm"], r["doc_id"]): r for r in rows}
        for arm in ("custodian", "faker", "redacted"):
            pairs = [(idx[("meddocan", "original", d)], r) for (b, a, d), r in idx.items()
                     if b == "meddocan" and a == arm and ("meddocan", "original", d) in idx
                     and all(idx[("meddocan", "original", d)]["gold"].get(c) for c in CH)]
            ao, aa, uo, ua = stats(pairs)
            rng = random.Random(0); bs = []
            for _ in range(1000):
                s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
                x = stats(s); bs.append((x[1] - x[0], x[3] - x[2]))
            da = sorted(b[0] for b in bs); du = sorted(b[1] for b in bs)
            ci = lambda v: (v[25], v[974])
            def gold_delta(q):
                p = [(o, a) for o, a in pairs if o["gold"].get(q)]
                if not p:
                    return float("nan")
                opts = next(t for t in D.tasks("meddocan") if t["id"] == q)["options"]
                hit = lambda r: opts[r["probs"][q].index(max(r["probs"][q]))] == r["gold"][q]
                return 100 * (sum(hit(a) for _, a in p) - sum(hit(o) for o, _ in p)) / len(p)
            asq = [(idx[("asq_phi", "original", d)], r) for (b, a, d), r in idx.items()
                   if b == "asq_phi" and a == arm and ("asq_phi", "original", d) in idx]
            qs = [t["id"] for t in D.tasks("asq_phi")]
            agree = 100 * sum(o["probs"][q].index(max(o["probs"][q])) == r["probs"][q].index(max(r["probs"][q]))
                              for o, r in asq for q in qs) / (len(asq) * len(qs))
            cell = {"n_docs": len(pairs), "icd_acc": [ao, aa, ci(da)], "icd_auroc": [uo, ua, ci(du)],
                    "sex_delta": gold_delta("sex"), "age_delta": gold_delta("age_group"), "asq_agreement": agree}
            res[f"{model}|{arm}"] = cell
            md.append(f"| {model} | {arm} | {ao:.1f} → {aa:.1f} ({aa-ao:+.1f}, [{ci(da)[0]:+.1f}, {ci(da)[1]:+.1f}]) | "
                      f"{uo:.1f} → {ua:.1f} ({ua-uo:+.1f}, [{ci(du)[0]:+.1f}, {ci(du)[1]:+.1f}]) | "
                      f"{cell['sex_delta']:+.1f} | {cell['age_delta']:+.1f} | {agree:.1f}% |")
    json.dump(res, open(os.path.join(OUT, "medical_summary.json"), "w"), indent=1)
    open(os.path.join(OUT, "medical_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
