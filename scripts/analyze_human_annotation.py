#!/usr/bin/env python3
"""E1 analysis of the two-rater surrogate annotation (v2). Runs only on real rater files.

  python scripts/analyze_human_annotation.py raterA.csv raterB.csv [adjudicated.csv]

1. Agreement: raw agreement and Cohen's kappa on `category` and `boundary_problem`;
   confusion matrix; each rater vs the AI labels on the audit subset.
2. Consensus label per row: the agreed category, the adjudicated one if a third
   file is given, otherwise `disagreement`.
3. Outcome linkage: join rows to every detector's original/transformed outcome
   (paired_core, snap_sym, overlap) and report, per consensus category, the loss
   probability P(missed after | found on original) with a document-cluster
   bootstrap 95% CI, and the share of all lost outcomes.
4. Attribution buckets: generator defect (malformed, x-masked, type, locale,
   context), detector sensitivity (well-formed or salience change, no boundary
   problem), span-mapping artefact (boundary_problem = Y on both raters),
   uncertain (other_uncertain or disagreement).
Note: the loss-enriched half of the sample over-represents lost spans by design;
rates are reported separately for the audit (representative) half.
Writes results/annotation/human_v2.{json,md}.
"""
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
HD = os.path.join(ROOT, "data", "annotation", "human")
CATS = ["well_formed", "malformed_truncated", "x_masked", "type_inconsistent", "locale_inconsistent",
        "context_inconsistent", "salience_change", "other_uncertain"]
AI_MAP = {"none": "well_formed", "truncated_garbled": "malformed_truncated", "x_masked": "x_masked",
          "salience_loss": "salience_change", "other": "other_uncertain"}
GENERATOR = {"malformed_truncated", "x_masked", "type_inconsistent", "locale_inconsistent", "context_inconsistent"}


def kappa(a, b):
    """Cohen's kappa for two aligned label lists (labels compared after strip/lower)."""
    a = [x.strip().lower() for x in a]; b = [x.strip().lower() for x in b]
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def load_rater(path):
    rows = {r["id"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}
    bad = [i for i, r in rows.items() if r["category"].strip() not in CATS]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} rows without a valid category (first ids: {bad[:10]})")
    return rows


def consensus(a, b, adj=None):
    out = {}
    for i in a:
        ca, cb = a[i]["category"].strip(), b[i]["category"].strip()
        if adj and i in adj and adj[i].get("category", "").strip() in CATS:
            out[i] = adj[i]["category"].strip()
        else:
            out[i] = ca if ca == cb else "disagreement"
    return out


def bucket(cat, bnd_both):
    if bnd_both:
        return "span-mapping artefact"
    if cat in GENERATOR:
        return "generator defect"
    if cat in ("well_formed", "salience_change"):
        return "detector sensitivity"
    return "uncertain"


def boot_rate(units, B=2000, seed=0):
    """units: list of (doc_key, lost, found_orig). Cluster bootstrap CI of sum(lost)/sum(found_orig)."""
    by = defaultdict(lambda: [0, 0])
    for d, l, f in units:
        by[d][0] += l; by[d][1] += f
    keys = list(by); rng = random.Random(seed); vals = []
    for _ in range(B):
        s = [by[keys[rng.randrange(len(keys))]] for _ in keys]
        den = sum(x[1] for x in s)
        if den:
            vals.append(100 * sum(x[0] for x in s) / den)
    vals.sort()
    return (vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1]) if vals else (float("nan"), float("nan"))


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    a, b = load_rater(sys.argv[1]), load_rater(sys.argv[2])
    adj = {r["id"]: r for r in csv.DictReader(open(sys.argv[3]))} if len(sys.argv) > 3 else None
    if set(a) != set(b):
        raise SystemExit("rater files have different row ids")
    ids = sorted(a, key=int)
    key = {r["id"]: r for r in csv.DictReader(open(os.path.join(HD, "surrogate_quality_v2_key.csv")))}
    md, res = ["# Human surrogate annotation v2 (E1)", ""], {}

    po, k = kappa([a[i]["category"] for i in ids], [b[i]["category"] for i in ids])
    pb, kb = kappa([a[i]["boundary_problem"] or "N" for i in ids], [b[i]["boundary_problem"] or "N" for i in ids])
    res["agreement"] = {"n": len(ids), "category": {"raw": po, "kappa": k}, "boundary": {"raw": pb, "kappa": kb}}
    md += [f"Rows: {len(ids)}. Category: raw agreement {100*po:.1f}%, Cohen's κ = {k:.3f}. "
           f"Boundary problem: raw {100*pb:.1f}%, κ = {kb:.3f}.", "", "## Confusion (rater A rows × rater B columns)", "",
           "| A \\ B | " + " | ".join(CATS) + " |", "|---|" + "--:|" * len(CATS)]
    conf = Counter((a[i]["category"].strip(), b[i]["category"].strip()) for i in ids)
    md += [f"| {x} | " + " | ".join(str(conf[(x, y)]) for y in CATS) + " |" for x in CATS]

    ai = {r["id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data", "annotation", "surrogate_quality_audit.csv")))}
    sub = [i for i in ids if key[i]["origin"] == "audit200" and key[i]["audit_id"] in ai]
    for name, R in (("A", a), ("B", b)):
        p, kk = kappa([R[i]["category"] for i in sub], [AI_MAP.get(ai[key[i]["audit_id"]]["failure"], "other_uncertain") for i in sub])
        res[f"vs_ai_{name}"] = {"n": len(sub), "raw": p, "kappa": kk}
        md += ["", f"Rater {name} vs AI labels (audit subset, n = {len(sub)}): raw {100*p:.1f}%, κ = {kk:.3f}."]

    cons = consensus(a, b, adj)
    md += ["", "## Consensus categories", "", "| Category | Audit subset | Loss-enriched |", "|---|--:|--:|"]
    for c in CATS + ["disagreement"]:
        md.append(f"| {c} | {sum(cons[i] == c for i in ids if key[i]['origin'] == 'audit200')} | "
                  f"{sum(cons[i] == c for i in ids if key[i]['origin'] == 'loss_enriched')} |")

    # outcome linkage
    import paired_core as P
    outcomes = defaultdict(list)
    for r in P.paired_rows("snap_sym"):
        outcomes[(r["bench"], r["doc"], r["idx"])].append(r)
    md += ["", "## Loss given found on the original, by consensus category", "",
           "| Sample | Category | Spans | Outcomes found on original | Lost | Loss rate | 95% CI (doc cluster) |",
           "|---|---|--:|--:|--:|--:|--|"]
    res["linkage"] = {}
    for origin in ("audit200", "loss_enriched"):
        for c in CATS + ["disagreement"]:
            units, nsp = [], 0
            for i in ids:
                if key[i]["origin"] != origin or cons[i] != c:
                    continue
                k3 = (key[i]["benchmark"], key[i]["doc_id"], int(key[i]["span_index"]))
                nsp += 1
                for r in outcomes.get(k3, []):
                    if r["o"]["overlap"]:
                        units.append(((k3[0], k3[1]), int(not r["t"]["overlap"]), 1))
            if not units:
                continue
            fo = len(units); lost = sum(u[1] for u in units); lo, hi = boot_rate(units)
            res["linkage"][f"{origin}|{c}"] = {"spans": nsp, "found_orig": fo, "lost": lost, "rate": 100 * lost / fo, "ci95": [lo, hi]}
            md.append(f"| {origin} | {c} | {nsp} | {fo} | {lost} | {100*lost/fo:.1f}% | [{lo:.1f}, {hi:.1f}] |")

    # attribution of lost outcomes
    att = Counter()
    for i in ids:
        bnd = a[i]["boundary_problem"].strip().upper() == "Y" and b[i]["boundary_problem"].strip().upper() == "Y"
        k3 = (key[i]["benchmark"], key[i]["doc_id"], int(key[i]["span_index"]))
        att[bucket(cons[i], bnd)] += sum(r["o"]["overlap"] and not r["t"]["overlap"] for r in outcomes.get(k3, []))
    tot = sum(att.values())
    res["attribution_lost_outcomes"] = dict(att)
    md += ["", "## Attribution of lost outcomes (both samples; loss-enriched rows dominate)", "",
           "| Bucket | Lost outcomes | Share |", "|---|--:|--:|"]
    md += [f"| {k} | {v} | {100*v/max(1,tot):.1f}% |" for k, v in att.most_common()]
    os.makedirs(os.path.join(ROOT, "results", "annotation"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "annotation", "human_v2.json"), "w"), indent=1)
    open(os.path.join(ROOT, "results", "annotation", "human_v2.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
