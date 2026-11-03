#!/usr/bin/env python3
"""E1: blind two-rater annotation set, v2 (extended categories + loss-enriched sample).

Rows = the 200 spans of the AI-labelled audit (data/annotation/surrogate_quality_audit.csv)
plus ~200 new masked spans enriched for detection losses: spans that at least one
detector found on the original and missed on the transformed text, stratified by
benchmark. Surrogates and contexts use the snap_sym re-projection (paired_core).

Raters see: original value, surrogate, type, original and transformed context.
They do NOT see detector outcomes, the AI labels, or which sample a row came from
(that key is in a separate file). Row order is shuffled with a fixed seed.

Writes data/annotation/human/
  surrogate_quality_v2_blind.csv   give one copy to each rater
  surrogate_quality_v2_key.csv     row id -> sample origin, audit id (do not share with raters)
  GUIDELINES_v2.md                 categories and decision rules
"""
import csv
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import paired_core as P  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUTD = os.path.join(ROOT, "data", "annotation", "human")
SEED, N_NEW = 20261016, 200
CATEGORIES = ["well_formed", "malformed_truncated", "x_masked", "type_inconsistent", "locale_inconsistent",
              "context_inconsistent", "salience_change", "other_uncertain"]
AUDIT_BENCH = {"ASQ-PHI": "ASQ-PHI", "MEDDOCAN": "MEDDOCAN", "MultiCoNER": "MultiCoNER v2", "PII-en": "PII en",
               "PII-nl": "PII nl", "PII-fr": "PII fr", "PII-de": "PII de"}


def ctx(text, s, e, w=70):
    return (text[max(0, s - w):s] + "⟦" + text[s:e] + "⟧" + text[e:e + w]).replace("\n", " ")


def main():
    cache = {}
    spans, lost_any = {}, defaultdict(bool)
    for r in P.paired_rows("snap_sym", cache=cache):
        k = (r["bench"], r["doc"], r["idx"])
        spans.setdefault(k, r)
        lost_any[k] |= r["o"]["overlap"] and not r["t"]["overlap"]
    otexts = {b: P._texts(P.BENCH[b][0]) for b in P.BENCH}
    ttexts = {b: P._texts(P.BENCH[b][1]) for b in P.BENCH}

    def row(k):
        b, did, i = k
        go, gt = cache[b][0][did]["gold"][i], cache[b][1][did]["gold"][i]
        ts, te = P.snap_sym(otexts[b][did], go["start"], go["end"], ttexts[b][did], gt["start"], gt["end"])
        return {"benchmark": b, "doc_id": did, "span_index": i, "type": go["label"], "original": go["text"],
                "surrogate": ttexts[b][did][ts:te], "context_original": ctx(otexts[b][did], go["start"], go["end"]),
                "context_transformed": ctx(ttexts[b][did], ts, te)}

    # 1) the 200 audited spans, located by (benchmark, doc, original value)
    audit = list(csv.DictReader(open(os.path.join(ROOT, "data", "annotation", "surrogate_quality_audit.csv"))))
    picked, keyrows = [], []
    for a in audit:
        b = AUDIT_BENCH[a["benchmark"]]
        cands = [k for k in spans if k[0] == b and k[1] == a["doc_id"] and spans[k]["orig_text"] == a["original"]]
        if not cands:
            keyrows.append({"origin": "audit200_unmatched", "audit_id": a["id"]})
            continue
        picked.append(cands[0]); keyrows.append({"origin": "audit200", "audit_id": a["id"], "key": cands[0]})
    # 2) loss-enriched new spans, stratified by benchmark (proportional, at least 10 each where available)
    rng = random.Random(SEED)
    pool = defaultdict(list)
    for k, lost in lost_any.items():
        if lost and k not in picked:
            pool[k[0]].append(k)
    tot = sum(len(v) for v in pool.values())
    quota = {b: max(10, round(N_NEW * len(v) / tot)) for b, v in pool.items()}
    while sum(min(quota[b], len(pool[b])) for b in pool) > N_NEW:
        quota[max(quota, key=quota.get)] -= 1
    new = []
    for b, v in sorted(pool.items()):
        v = sorted(v); rng.shuffle(v); new += v[:quota[b]]
    rows = [(k, "audit200") for k in picked] + [(k, "loss_enriched") for k in new]
    rng.shuffle(rows)
    os.makedirs(OUTD, exist_ok=True)
    blind = os.path.join(OUTD, "surrogate_quality_v2_blind.csv")
    cols = ["id", "benchmark", "type", "original", "surrogate", "context_original", "context_transformed",
            "category", "secondary_category", "boundary_problem", "confidence", "notes"]
    akey = {kr["key"]: kr["audit_id"] for kr in keyrows if "key" in kr}
    with open(blind, "w", newline="") as f, open(os.path.join(OUTD, "surrogate_quality_v2_key.csv"), "w", newline="") as fk:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        wk = csv.DictWriter(fk, fieldnames=["id", "origin", "audit_id", "benchmark", "doc_id", "span_index"]); wk.writeheader()
        for n, (k, origin) in enumerate(rows, 1):
            r = row(k)
            w.writerow({"id": n, **{c: r[c] for c in ("benchmark", "type", "original", "surrogate", "context_original",
                                                     "context_transformed")},
                        "category": "", "secondary_category": "", "boundary_problem": "", "confidence": "", "notes": ""})
            wk.writerow({"id": n, "origin": origin, "audit_id": akey.get(k, ""), "benchmark": k[0], "doc_id": k[1], "span_index": k[2]})
    print(f"wrote {len(rows)} rows ({len(picked)} audit, {len(new)} loss-enriched) -> {blind}")
    print("unmatched audit rows:", sum(kr["origin"] == "audit200_unmatched" for kr in keyrows))
    print("loss-enriched by benchmark:", {b: min(quota[b], len(pool[b])) for b in pool})


if __name__ == "__main__":
    main()
