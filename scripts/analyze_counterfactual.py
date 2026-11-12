#!/usr/bin/env python3
"""Same-slot counterfactual: is the residual loss caused by the
Custodian surrogates or by the detector?

For every masked span x detector outcome on the first 120 docs per benchmark we
have three paired results on the same gold span (same index):
  O  detector on the original value            (results/<bench>_250*.jsonl)
  C  detector on the Custodian surrogate       (results/t_<bench>_250*.jsonl, snapped re-projection)
  F  detector on a Faker same-type surrogate   (results/c1c2/preds/faker_<bench>_<det>.jsonl)

Among outcomes found on O and missed on C ("lost under Custodian"):
  found under Faker  -> the loss is specific to the Custodian surrogate
  missed under Faker -> the detector also misses a well-formed surrogate in that
                        slot, i.e. a detector / context limitation
and symmetrically for Faker losses. Faker is not a neutral control (its values
are canonical and often easier than real PHI), so the "specific to Custodian"
share is an upper bound on the surrogate-attributable part.

Usage: python scripts/analyze_counterfactual.py [--det presidio obi qwen3_5_9b]
"""
import argparse, json, os, random, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from analyze_equivalence import BENCH, load, found  # noqa: E402
from analyze_reprojection import snapped_load  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
PREDS = os.path.join(ROOT, "results", "c1c2", "preds")
B2C = {"ASQ-PHI": "asq_phi", "MEDDOCAN": "meddocan", "MultiCoNER v2": "multiconer_v2", "PII en": "pii_masking_300k",
       "PII nl": "pii_dutch", "PII fr": "pii_french", "PII de": "pii_german"}
DETKEY = {"presidio": "presidio", "obi": "deid_roberta_i2b2", "qwen3_5_9b": "qwen3.5_9b", "gemma_4_31b": "gemma_4_31b"}


FAKER = {c: {r["doc_id"]: r for r in map(json.loads, filter(str.strip, open(os.path.join(ROOT, "data", "faker", f"{c}.jsonl"))))}
         for c in B2C.values()}


def outcomes(dets):
    """rows: (bench, doc_id, det, O, C, F) for masked spans that have all three results."""
    rows, skipped = [], defaultdict(int)
    for bench, (og, tg) in BENCH.items():
        do, dt = load(og), snapped_load(tg)
        for det in dets:
            fn = os.path.join(PREDS, f"faker_{B2C[bench]}_{det}.jsonl")
            if not os.path.exists(fn):
                skipped[f"{det} (no faker preds)"] += 1
                continue
            key = DETKEY[det]
            for r in map(json.loads, open(fn)):
                o, t = do.get(r["doc_id"]), dt.get(r["doc_id"])
                if not o or not t or key not in o["preds"] or key not in t["preds"] or len(o["gold"]) != len(t["gold"]):
                    skipped[f"{bench}/{det} doc"] += 1
                    continue
                # the Faker corpus lists gold spans sorted by start; the original files may not
                # (MEDDOCAN), so map Faker index -> original index through the start order
                order = sorted(range(len(o["gold"])), key=lambda j: o["gold"][j]["start"])
                fg = FAKER[B2C[bench]][r["doc_id"]]["gold_spans"]
                if len(order) != len(fg) or any(o["gold"][order[k]]["label"] != g["label"] for k, g in enumerate(fg)):
                    skipped[f"{bench}/{det} doc misaligned"] += 1
                    continue
                for fi, f in zip(r["masked_idx"], r["found"]):
                    i = order[fi]
                    if o["gold"][i]["text"] == t["gold"][i]["text"]:
                        skipped[f"{bench}/{det} span not masked in t"] += 1
                        continue
                    rows.append((bench, r["doc_id"], det, found(o["gold"][i], o["preds"][key]),
                                 found(t["gold"][i], t["preds"][key]), bool(f)))
    return rows, skipped


def boot_share(rows, num, den, B=2000, seed=0):
    """Document-cluster bootstrap 95% CI for sum(num)/sum(den) over rows."""
    by = defaultdict(list)
    for r in rows:
        by[(r[0], r[1])].append(r)
    keys = list(by); rng = random.Random(seed); vals = []
    for _ in range(B):
        s = [r for k in (keys[rng.randrange(len(keys))] for _ in keys) for r in by[k]]
        d = sum(map(den, s))
        if d:
            vals.append(100 * sum(map(num, s)) / d)
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1]


def report(rows, label):
    n = len(rows)
    O = sum(r[3] for r in rows); C = sum(r[4] for r in rows); F = sum(r[5] for r in rows)
    lostC = [r for r in rows if r[3] and not r[4]]; lostF = [r for r in rows if r[3] and not r[5]]
    lc_ff = sum(r[5] for r in lostC); lf_cf = sum(r[4] for r in lostF)
    print(f"\n{label}: {n} masked span x detector outcomes in {len({(r[0], r[1]) for r in rows})} docs")
    print(f"  recall  original {100*O/n:.1f}  custodian {100*C/n:.1f}  faker {100*F/n:.1f}")
    if lostC:
        lo, hi = boot_share(rows, lambda r: r[3] and not r[4] and r[5], lambda r: r[3] and not r[4])
        print(f"  lost under Custodian (found on O): {len(lostC)}; of these found under Faker: {lc_ff} "
              f"({100*lc_ff/len(lostC):.0f}%, 95% CI [{lo:.0f}, {hi:.0f}]), missed under Faker too: {len(lostC)-lc_ff}")
    if lostF:
        print(f"  lost under Faker     (found on O): {len(lostF)}; of these found under Custodian: {lf_cf} "
              f"({100*lf_cf/len(lostF):.0f}%)")
    gainC = sum(1 for r in rows if not r[3] and r[4]); gainF = sum(1 for r in rows if not r[3] and r[5])
    print(f"  gained (missed on O, found after): Custodian {gainC}, Faker {gainF}")
    return {"n": n, "rec_o": 100 * O / n, "rec_c": 100 * C / n, "rec_f": 100 * F / n,
            "lost_c": len(lostC), "lost_c_found_f": lc_ff, "lost_f": len(lostF), "lost_f_found_c": lf_cf,
            "gain_c": gainC, "gain_f": gainF}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", nargs="+", default=["presidio", "obi", "qwen3_5_9b"])
    args = ap.parse_args()
    rows, skipped = outcomes(args.det)
    if skipped:
        print("skipped:", dict(skipped))
    out = {"all": report(rows, "ALL")}
    for det in args.det:
        sub = [r for r in rows if r[2] == det]
        if sub:
            out[det] = report(sub, f"detector {det}")
    for bench in BENCH:
        sub = [r for r in rows if r[0] == bench]
        if sub:
            out[bench] = report(sub, f"benchmark {bench}")
    json.dump(out, open(os.path.join(ROOT, "results", "c1c2", "counterfactual.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
