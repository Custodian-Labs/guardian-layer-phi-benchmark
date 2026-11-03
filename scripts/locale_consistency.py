#!/usr/bin/env python3
"""Does the surrogate stay in the document's language/locale?

For every masked NAME or LOCATION gold span in all 250 docs per benchmark, an
LLM judge answers a yes/no question about the value alone: is this a name / place
one would expect in a document written in <language>? It judges three values for
the same slot:
  original   the real value (control: how often the judge says "yes" for real PHI)
  custodian  the Custodian surrogate (snapped re-projection on the transformed text)
  faker      the locale-matched Faker surrogate (data/faker/, Faker locale = document language)
and we report the share judged locale-consistent per benchmark. Scoring is the
option log-likelihood backend of decision_eval.py (no generation, no parsing).

Usage: CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python scripts/locale_consistency.py --model qwen3_5_9b [--limit 250]
       PYTHONPATH=. python scripts/locale_consistency.py --analyze
"""
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from analyze_reprojection import snap  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
DL = os.path.join(ROOT, "web", "data", "downloads")
OUT = os.path.join(ROOT, "results", "locale")
BENCH = {  # corpus name -> (original file, transformed file, language)
    "asq_phi": ("asq_phi_250", "asq_phi_250_transformed", "English"),
    "meddocan": ("meddocan_250", "meddocan_250_transformed", "Spanish"),
    "pii_masking_300k": ("pii_masking_300k_250", "pii_masking_300k_250_transformed", "English"),
    "pii_dutch": ("pii_masking_300k_dutch_250", "pii_masking_300k_dutch_250_transformed", "Dutch"),
    "pii_french": ("pii_masking_300k_french_250", "pii_masking_300k_french_250_transformed", "French"),
    "pii_german": ("pii_masking_300k_german_250", "pii_masking_300k_german_250_transformed", "German"),
}
KIND = {"name": "person name", "loc": "place name (city, region, street or institution)"}
# explicit label lists (category() maps unknown labels such as PASS or TITLE to "name")
NAME_LABELS = {"NAME", "NOMBRE_SUJETO_ASISTENCIA", "NOMBRE_PERSONAL_SANITARIO",
               "GIVENNAME1", "GIVENNAME2", "LASTNAME1", "LASTNAME2", "LASTNAME3"}
LOC_LABELS = {"GEOGRAPHIC_LOCATION", "TERRITORIO", "CALLE", "PAIS", "HOSPITAL", "INSTITUCION", "CENTRO_SALUD",
              "STATE", "CITY", "STREET", "COUNTRY"}


def kind_of(label):
    return "name" if label in NAME_LABELS else "loc" if label in LOC_LABELS else None


def load(fn):
    return {r["doc_id"]: r for r in map(json.loads, filter(str.strip, open(os.path.join(DL, fn + ".jsonl"))))}


def items(bench, limit):
    """(doc_id, idx, kind, original, custodian, faker) for masked name/location spans."""
    of, tf, _ = BENCH[bench]
    O, T = load(of), load(tf)
    F = {r["doc_id"]: r for r in map(json.loads, filter(str.strip, open(os.path.join(ROOT, "data", "faker", bench + ".jsonl"))))}
    out = []
    for did in list(F)[:limit]:
        o, t, f = O[did], T.get(did), F[did]
        if not t or len(o["gold_spans"]) != len(t["gold_spans"]):
            continue
        order = sorted(range(len(o["gold_spans"])), key=lambda j: o["gold_spans"][j]["start"])  # faker is start-sorted
        if len(order) != len(f["gold_spans"]) or any(o["gold_spans"][order[k]]["label"] != g["label"]
                                                     for k, g in enumerate(f["gold_spans"])):
            continue  # ties in start order (a handful of PII-fr/de docs): alignment ambiguous
        for fi, fg in enumerate(f["gold_spans"]):
            i = order[fi]
            go, gt = o["gold_spans"][i], t["gold_spans"][i]
            kind = kind_of(go.get("label", ""))
            if kind not in KIND or not fg.get("masked"):
                continue
            ov = o["text"][go["start"]:go["end"]]
            s, e = snap(t["text"], gt["start"], gt["end"])
            cv = t["text"][s:e]
            if ov == cv:
                continue  # not replaced by the transform
            out.append((did, i, kind, ov, cv, f["text"][fg["start"]:fg["end"]]))
    return out


def run(args):
    from decision_eval import LocalBackend
    be = LocalBackend(args.model)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{args.model}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["bench"], r["doc_id"], r["idx"]) for r in map(json.loads, open(path))}
    with open(path, "a") as w:
        for bench, (_, _, lang) in BENCH.items():
            its = [x for x in items(bench, args.limit) if (bench, x[0], x[1]) not in done]
            print(f"[run] {bench}: {len(its)} spans", flush=True)
            for did, i, kind, ov, cv, fv in its:
                q = (f"This {KIND[kind]} appears in a document written in {lang}. Is it a {KIND[kind]} that would be "
                     f"typical in {lang}-speaking countries (for example a {lang}-sounding name, or a place located in "
                     f"a {lang}-speaking region)?")
                p = {arm: be.choose(f"Value: {v}", q, ["yes", "no"])[0] for arm, v in
                     (("original", ov), ("custodian", cv), ("faker", fv))}
                w.write(json.dumps({"bench": bench, "doc_id": did, "idx": i, "kind": kind,
                                    "values": {"original": ov, "custodian": cv, "faker": fv}, "p_yes": p}) + "\n")
                w.flush()
    print("[done]", flush=True)


def analyze():
    lines = ["# Locale consistency of surrogates (LLM judge, P(yes) > 0.5)", "",
             "| Model | Benchmark | Kind | N | Original (control) | Custodian | Faker |", "|---|---|---|--:|--:|--:|--:|"]
    summ = {}
    for fn in sorted(os.listdir(OUT)):
        if not fn.endswith(".jsonl"):
            continue
        rows = [json.loads(l) for l in open(os.path.join(OUT, fn))]
        groups = defaultdict(list)
        for r in rows:
            groups[(r["bench"], r["kind"])].append(r); groups[(r["bench"], "all")].append(r); groups[("all", "all")].append(r)
        for (b, k), rs in sorted(groups.items()):
            pct = {a: 100 * sum(r["p_yes"][a] > 0.5 for r in rs) / len(rs) for a in ("original", "custodian", "faker")}
            summ[f"{fn[:-6]}|{b}|{k}"] = {"n": len(rs), **pct}
            lines.append(f"| {fn[:-6]} | {b} | {k} | {len(rs)} | {pct['original']:.1f} | {pct['custodian']:.1f} | {pct['faker']:.1f} |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(lines) + "\n")
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3_5_9b")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else run(a)
