#!/usr/bin/env python3
"""C1 -- redaction upper-bound (the detectability floor).

Redaction replaces each masked PHI value with a mask token (`*` x len),
carrying no surrogate signal. This script measures masked-span recall under
three conditions with one CPU detector (Presidio) on the same spans:

  original  : detector on the real value
  transform : detector on the surrogate  (from results/t_*.jsonl, precomputed)
  redact    : detector on `*****`        (computed here)

The transform>redact gap quantifies what structure preservation buys. We run
Presidio because it is CPU-cheap and already part of the panel; the argument
(redaction destroys the signal by construction) is detector-independent.

Usage: python scripts/run_redact_baseline.py [bench ...]   # default: a few
"""
from __future__ import annotations

import glob
import json
import os
import sys

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")

# benchmark -> (original result glob w/ text+gold, spaCy language)
BENCH = {
    "ASQ-PHI": ("asq_250_baselines.jsonl", "en"),
    "MEDDOCAN": ("meddocan_250_baselines.jsonl", "es"),
    "PII en": ("pii_250_baselines.jsonl", "en"),
    "PII nl": ("pii_dutch_250_baselines.jsonl", "nl"),
    "PII fr": ("pii_french_250_baselines.jsonl", "fr"),
    "PII de": ("pii_german_250_baselines.jsonl", "de"),
}
# transformed counterpart (has surrogate text + remapped gold) for the masked set
TBASE = {
    "ASQ-PHI": "t_asq_phi_250_baselines.jsonl",
    "MEDDOCAN": "t_meddocan_250_baselines.jsonl",
    "PII en": "t_pii_masking_300k_250_baselines.jsonl",
    "PII nl": "t_pii_dutch_250_baselines.jsonl",
    "PII fr": "t_pii_french_250_baselines.jsonl",
    "PII de": "t_pii_german_250_baselines.jsonl",
}
PRESIDIO_KEY = "presidio"


def overlap(a_s, a_e, b_s, b_e):
    return not (b_e <= a_s or b_s >= a_e)


def load_docs(fn):
    docs = {}
    for line in open(os.path.join(RESULTS, fn)):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        docs[r["doc_id"]] = r
    return docs


def redact_text(text, spans):
    chars = list(text)
    for s in spans:
        for i in range(s["start"], min(s["end"], len(chars))):
            chars[i] = "*"
    return "".join(chars)


def recall_on(spans, pred_spans):
    if not spans:
        return None
    hit = sum(1 for g in spans if any(overlap(g["start"], g["end"], p.start, p.end) for p in pred_spans))
    return hit, len(spans)


def main():
    benches = [b for b in sys.argv[1:] if b in BENCH] or ["ASQ-PHI", "MEDDOCAN", "PII en"]
    print(f"{'benchmark':12s} {'masked':>7s} {'orig':>7s} {'transf':>7s} {'redact':>7s}   (Presidio recall on masked spans)")
    for name in benches:
        ofn, lang = BENCH[name]
        odocs = load_docs(ofn)
        tdocs = load_docs(TBASE[name])
        from systems.presidio import Presidio
        det = Presidio(language=lang)

        tot = ho = ht = hr = 0
        for did, od in odocs.items():
            td = tdocs.get(did)
            if not td:
                continue
            og, tg = od.get("gold_spans", []), td.get("gold_spans", [])
            if len(og) != len(tg):
                continue
            # masked spans = text changed between orig and transf
            masked_o = [g for g, t in zip(og, tg) if g.get("text") != t.get("text")]
            masked_t = [t for g, t in zip(og, tg) if g.get("text") != t.get("text")]
            if not masked_o:
                continue
            # original: reuse precomputed presidio preds if present, else run
            op = od.get("predictions", {}).get(PRESIDIO_KEY)
            o_preds = det.predict(od["text"]).spans if op is None else \
                [type("P", (), {"start": s["start"], "end": s["end"]})() for s in op["spans"]]
            tp = td.get("predictions", {}).get(PRESIDIO_KEY)
            t_preds = det.predict(td["text"]).spans if tp is None else \
                [type("P", (), {"start": s["start"], "end": s["end"]})() for s in tp["spans"]]
            # redact: replace masked spans in the ORIGINAL text, then detect
            r_text = redact_text(od["text"], masked_o)
            r_preds = det.predict(r_text).spans

            for masked, preds, acc in ((masked_o, o_preds, "o"), (masked_t, t_preds, "t"), (masked_o, r_preds, "r")):
                res = recall_on(masked, preds)
                if res:
                    hit, n = res
                    if acc == "o":
                        ho += hit
                    elif acc == "t":
                        ht += hit
                    else:
                        hr += hit
            tot += len(masked_o)

        def pct(x):
            return f"{100*x/tot:6.1f}%" if tot else "   n/a"
        print(f"{name:12s} {tot:7d} {pct(ho)} {pct(ht)} {pct(hr)}")
        det.close()


if __name__ == "__main__":
    main()
