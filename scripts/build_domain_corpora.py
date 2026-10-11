#!/usr/bin/env python3
"""Build the NON-MEDICAL domain benchmarks (finance, legal, multi-domain enterprise).

Downloads each public dataset, draws a seeded 250-document sample, converts it
to the repo's Document/PHISpan JSONL shape and sanity-checks every gold span
(text[start:end] == span text). Output (one row per doc, same shape as
data/transformed/*.jsonl, so benchmarks/domains.py and scripts/transform_corpus.py
can consume it directly):

  data/domains/original/<bench>.jsonl   {doc_id, text, gold_spans:[{start,end,label,text}], meta}

Benchmarks
----------
finance_gretel     gretelai/synthetic_pii_finance_multilingual, English *test* split
                   (Apache-2.0). Docs with >=1 PII span and <= 4000 chars.
legal_tab          Text Anonymization Benchmark (TAB; ECHR court judgments, MIT),
                   *test* split, first annotator per case. Gold = mentions whose
                   identifier_type is DIRECT or QUASI (TAB's "must be masked" set;
                   NO_MASK mentions are not gold). Cases are long (~5-30k chars), so
                   they are CHUNKED at paragraph ("\\n") boundaries into windows of
                   <= 1500 chars; a gold span crossing a window boundary is dropped
                   (counted). 250 windows with >=1 gold span are sampled.
enterprise_gretel  gretelai/gretel-pii-masking-en-v1, *test* split (Apache-2.0),
                   restricted to NON-medical domains (drops healthcare,
                   healthcare-administration, pharmaceuticals-biotechnology).
                   The dataset gives entity strings, not offsets; offsets are
                   recovered with exact string search (every entity occurs exactly
                   once in the test split, verified at build time).

Overlapping / nested gold spans are resolved by keeping the longest span.

Usage:
  python scripts/build_domain_corpora.py            # all three
  python scripts/build_domain_corpora.py --only legal_tab
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import zipfile
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT = os.path.join(ROOT, "data", "domains", "original")
N = 250
SEED = 20260926
MEDICAL_DOMAINS = {"healthcare", "healthcare-administration", "pharmaceuticals-biotechnology"}


def _resolve_overlaps(spans):
    """Keep longest span among overlapping ones; return sorted, non-overlapping."""
    spans = sorted(spans, key=lambda s: (-(s["end"] - s["start"]), s["start"]))
    kept = []
    for s in spans:
        if all(s["end"] <= k["start"] or s["start"] >= k["end"] for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: s["start"])


def _check(rows, name):
    bad = 0
    for r in rows:
        for s in r["gold_spans"]:
            if r["text"][s["start"]:s["end"]] != s["text"] or s["end"] <= s["start"]:
                bad += 1
    n_sp = sum(len(r["gold_spans"]) for r in rows)
    labels = Counter(s["label"] for r in rows for s in r["gold_spans"])
    print(f"[{name}] docs={len(rows)} spans={n_sp} offset_mismatches={bad} "
          f"mean_chars={sum(len(r['text']) for r in rows)/max(1,len(rows)):.0f}")
    print(f"[{name}] labels: {dict(labels.most_common())}")
    assert bad == 0, f"{name}: {bad} span offsets do not match text"


def _write(rows, name):
    os.makedirs(OUT, exist_ok=True)
    _check(rows, name)
    with open(os.path.join(OUT, f"{name}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------- finance
def build_finance():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("gretelai/synthetic_pii_finance_multilingual",
                        "data/English_test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    cands = []
    for _, r in df.iterrows():
        text = r["generated_text"]
        if not text or len(text) > 4000:
            continue
        spans = []
        for e in json.loads(r["pii_spans"]):
            s, t = int(e["start"]), int(e["end"])
            if 0 <= s < t <= len(text):
                spans.append({"start": s, "end": t, "label": e["label"], "text": text[s:t]})
        spans = _resolve_overlaps(spans)
        if spans:
            cands.append({"doc_id": f"fin_{r['__index_level_0__']}", "text": text, "gold_spans": spans,
                          "meta": {"domain": "finance", "document_type": r["document_type"],
                                   "language": "en", "source": "gretelai/synthetic_pii_finance_multilingual:English_test"}})
    rows = random.Random(SEED).sample(cands, N)
    print(f"[finance_gretel] candidates={len(cands)} (of {len(df)} test docs)")
    _write(rows, "finance_gretel")


# ----------------------------------------------------------------------- legal
def build_tab(max_chars: int = 1500):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("ildpil/text-anonymization-benchmark", "echr_test.zip", repo_type="dataset")
    raw = os.path.join(ROOT, "data", "domains", "raw", "tab")
    os.makedirs(raw, exist_ok=True)
    zipfile.ZipFile(p).extractall(raw)
    data = json.load(open(os.path.join(raw, "echr_test.json")))
    seen, cases = set(), []
    for x in data:  # first annotator per case, in file order
        if x["doc_id"] in seen:
            continue
        seen.add(x["doc_id"]); cases.append(x)
    windows, dropped, kept_total = [], 0, 0
    for x in cases:
        text = x["text"]
        em = x["entity_mentions"]
        em = em if isinstance(em, list) else ast.literal_eval(em)
        gold = [{"start": e["start_offset"], "end": e["end_offset"], "label": e["entity_type"],
                 "text": text[e["start_offset"]:e["end_offset"]], "id_type": e["identifier_type"]}
                for e in em if e["identifier_type"] in ("DIRECT", "QUASI")]
        gold = _resolve_overlaps(gold)
        # paragraph boundaries
        paras, pos = [], 0
        for line in text.split("\n"):
            paras.append((pos, pos + len(line))); pos += len(line) + 1
        # greedy windows of consecutive non-empty paragraphs
        cur = None
        wins = []
        for s, e in paras:
            if e == s:
                continue
            if cur is None:
                cur = [s, e]
            elif e - cur[0] <= max_chars:
                cur[1] = e
            else:
                wins.append(tuple(cur)); cur = [s, e]
        if cur:
            wins.append(tuple(cur))
        for wi, (ws, we) in enumerate(wins):
            wtext = text[ws:we]
            spans = []
            for g in gold:
                if g["start"] >= ws and g["end"] <= we:
                    spans.append({"start": g["start"] - ws, "end": g["end"] - ws, "label": g["label"],
                                  "text": g["text"], "id_type": g["id_type"]})
                elif g["start"] < we and g["end"] > ws:
                    dropped += 1
            kept_total += len(spans)
            if spans:
                windows.append({"doc_id": f"tab_{x['doc_id']}_w{wi}", "text": wtext, "gold_spans": spans,
                                "meta": {"domain": "legal", "case": x["doc_id"], "annotator": x["annotator_id"],
                                         "window": [ws, we], "language": "en",
                                         "source": "TAB echr_test (ildpil/text-anonymization-benchmark)"}})
    print(f"[legal_tab] cases={len(cases)} windows_with_gold={len(windows)} "
          f"spans_in_windows={kept_total} spans_dropped_crossing_boundary={dropped}")
    rows = random.Random(SEED).sample(windows, N)
    _write(rows, "legal_tab")


# ------------------------------------------------------------------ enterprise
def build_enterprise():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("gretelai/gretel-pii-masking-en-v1", "data/test-00000-of-00001.parquet",
                        repo_type="dataset")
    df = pd.read_parquet(p)
    cands, multi = [], 0
    for _, r in df.iterrows():
        if r["domain"] in MEDICAL_DOMAINS:
            continue
        text = r["text"]
        spans = []
        for e in ast.literal_eval(r["entities"]):
            ent = e["entity"]
            if not ent:
                continue
            i = text.find(ent)
            if text.count(ent) > 1:
                multi += 1
            while i >= 0:
                spans.append({"start": i, "end": i + len(ent), "label": e["types"][0], "text": ent})
                i = text.find(ent, i + len(ent))
        spans = _resolve_overlaps(spans)
        if spans:
            cands.append({"doc_id": f"ent_{r['uid'][:12]}", "text": text, "gold_spans": spans,
                          "meta": {"domain": r["domain"], "document_type": r["document_type"],
                                   "language": "en", "source": "gretelai/gretel-pii-masking-en-v1:test"}})
    print(f"[enterprise_gretel] non-medical candidates={len(cands)} multi-occurrence entities={multi}")
    rows = random.Random(SEED).sample(cands, N)
    print(f"[enterprise_gretel] domains in sample={len(set(r['meta']['domain'] for r in rows))}")
    _write(rows, "enterprise_gretel")


BUILDERS = {"finance_gretel": build_finance, "legal_tab": build_tab, "enterprise_gretel": build_enterprise}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(BUILDERS), nargs="*")
    a = ap.parse_args()
    for k in (a.only or BUILDERS):
        BUILDERS[k]()
