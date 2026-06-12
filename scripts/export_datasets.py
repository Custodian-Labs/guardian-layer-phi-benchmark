"""Export benchmark subsets as clean downloadable JSONL files.

Writes web/data/downloads/:
  <bench>_250.jsonl              original 250-doc subset (doc_id, text, gold_spans)
  <bench>_250_transformed.jsonl  Custodian-transformed twin (when cached)
  downloads.json                 manifest the dashboard renders

Only SAMPLE_SAFE benchmarks are exported (synthetic / open-license corpora,
the same set whose raw text the dashboard already displays in samples).
Source for originals: the merged results files, which hold the exact graded
subsets; predictions are stripped.

Usage: python scripts/export_datasets.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "web" / "data" / "downloads"

SOURCES = {  # benchmark -> (merged results file, language, license note)
    "asq_phi": ("results/asq_phi_250.jsonl", "en",
                "MIT (Mendeley DOI 10.17632/csz5dzp7nx.1)"),
    "meddocan": ("results/meddocan_250.jsonl", "es",
                 "CC-BY 4.0 (MEDDOCAN / Plan TL shared task)"),
    "pii_masking_300k": ("results/pii_masking_300k_250.jsonl", "en",
                         "ai4privacy pii-masking-300k (open, attribution)"),
    "pii_masking_300k_dutch": ("results/pii_dutch_250.jsonl", "nl",
                               "ai4privacy pii-masking-300k (open, attribution)"),
    "pii_masking_300k_french": ("results/pii_french_250.jsonl", "fr",
                                "ai4privacy pii-masking-300k (open, attribution)"),
    "pii_masking_300k_german": ("results/pii_german_250.jsonl", "de",
                                "ai4privacy pii-masking-300k (open, attribution)"),
    "multiconer_v2": ("results/multiconer_v2_250.jsonl", "en",
                      "CC-BY 4.0 (MultiCoNER v2 shared task)"),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
                "datasets": []}

    for bench, (src, lang, lic) in SOURCES.items():
        src_path = ROOT / src
        if not src_path.exists():
            print(f"[skip] {bench}: {src} missing")
            continue

        # Original subset (strip predictions).
        orig_name = f"{bench}_250.jsonl"
        n_docs = n_spans = 0
        with (OUT / orig_name).open("w") as out_f, src_path.open() as in_f:
            for line in in_f:
                r = json.loads(line)
                spans = r.get("gold_spans", [])
                n_docs += 1
                n_spans += len(spans)
                out_f.write(json.dumps({
                    "doc_id": r["doc_id"],
                    "language": lang,
                    "text": r.get("text", ""),
                    "gold_spans": spans,
                }, ensure_ascii=False) + "\n")

        entry = {
            "benchmark": bench,
            "language": lang,
            "license": lic,
            "original": {"file": orig_name, "n_docs": n_docs, "n_spans": n_spans,
                         "bytes": (OUT / orig_name).stat().st_size},
            "transformed": None,
        }

        # Transformed twin (may be partial or absent while stage 1 runs).
        t_path = ROOT / "data" / "transformed" / f"{bench}.jsonl"
        if t_path.exists() and t_path.stat().st_size > 0:
            t_name = f"{bench}_250_transformed.jsonl"
            tn = ts = 0
            with (OUT / t_name).open("w") as out_f, t_path.open() as in_f:
                for line in in_f:
                    r = json.loads(line)
                    tn += 1
                    ts += len(r.get("gold_spans", []))
                    out_f.write(json.dumps({
                        "doc_id": r["doc_id"],
                        "language": lang,
                        "text": r["text"],
                        "gold_spans": r["gold_spans"],
                        "transform_meta": r.get("meta", {}),
                    }, ensure_ascii=False) + "\n")
            entry["transformed"] = {"file": t_name, "n_docs": tn, "n_spans": ts,
                                    "bytes": (OUT / t_name).stat().st_size,
                                    "complete": tn >= n_docs}
        manifest["datasets"].append(entry)
        t_note = (f"transformed={entry['transformed']['n_docs']}"
                  if entry["transformed"] else "transformed=–")
        print(f"[ok] {bench}: original={n_docs} docs/{n_spans} spans, {t_note}")

    (OUT / "downloads.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nmanifest -> {OUT/'downloads.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
