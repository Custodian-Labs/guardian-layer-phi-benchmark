"""Convert a manually-downloaded ASQ-PHI .txt into the JSONL format the
loader expects.

Mendeley does not allow programmatic anonymous download, so do this first:

  1. Open https://data.mendeley.com/datasets/csz5dzp7nx/1 in a browser
  2. Click `synthetic_clinical_queries.txt` -> Download
  3. scp it to <repo>/data/asq_phi/raw/synthetic_clinical_queries.txt

Then:

  python scripts/import_asq_phi.py
  python scripts/run_benchmark.py --benchmark asq_phi --systems presidio obi --include-text
  python scripts/publish_results.py

The upstream format is one query per line with PHI tagged inline using a
`<PHI type="...">value</PHI>` markup. This script strips the tags, recovers
character offsets, and writes one JSONL row per query with fields
{id, text, phi_spans:[{start,end,label,text}]}.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG_RE = re.compile(r'<PHI\s+type="([^"]+)">(.*?)</PHI>', re.DOTALL)


def convert(src: Path, dst: Path) -> int:
    n = 0
    with src.open(encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            text, spans = _strip_tags(line)
            row = {
                "id": f"asq_{i:05d}",
                "text": text,
                "phi_spans": spans,
                "hard_negative": len(spans) == 0,
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _strip_tags(text: str) -> tuple[str, list[dict]]:
    """Return (plain_text, spans) where spans use offsets into plain_text."""
    out, spans, cursor = [], [], 0
    for m in TAG_RE.finditer(text):
        out.append(text[cursor:m.start()])
        plain_so_far = "".join(out)
        start = len(plain_so_far)
        value = m.group(2)
        out.append(value)
        end = start + len(value)
        spans.append({"start": start, "end": end, "label": m.group(1).upper(), "text": value})
        cursor = m.end()
    out.append(text[cursor:])
    return "".join(out), spans


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(ROOT / "data/asq_phi/raw/synthetic_clinical_queries.txt"))
    p.add_argument("--dst", default=str(ROOT / "data/asq_phi/raw/asq_phi.jsonl"))
    args = p.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"missing {src}. Download via browser first; see file docstring.")
        return 1

    n = convert(src, Path(args.dst))
    print(f"wrote {n} queries to {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
