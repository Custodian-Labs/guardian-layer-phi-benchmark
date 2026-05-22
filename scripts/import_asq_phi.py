"""Convert the upstream ASQ-PHI flat-file release into one JSONL row per
query, with character-level PHI spans.

Upstream format (in `synthetic_clinical_queries.txt`):

    ===QUERY===
    <free-text query string>
    ===PHI_TAGS===
    {"identifier_type": "NAME", "value": "Anna S."}
    {"identifier_type": "DATE", "value": "April 12, 2023"}

    ===QUERY===
    <next query>
    ===PHI_TAGS===
    (empty for hard negatives)

PHI is annotated by *value*, not by offset. We recover (start, end) by
locating each value in the query text. Repeated values are assigned to the
first un-claimed occurrence so two "John" entities in one query each get
distinct spans.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_file(src: Path) -> list[dict]:
    text = src.read_text(encoding="utf-8")
    blocks = []
    cur_query: str | None = None
    cur_tags: list[dict] = []
    state = "outside"
    lines = text.splitlines()

    def flush():
        if cur_query is not None:
            blocks.append({"query": cur_query.strip("\n"), "tags": list(cur_tags)})

    buf: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if line.strip() == "===QUERY===":
            if state != "outside":
                if state == "in_query":
                    cur_query = "\n".join(buf).rstrip()
                flush()
            cur_query = None
            cur_tags = []
            buf = []
            state = "in_query"
            continue
        if line.strip() == "===PHI_TAGS===":
            cur_query = "\n".join(buf).rstrip()
            buf = []
            state = "in_tags"
            continue
        if state == "in_query":
            buf.append(line)
        elif state == "in_tags":
            stripped = line.strip()
            if not stripped:
                continue
            try:
                cur_tags.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    flush()
    return blocks


def to_spans(query: str, tags: list[dict]) -> list[dict]:
    """Locate each tag value in the query; allocate distinct occurrences."""
    used: list[tuple[int, int]] = []
    spans: list[dict] = []
    for tag in tags:
        value = (tag.get("value") or "").strip()
        label = (tag.get("identifier_type") or "OTHER").strip().upper()
        if not value:
            continue
        start = -1
        cursor = 0
        while True:
            idx = query.find(value, cursor)
            if idx == -1:
                break
            if not any(s <= idx < e for s, e in used):
                start = idx
                break
            cursor = idx + 1
        if start == -1:
            continue  # value not literally present (paraphrased); skip
        end = start + len(value)
        used.append((start, end))
        spans.append({"start": start, "end": end, "label": label, "text": value})
    spans.sort(key=lambda s: s["start"])
    return spans


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(ROOT / "data/asq_phi/raw/synthetic_clinical_queries.txt"))
    p.add_argument("--dst", default=str(ROOT / "data/asq_phi/raw/asq_phi.jsonl"))
    args = p.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"missing {src}")
        return 1

    blocks = parse_file(src)
    n_pos = sum(1 for b in blocks if b["tags"])
    n_neg = len(blocks) - n_pos

    with open(args.dst, "w", encoding="utf-8") as f:
        for i, b in enumerate(blocks):
            spans = to_spans(b["query"], b["tags"])
            row = {
                "id": f"asq_{i:05d}",
                "text": b["query"],
                "phi_spans": spans,
                "hard_negative": len(b["tags"]) == 0,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(blocks)} queries to {args.dst} ({n_pos} PHI+, {n_neg} hard negatives)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
