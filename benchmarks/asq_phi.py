"""ASQ-PHI loader.

Source: Weatherhead, Golovko & McCaffrey (UTMB, 2026).
DOI:    10.17632/csz5dzp7nx.1  (Mendeley Data, MIT license)
Paper:  https://www.sciencedirect.com/science/article/pii/S2352340926001393

Layout expected under `data/asq_phi/`:
    raw/asq_phi.jsonl   # one query per line, with fields {id, text, phi_spans}

Each `phi_spans` entry must have keys (start, end, label, text). The exact
field names in the upstream release may differ; adjust `_parse_row` once the
file is downloaded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from benchmarks.base import Benchmark, Document, PHISpan


class ASQPHI(Benchmark):
    name = "asq_phi"
    language = "en"

    DEFAULT_FILE = "raw/asq_phi.jsonl"

    def __init__(self, root: Path, file: str | None = None):
        super().__init__(root)
        self.file = self.root / (file or self.DEFAULT_FILE)

    def __iter__(self) -> Iterator[Document]:
        if not self.file.exists():
            raise FileNotFoundError(
                f"ASQ-PHI file not found at {self.file}. "
                "Download from https://doi.org/10.17632/csz5dzp7nx.1 "
                "and place jsonl under data/asq_phi/raw/."
            )
        with self.file.open() as f:
            for line in f:
                row = json.loads(line)
                yield self._parse_row(row)

    def __len__(self) -> int:
        with self.file.open() as f:
            return sum(1 for _ in f)

    @staticmethod
    def _parse_row(row: dict) -> Document:
        spans = [
            PHISpan(
                start=int(s["start"]),
                end=int(s["end"]),
                label=str(s.get("label", "PHI")),
                text=s.get("text", ""),
            )
            for s in row.get("phi_spans", [])
        ]
        return Document(
            doc_id=str(row.get("id") or row.get("query_id")),
            text=row["text"],
            gold_spans=spans,
            metadata={"hard_negative": bool(row.get("hard_negative", False))},
        )
