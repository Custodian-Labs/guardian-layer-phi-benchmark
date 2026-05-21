"""MEDDOCAN loader (BRAT format).

Source:  https://zenodo.org/records/4279323  (IberLEF 2019, public)
Layout expected under `data/meddocan/`:
    raw/train/brat/*.txt   *.ann
    raw/dev/brat/*.txt     *.ann
    raw/test/brat/*.txt    *.ann

BRAT .ann lines look like:
    T1   PAIS 35 41    España
    T2   NOMBRE_SUJETO_ASISTENCIA 0 14   Carlos Martinez
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

from benchmarks.base import Benchmark, Document, PHISpan


Split = Literal["train", "dev", "test"]


class MEDDOCAN(Benchmark):
    name = "meddocan"
    language = "es"

    def __init__(self, root: Path, split: Split = "test"):
        super().__init__(root)
        self.split = split
        # Zenodo zip extracts as raw/meddocan/<split>/brat/. Tolerate both
        # that layout and the flatter raw/<split>/brat/ from older docs.
        candidates = [
            self.root / "raw" / "meddocan" / split / "brat",
            self.root / "raw" / split / "brat",
        ]
        self.brat_dir = next((p for p in candidates if p.exists()), candidates[0])

    def __iter__(self) -> Iterator[Document]:
        if not self.brat_dir.exists():
            raise FileNotFoundError(
                f"MEDDOCAN split '{self.split}' not found at {self.brat_dir}. "
                "Download from https://zenodo.org/records/4279323 and extract "
                "the brat folders under data/meddocan/raw/<split>/brat/."
            )
        for txt_path in sorted(self.brat_dir.glob("*.txt")):
            ann_path = txt_path.with_suffix(".ann")
            text = txt_path.read_text(encoding="utf-8")
            spans = self._parse_ann(ann_path, text) if ann_path.exists() else []
            yield Document(
                doc_id=txt_path.stem,
                text=text,
                gold_spans=spans,
                metadata={"split": self.split},
            )

    def __len__(self) -> int:
        return sum(1 for _ in self.brat_dir.glob("*.txt"))

    @staticmethod
    def _parse_ann(ann_path: Path, text: str) -> list[PHISpan]:
        spans: list[PHISpan] = []
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("T"):
                continue
            try:
                _, body, surface = line.split("\t", 2)
            except ValueError:
                continue
            parts = body.split()
            label = parts[0]
            # BRAT supports discontiguous spans (semicolon-separated). Keep first segment.
            start = int(parts[1])
            end = int(parts[-1])
            spans.append(PHISpan(start=start, end=end, label=label, text=surface))
        return spans
