"""Abstract interfaces shared by every benchmark loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class PHISpan:
    start: int
    end: int
    label: str
    text: str


@dataclass
class Document:
    doc_id: str
    text: str
    gold_spans: list[PHISpan] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class Benchmark:
    """Subclass per dataset. Yields Documents with gold PHI spans."""

    name: str = "abstract"
    language: str = "en"

    def __init__(self, root: Path):
        self.root = Path(root)

    def __iter__(self) -> Iterator[Document]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
