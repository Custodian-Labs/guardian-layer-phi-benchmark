"""ai4privacy/pii-masking-300k loader.

Source: https://huggingface.co/datasets/ai4privacy/pii-masking-300k
Schema: source_text / privacy_mask=[{value,start,end,label}] / language

We sample English entries (the largest split) and yield them as the same
Document(text, gold_spans=[PHISpan]) shape every other benchmark uses, so the
runner / metrics / dashboard pipeline is unchanged.

This benchmark widens the comparison from MEDDOCAN's narrow Spanish-clinical
focus to a multilingual general-PII setting. It does not replace MEDDOCAN; it
sits alongside it on the dashboard.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from benchmarks.base import Benchmark, Document, PHISpan


class PIIMasking300k(Benchmark):
    name = "pii_masking_300k"
    language = "en"

    DEFAULT_LANGUAGE = "English"

    def __init__(
        self,
        root: Path,
        split: str = "validation",
        language: str | None = None,
        limit: int | None = None,
        seed: int = 0,
    ):
        super().__init__(root)
        self.split = split
        self.target_language = language or self.DEFAULT_LANGUAGE
        self.limit = limit
        self.seed = seed
        self._docs: list[Document] | None = None

    def _load(self) -> list[Document]:
        from datasets import load_dataset

        ds = load_dataset("ai4privacy/pii-masking-300k", split=self.split)
        if self.target_language:
            ds = ds.filter(lambda r: r.get("language") == self.target_language)
        if self.limit and self.limit < len(ds):
            ds = ds.shuffle(seed=self.seed).select(range(self.limit))

        out: list[Document] = []
        for i, row in enumerate(ds):
            text = row["source_text"] or ""
            spans = []
            for m in row.get("privacy_mask") or []:
                try:
                    spans.append(PHISpan(
                        start=int(m["start"]),
                        end=int(m["end"]),
                        label=str(m["label"]),
                        text=str(m.get("value", text[int(m["start"]):int(m["end"])])),
                    ))
                except (KeyError, ValueError, TypeError):
                    continue
            out.append(Document(
                doc_id=f"pii_{self.split}_{row.get('id', i)}",
                text=text,
                gold_spans=spans,
                metadata={"language": row.get("language", "?")},
            ))
        return out

    def __iter__(self) -> Iterator[Document]:
        if self._docs is None:
            self._docs = self._load()
        yield from self._docs

    def __len__(self) -> int:
        if self._docs is None:
            self._docs = self._load()
        return len(self._docs)


class PIIMasking300kDutch(PIIMasking300k):
    name = "pii_masking_300k_dutch"
    language = "nl"
    DEFAULT_LANGUAGE = "Dutch"
