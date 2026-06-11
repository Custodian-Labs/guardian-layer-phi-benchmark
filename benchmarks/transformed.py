"""Benchmarks backed by Custodian-transformed documents.

Each class reads `data/transformed/<orig_benchmark>.jsonl` produced by
`scripts/transform_corpus.py` (same 250-doc subsets as the original runs,
with PHI substituted by Guardian Layer transform and gold spans remapped).

Running the same detector suite over these gives the "after transform"
column of the structure-preservation comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from benchmarks.base import Benchmark, Document, PHISpan

_DATA = Path(__file__).resolve().parents[1] / "data" / "transformed"


class _TransformedBase(Benchmark):
    orig_name: str = ""

    def __init__(self, root: Path | None = None):
        # `root` is ignored: the cache location is fixed by orig_name.
        self.path = _DATA / f"{self.orig_name}.jsonl"
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} missing - run scripts/transform_corpus.py first"
            )

    def __iter__(self) -> Iterator[Document]:
        with self.path.open() as f:
            for line in f:
                r = json.loads(line)
                yield Document(
                    doc_id=r["doc_id"],
                    text=r["text"],
                    gold_spans=[PHISpan(**s) for s in r["gold_spans"]],
                    metadata=r.get("meta", {}),
                )

    def __len__(self) -> int:
        with self.path.open() as f:
            return sum(1 for _ in f)


class ASQPHITransformed(_TransformedBase):
    name = "asq_phi_transformed"
    orig_name = "asq_phi"
    language = "en"


class MEDDOCANTransformed(_TransformedBase):
    name = "meddocan_transformed"
    orig_name = "meddocan"
    language = "es"


class PIIMasking300kTransformed(_TransformedBase):
    name = "pii_masking_300k_transformed"
    orig_name = "pii_masking_300k"
    language = "en"


class PIIMasking300kDutchTransformed(_TransformedBase):
    name = "pii_masking_300k_dutch_transformed"
    orig_name = "pii_masking_300k_dutch"
    language = "nl"


class PIIMasking300kFrenchTransformed(_TransformedBase):
    name = "pii_masking_300k_french_transformed"
    orig_name = "pii_masking_300k_french"
    language = "fr"


class PIIMasking300kGermanTransformed(_TransformedBase):
    name = "pii_masking_300k_german_transformed"
    orig_name = "pii_masking_300k_german"
    language = "de"


class MultiCoNERv2Transformed(_TransformedBase):
    name = "multiconer_v2_transformed"
    orig_name = "multiconer_v2"
    language = "en"
