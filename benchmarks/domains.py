"""Non-medical domain benchmarks (finance, legal, multi-domain enterprise).

Each class reads a seeded 250-doc sample materialised by
`scripts/build_domain_corpora.py` at `data/domains/original/<name>.jsonl`
(the build script documents sampling, TAB chunking and offset recovery).

  finance_gretel     gretelai/synthetic_pii_finance_multilingual (English test; Apache-2.0)
  legal_tab          Text Anonymization Benchmark, ECHR test split (Pilan et al. 2022; MIT),
                     paragraph windows <= 1500 chars, gold = DIRECT+QUASI mentions
  enterprise_gretel  gretelai/gretel-pii-masking-en-v1 (test; Apache-2.0), non-medical domains

Same Document(text, gold_spans=[PHISpan]) shape as every other benchmark, so
scripts/run_benchmark.py / evaluation.metrics work unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from benchmarks.base import Benchmark, Document, PHISpan

_DATA = Path(__file__).resolve().parents[1] / "data" / "domains" / "original"


class _DomainBase(Benchmark):
    file_name: str = ""
    domain: str = ""

    def __init__(self, root: Path | None = None, limit: int | None = None):
        # `root` ignored: the sample location is fixed (data/domains/original/).
        self.path = _DATA / f"{self.file_name}.jsonl"
        self.limit = limit
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} missing - run scripts/build_domain_corpora.py first"
            )

    def __iter__(self) -> Iterator[Document]:
        with self.path.open() as f:
            for i, line in enumerate(f):
                if self.limit and i >= self.limit:
                    break
                r = json.loads(line)
                yield Document(
                    doc_id=r["doc_id"],
                    text=r["text"],
                    gold_spans=[PHISpan(start=s["start"], end=s["end"], label=s["label"], text=s["text"])
                                for s in r["gold_spans"]],
                    metadata=r.get("meta", {}),
                )

    def __len__(self) -> int:
        with self.path.open() as f:
            n = sum(1 for _ in f)
        return min(n, self.limit) if self.limit else n


class FinanceGretel(_DomainBase):
    name = "finance_gretel"
    file_name = "finance_gretel"
    domain = "finance"
    language = "en"


class LegalTAB(_DomainBase):
    name = "legal_tab"
    file_name = "legal_tab"
    domain = "legal"
    language = "en"


class EnterpriseGretel(_DomainBase):
    name = "enterprise_gretel"
    file_name = "enterprise_gretel"
    domain = "enterprise"
    language = "en"
