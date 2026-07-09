"""Span-level P/R/F1 with strict and relaxed matching, plus leakage rate.

Matching modes
--------------
strict  : (start, end) must match exactly AND label must match.
type    : (start, end) match exactly, label ignored.
relaxed : any character overlap counts as a match (label ignored).

Leakage rate
------------
Fraction of *gold* spans that the system failed to mask. Equivalent to
(1 - recall_type) at the span level. We also report a token-weighted version
that captures how many sensitive characters slipped through.

Over-redaction
--------------
Fraction of *predicted* spans that do not overlap any gold span. On
hard-negative queries (no PHI), this collapses to a false-positive rate.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Literal

from benchmarks.base import PHISpan
from systems.base import PredictedSpan


Mode = Literal["strict", "type", "relaxed"]


@dataclass
class SpanScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    gold_chars_leaked: int = 0
    gold_chars_total: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def leakage_rate(self) -> float:
        return 1.0 - self.recall

    @property
    def char_leakage_rate(self) -> float:
        return self.gold_chars_leaked / self.gold_chars_total if self.gold_chars_total else 0.0


def _match(gold: PHISpan, pred: PredictedSpan, mode: Mode) -> bool:
    if mode == "strict":
        return gold.start == pred.start and gold.end == pred.end and gold.label == pred.label
    if mode == "type":
        return gold.start == pred.start and gold.end == pred.end
    # relaxed: any overlap
    return not (pred.end <= gold.start or pred.start >= gold.end)


def score_document(
    gold: list[PHISpan],
    pred: list[PredictedSpan],
    mode: Mode = "type",
) -> SpanScore:
    score = SpanScore()
    matched_pred: set[int] = set()
    matched_gold: set[int] = set()

    for gi, g in enumerate(gold):
        score.gold_chars_total += g.end - g.start
        for pi, p in enumerate(pred):
            if pi in matched_pred:
                continue
            if _match(g, p, mode):
                matched_pred.add(pi)
                matched_gold.add(gi)
                break
        else:
            score.gold_chars_leaked += g.end - g.start

    score.tp = len(matched_gold)
    score.fn = len(gold) - score.tp
    score.fp = len(pred) - len(matched_pred)
    return score


@dataclass
class Aggregate:
    system: str
    benchmark: str
    mode: Mode
    n_docs: int
    precision: float
    recall: float
    f1: float
    leakage_rate: float
    char_leakage_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate(
    system: str,
    benchmark: str,
    per_doc: Iterable[SpanScore],
    mode: Mode,
) -> Aggregate:
    total = SpanScore()
    n = 0
    for s in per_doc:
        total.tp += s.tp
        total.fp += s.fp
        total.fn += s.fn
        total.gold_chars_leaked += s.gold_chars_leaked
        total.gold_chars_total += s.gold_chars_total
        n += 1
    return Aggregate(
        system=system,
        benchmark=benchmark,
        mode=mode,
        n_docs=n,
        precision=total.precision,
        recall=total.recall,
        f1=total.f1,
        leakage_rate=total.leakage_rate,
        char_leakage_rate=total.char_leakage_rate,
    )
