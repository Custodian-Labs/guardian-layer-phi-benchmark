"""Abstract de-id system interface.

Every system — Custodian Layer, Presidio, an LLM, etc. — must implement
`predict(text)` and return a `Prediction` (detected spans + transformed text).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PredictedSpan:
    start: int
    end: int
    label: str
    text: str
    score: float = 1.0


@dataclass
class Prediction:
    spans: list[PredictedSpan] = field(default_factory=list)
    transformed_text: str | None = None  # text with PHI replaced/masked
    raw: object = None                    # provider-specific raw response, for debugging


class DeIDSystem:
    name: str = "abstract"

    def predict(self, text: str) -> Prediction:
        raise NotImplementedError

    def close(self) -> None:
        """Optional hook for releasing API clients / GPU memory."""
        return None
