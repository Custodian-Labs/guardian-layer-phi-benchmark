"""Philter wrapper (UCSF).

Philter is *not* pip-installable from PyPI as a clean library. You have two
options:

  (a) Clone https://github.com/BCHSI/philter-ucsf and run via subprocess.
  (b) Use the pip-published `philter-lite` fork (lighter feature set).

This wrapper assumes option (b) is available; swap `_run` for a subprocess
call if you opt for the full UCSF version.

Install:
    pip install philter-lite
"""
from __future__ import annotations

from systems.base import DeIDSystem, PredictedSpan, Prediction


class Philter(DeIDSystem):
    name = "philter"

    def __init__(self, config_path: str | None = None):
        try:
            import philter_lite  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "philter_lite not installed. `pip install philter-lite` or "
                "wire the UCSF philter repo via subprocess."
            ) from e
        self._philter = philter_lite

    def predict(self, text: str) -> Prediction:
        # API surface for philter-lite is small; main entry point typically
        # returns a list of (start, end, type) tuples and the filtered text.
        filtered_text, coords = self._philter.filter_text(text)
        spans = [
            PredictedSpan(start=int(s), end=int(e), label=str(t), text=text[s:e])
            for s, e, t in coords
        ]
        return Prediction(spans=spans, transformed_text=filtered_text, raw=coords)
