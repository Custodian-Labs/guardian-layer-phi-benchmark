"""Gemma 4 wrapper.

Two install paths:
  1. Hosted via Google AI Studio / Vertex (uses GOOGLE_API_KEY).
  2. Local via HuggingFace transformers (no API key, needs GPU).

This module implements path (1). For path (2), swap `_invoke` with a local
`transformers.pipeline("text-generation", model=...)` call.
"""
from __future__ import annotations

import json
import os
import re

from systems.base import DeIDSystem, PredictedSpan, Prediction
from systems.llm_openai_compatible import SYSTEM_PROMPT, _parse_spans, _mask


class GemmaAIStudio(DeIDSystem):
    name = "gemma_4"

    def __init__(self, model: str | None = None):
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as e:
            raise RuntimeError("Install google-generativeai for hosted Gemma access.") from e

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model or os.environ.get("GEMMA_MODEL") or "gemma-4-31b-it"

    def predict(self, text: str) -> Prediction:
        model = self._genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=SYSTEM_PROMPT,
        )
        resp = model.generate_content(
            text,
            generation_config={"temperature": 0, "response_mime_type": "application/json"},
        )
        content = resp.text or ""
        spans = _parse_spans(content, text)
        transformed = _mask(text, spans)
        return Prediction(spans=spans, transformed_text=transformed, raw=content)
