"""Generic OpenAI-compatible chat completion wrapper.

Reused for:
  * OpenAI (GPT-5, GPT-4.1, ...)
  * DeepSeek (V4-Pro / V4-Flash)
  * Moonshot Kimi (K2.6)
  * Alibaba Qwen DashScope (OpenAI-compatible mode)

LLMs are not PHI detectors out of the box — we prompt them to emit a JSON
array of spans. The prompt is intentionally minimal to keep the comparison
fair; tweak per provider only if you also re-run the others.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from systems.base import DeIDSystem, PredictedSpan, Prediction


SYSTEM_PROMPT = """You are a clinical de-identification engine.
Given a piece of text, identify every span that contains Protected Health
Information (PHI) per HIPAA Safe Harbor (names, dates, geographic subdivisions
smaller than a state, phone numbers, fax numbers, email, SSN, MRN, account
numbers, license numbers, vehicle IDs, device IDs, URLs, IP, biometric IDs,
photos, any other unique identifier).

Return ONLY a JSON object with key "spans" whose value is a list of objects
with fields:
  - start: int (character offset, inclusive)
  - end:   int (character offset, exclusive)
  - label: string (one of NAME, DATE, LOCATION, PHONE, EMAIL, ID, AGE, OTHER)
  - text:  the matched substring

Do not include any text outside the JSON object."""


@dataclass
class LLMConfig:
    api_key_env: str
    base_url_env: str | None
    model_env: str
    default_model: str
    name: str
    default_base_url: str | None = None


def make_llm(cfg: LLMConfig) -> "OpenAICompatibleLLM":
    return OpenAICompatibleLLM(cfg)


class OpenAICompatibleLLM(DeIDSystem):
    def __init__(self, cfg: LLMConfig):
        from openai import OpenAI
        import os

        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"{cfg.api_key_env} not set")

        base_url = None
        if cfg.base_url_env:
            base_url = os.environ.get(cfg.base_url_env) or cfg.default_base_url

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = os.environ.get(cfg.model_env) or cfg.default_model
        self.name = cfg.name

    def predict(self, text: str) -> Prediction:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        spans = _parse_spans(content, text)
        transformed = _mask(text, spans)
        return Prediction(spans=spans, transformed_text=transformed, raw=content)


def _parse_spans(content: str, original: str) -> list[PredictedSpan]:
    """Parse the LLM's JSON, then recover correct offsets by searching the
    predicted substring in the original text.

    LLMs identify the right PHI tokens but cannot count characters reliably,
    so their `start`/`end` are typically wrong by tens of characters. We use
    their offsets only as a hint to disambiguate when the predicted substring
    occurs multiple times in the document.
    """
    data = _robust_json_parse(content)
    spans_raw = data.get("spans", []) if isinstance(data, dict) else []
    out: list[PredictedSpan] = []
    used: set[tuple[int, int]] = set()
    for s in spans_raw:
        surface = (s.get("text") or "").strip()
        if not surface:
            continue
        try:
            hint = int(s.get("start", 0))
        except (TypeError, ValueError):
            hint = 0

        # Find every occurrence of the predicted substring; pick the one
        # closest to the LLM's hinted offset that we haven't used yet.
        occurrences = [
            i for i in range(len(original))
            if original.startswith(surface, i) and (i, i + len(surface)) not in used
        ]
        if not occurrences:
            continue
        start = min(occurrences, key=lambda i: abs(i - hint))
        end = start + len(surface)
        used.add((start, end))
        out.append(PredictedSpan(
            start=start,
            end=end,
            label=str(s.get("label", "OTHER")),
            text=surface,
        ))
    return out


def _robust_json_parse(content: str) -> dict:
    """Best-effort JSON parser that recovers from truncated LLM output.

    Tries in order:
      1. parse the whole content as JSON
      2. strip ``` fences and retry
      3. extract every well-formed span object via regex, even if the outer
         array / object was cut off by max_new_tokens.
    """
    candidates = [content]
    # Drop ``` fences and language tags
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.M | re.S)
    if stripped != content:
        candidates.append(stripped)
    # Grab the first top-level object
    m = re.search(r"\{.*\}", stripped, re.S)
    if m:
        candidates.append(m.group(0))

    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue

    # Truncated JSON: harvest complete span objects via regex.
    span_re = re.compile(
        r'\{\s*"start"\s*:\s*\d+\s*,\s*"end"\s*:\s*\d+\s*,'
        r'\s*"label"\s*:\s*"[^"]*"\s*,'
        r'\s*"text"\s*:\s*"(?:\\.|[^"\\])*"\s*\}',
        re.S,
    )
    objs = []
    for chunk in span_re.findall(stripped):
        try:
            objs.append(json.loads(chunk))
        except json.JSONDecodeError:
            continue
    return {"spans": objs}


def _mask(text: str, spans: list[PredictedSpan]) -> str:
    if not spans:
        return text
    ordered = sorted(spans, key=lambda s: s.start)
    out, cursor = [], 0
    for s in ordered:
        if s.start < cursor:
            continue
        out.append(text[cursor:s.start])
        out.append(f"[{s.label}]")
        cursor = s.end
    out.append(text[cursor:])
    return "".join(out)


# Pre-built configs for the four providers.
OPENAI = LLMConfig(
    name="openai_gpt",
    api_key_env="OPENAI_API_KEY",
    base_url_env=None,
    model_env="OPENAI_MODEL",
    default_model="gpt-5",
)
DEEPSEEK = LLMConfig(
    name="deepseek",
    api_key_env="DEEPSEEK_API_KEY",
    base_url_env="DEEPSEEK_BASE_URL",
    default_base_url="https://api.deepseek.com/v1",
    model_env="DEEPSEEK_MODEL",
    default_model="deepseek-v4-pro",
)
KIMI = LLMConfig(
    name="kimi",
    api_key_env="KIMI_API_KEY",
    base_url_env="KIMI_BASE_URL",
    default_base_url="https://api.moonshot.cn/v1",
    model_env="KIMI_MODEL",
    default_model="kimi-k2.6",
)
QWEN = LLMConfig(
    name="qwen",
    api_key_env="QWEN_API_KEY",
    base_url_env="QWEN_BASE_URL",
    default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_env="QWEN_MODEL",
    default_model="qwen3.7-max-preview",
)
