"""Local HuggingFace transformer LLM wrapped as a PHI detector.

Loads any chat-template-compatible decoder via `transformers.AutoModelForCausalLM`
and runs the same JSON-spans prompt used for the API-hosted LLMs. We share
parsing + offset-recovery with `systems.llm_openai_compatible` so the
LLM-as-detector pipeline is identical across providers.

Usage in runner:
    --systems gemma_e4b qwen3_35b moonlight deepseek_v2_lite
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from systems.base import DeIDSystem, Prediction
from systems.llm_openai_compatible import SYSTEM_PROMPT, _parse_spans


@dataclass
class LocalLLMConfig:
    name: str
    model_id: str
    max_new_tokens: int = 400
    dtype: str = "bfloat16"   # or "float16"; A100 prefers bf16
    device_map: str = "auto"
    trust_remote_code: bool = False


class LocalHFLLM(DeIDSystem):
    def __init__(self, cfg: LocalLLMConfig):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.dtype]
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, trust_remote_code=cfg.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            dtype=dtype,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
        )
        self._model.eval()
        self._cfg = cfg
        self.name = cfg.name

    def predict(self, text: str) -> Prediction:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        completion = self._tokenizer.decode(gen, skip_special_tokens=True)

        spans = _parse_spans(completion, text)
        transformed = _mask(text, spans)
        return Prediction(spans=spans, transformed_text=transformed, raw=completion)


def _mask(text: str, spans) -> str:
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


GEMMA_E4B = LocalLLMConfig(
    name="gemma_4_e4b",
    model_id="google/gemma-4-E4B-it",
)
QWEN3_35B_A3B = LocalLLMConfig(
    name="qwen3.5_35b_a3b",
    model_id="Qwen/Qwen3.5-35B-A3B",
)
MOONLIGHT = LocalLLMConfig(
    name="moonlight_16b_a3b",
    model_id="moonshotai/Moonlight-16B-A3B-Instruct",
    trust_remote_code=True,  # uses custom modeling code
)
DEEPSEEK_V2_LITE = LocalLLMConfig(
    name="deepseek_v2_lite",
    model_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
    trust_remote_code=True,
)
