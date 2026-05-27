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


# Prompt tuned for reasoning-on models (Qwen-thinking). The default
# SYSTEM_PROMPT asks for character offsets; thinking models then waste their
# whole budget hand-counting indices and never reach the JSON. This variant:
#   - tells them not to compute offsets (the runner recovers them)
#   - explicitly caps the reasoning budget ("at most 3 short sentences")
#   - shows 2 worked examples so the format is unambiguous
#   - includes both a PHI+ and a hard-negative example
THINKING_SYSTEM_PROMPT = """You are a clinical de-identification engine.
Identify every span containing PHI per HIPAA Safe Harbor.

RULES (strict):
1. DO NOT compute character offsets. Use start=0 end=0 as placeholders;
   the calling system recovers offsets from the `text` field.
2. DO NOT enumerate characters or count indices.
3. Keep reasoning to at most 3 short sentences. Then OUTPUT the JSON.
4. End your response with the JSON. No extra prose after the JSON.

Allowed labels: NAME, DATE, LOCATION, PHONE, EMAIL, ID, AGE, OTHER.
Empty case: {"spans":[]}

EXAMPLE 1 (PHI present):
Input: "Treatment for Anna S. age 34, seen at Methodist Hospital on April 12, 2023."
Reasoning: Name Anna S., age 34, location Methodist Hospital, date April 12, 2023.
{"spans":[{"start":0,"end":0,"label":"NAME","text":"Anna S."},{"start":0,"end":0,"label":"AGE","text":"34"},{"start":0,"end":0,"label":"LOCATION","text":"Methodist Hospital"},{"start":0,"end":0,"label":"DATE","text":"April 12, 2023"}]}

EXAMPLE 2 (no PHI):
Input: "What are the guidelines for prescribing ACE inhibitors to a patient with hypertension?"
Reasoning: No personal identifiers; only generic clinical content.
{"spans":[]}"""


@dataclass
class LocalLLMConfig:
    name: str
    model_id: str
    max_new_tokens: int = 1000
    dtype: str = "bfloat16"   # or "float16"; A100 prefers bf16
    device_map: str = "auto"
    trust_remote_code: bool = False
    chat_template_kwargs: dict | None = None  # e.g. {"enable_thinking": False} for Qwen3
    system_prompt_override: str | None = None  # use this instead of SYSTEM_PROMPT
    max_memory: dict | None = None  # explicit per-device cap passed to from_pretrained


def _auto_max_memory(margin_gib: float = 3.0, frac: float = 0.92) -> dict:
    """Build a `max_memory` map from each visible GPU's *actually free* memory.

    On a shared cluster `device_map="auto"` is unsafe: it assumes it owns every
    GPU's full capacity and ignores other users' allocations, so a 60-70GB model
    OOMs even though no single GPU has that much free. Here we read live free
    memory (which already accounts for other processes) and cap each device to a
    conservative slice of it, letting a big model spread across the fragmented
    free space of several GPUs. A small CPU bucket is the last-resort overflow.
    """
    import torch

    mm: dict = {}
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)  # bytes, net of other procs
        usable_gib = max(0.0, (free / 1024**3 - margin_gib)) * frac
        mm[i] = f"{int(usable_gib)}GiB"
    mm["cpu"] = "96GiB"
    return mm


def _patch_dynamic_cache_compat() -> None:
    """Older custom modeling code (Moonlight / DeepSeek V2-Lite) calls
    DynamicCache APIs that transformers 4.43+ removed or renamed. Shim them
    back in so upstream weights work without patching the modeling files."""
    try:
        from transformers import DynamicCache
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(
                lambda self: self.get_seq_length() if hasattr(self, "get_seq_length") else 0
            )
        if not hasattr(DynamicCache, "get_usable_length"):
            def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
                try:
                    return self.get_seq_length(layer_idx)
                except Exception:
                    return self.get_seq_length() if hasattr(self, "get_seq_length") else 0
            DynamicCache.get_usable_length = _get_usable_length
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = lambda self: None
    except Exception:
        pass


class LocalHFLLM(DeIDSystem):
    def __init__(self, cfg: LocalLLMConfig):
        _patch_dynamic_cache_compat()
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.dtype]
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, trust_remote_code=cfg.trust_remote_code,
        )
        import os
        max_memory = cfg.max_memory
        if max_memory is None and os.environ.get("CUSTODIAN_AUTO_MAX_MEM") == "1":
            max_memory = _auto_max_memory()
            print(f"[local-hf] auto max_memory = {max_memory}")
        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            dtype=dtype,
            device_map=cfg.device_map,
            max_memory=max_memory,
            trust_remote_code=cfg.trust_remote_code,
        )
        self._model.eval()
        self._cfg = cfg
        self.name = cfg.name

    def predict(self, text: str) -> Prediction:
        import torch

        prompt_text = self._cfg.system_prompt_override or SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": text},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **(self._cfg.chat_template_kwargs or {}),
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
GEMMA_4_31B = LocalLLMConfig(
    name="gemma_4_31b",
    model_id="google/gemma-4-31B-it",
)
QWEN3_5_9B = LocalLLMConfig(
    name="qwen3.5_9b",
    model_id="Qwen/Qwen3.5-9B",
    chat_template_kwargs={"enable_thinking": False},
)
QWEN3_5_35B_A3B = LocalLLMConfig(
    name="qwen3.5_35b_a3b",
    model_id="Qwen/Qwen3.5-35B-A3B",
    chat_template_kwargs={"enable_thinking": False},
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
KIMI_VL_A3B = LocalLLMConfig(
    name="kimi_vl_a3b_thinking",
    model_id="moonshotai/Kimi-VL-A3B-Thinking-2506",
    trust_remote_code=True,
)
QWEN3_5_4B = LocalLLMConfig(
    name="qwen3.5_4b",
    model_id="Qwen/Qwen3.5-4B",
    chat_template_kwargs={"enable_thinking": False},
)
# Same Qwen weights but with the model's CoT mode left on. The first run at
# max_new_tokens=1200 hit the wall on 4/5 benchmarks; bumping to 4000 didn't
# help (model burns budget on character-index counting). Plan B fix: pair
# `enable_thinking=True` with a system prompt that explicitly forbids the
# offset-counting busywork and forces JSON on the first line of the answer.
QWEN3_5_4B_THINKING = LocalLLMConfig(
    name="qwen3.5_4b_thinking",
    model_id="Qwen/Qwen3.5-4B",
    chat_template_kwargs={"enable_thinking": True},
    max_new_tokens=2500,
    system_prompt_override=THINKING_SYSTEM_PROMPT,
)
# Meta Llama (standard chat template, no thinking mode, no remote code).
# 8B fits one GPU; 70B (~140GB bf16) needs the auto max_memory spread.
LLAMA_3_1_8B = LocalLLMConfig(
    name="llama3.1_8b",
    model_id="meta-llama/Llama-3.1-8B-Instruct",
)
LLAMA_3_3_70B = LocalLLMConfig(
    name="llama3.3_70b",
    model_id="meta-llama/Llama-3.3-70B-Instruct",
)
