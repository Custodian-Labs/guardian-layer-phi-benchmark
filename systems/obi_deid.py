"""OBI transformer baseline.

Uses `obi/deid_roberta_i2b2` (RoBERTa-large fine-tuned on i2b2 2014).
A ClinicalBERT variant is also available at `obi/deid_bert_i2b2`.

Install:
    pip install transformers torch
"""
from __future__ import annotations

from systems.base import DeIDSystem, PredictedSpan, Prediction


class OBIDeID(DeIDSystem):
    name = "obi_roberta_i2b2"

    def __init__(self, model_id: str = "obi/deid_roberta_i2b2", device: str | None = None):
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        import torch

        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(model_id)
        self._ner = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device,
        )
        self.name = model_id.split("/")[-1]

    def predict(self, text: str) -> Prediction:
        outs = self._ner(text)
        spans = [
            PredictedSpan(
                start=int(o["start"]),
                end=int(o["end"]),
                label=str(o["entity_group"]),
                text=o["word"],
                score=float(o["score"]),
            )
            for o in outs
        ]
        # Build a simple masked transformation: replace each span with [LABEL].
        transformed = _mask_spans(text, spans)
        return Prediction(spans=spans, transformed_text=transformed, raw=outs)


def _mask_spans(text: str, spans: list[PredictedSpan]) -> str:
    if not spans:
        return text
    ordered = sorted(spans, key=lambda s: s.start)
    out, cursor = [], 0
    for s in ordered:
        if s.start < cursor:
            continue  # overlap, skip
        out.append(text[cursor:s.start])
        out.append(f"[{s.label}]")
        cursor = s.end
    out.append(text[cursor:])
    return "".join(out)
