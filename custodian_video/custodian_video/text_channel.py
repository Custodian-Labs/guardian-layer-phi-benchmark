"""On-screen text channel: OCR (RapidOCR = PaddleOCR PP-OCRv5 models in ONNX,
Apache-2.0) every `stride` frames -> PII span detection with the local LLM ->
word-level quads of ONLY the PII words (field labels and benign text are left
intact, mirroring span-level text de-identification)."""
from __future__ import annotations

import re

import numpy as np

from .pii import Span, surrogate, value_key
from .tracking import Det


class OnScreenTextPII:
    name = "RapidOCR (PP-OCRv5 det + en rec, Apache-2.0)"

    def __init__(self, pii, device: str = "cpu", min_conf: float = 0.5):
        from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR
        params = {"Rec.lang_type": LangRec.EN, "Rec.ocr_version": OCRVersion.PPOCRV5,
                  "Rec.model_type": ModelType.MOBILE, "Det.ocr_version": OCRVersion.PPOCRV5,
                  "Det.lang_type": LangDet.CH, "Det.model_type": ModelType.MOBILE,
                  "Global.log_level": "error"}
        if device != "cpu":
            params["EngineConfig.onnxruntime.use_cuda"] = True
        try:
            self.ocr = RapidOCR(params=params)
        except Exception:
            params.pop("EngineConfig.onnxruntime.use_cuda", None)
            self.ocr = RapidOCR(params=params)
        self.pii, self.min_conf = pii, min_conf
        self.n_ocr = 0
        self.lines_seen = 0

    def __call__(self, frame_bgr: np.ndarray, fidx: int) -> list[Det]:
        r = self.ocr(frame_bgr, return_word_box=True)
        self.n_ocr += 1
        if r is None or r.txts is None or len(r.txts) == 0:
            return []
        lines = [(t, np.asarray(b, np.float32), s, (r.word_results[i] if r.word_results else None))
                 for i, (t, b, s) in enumerate(zip(r.txts, r.boxes, r.scores)) if s >= self.min_conf]
        self.lines_seen += len(lines)
        spans_per_line = self.pii.detect_many([l[0] for l in lines])
        out = []
        self.last_line_boxes = [(t, b) for t, b, _, _ in lines]
        for (text, lbox, score, words), spans in zip(lines, spans_per_line):
            for sp in spans:
                if sp.label == "OTHER" and not re.search(r"[\d@/]", sp.text):
                    continue  # OCR: drop word-only "OTHER" (signage, slogans)
                quad = _span_quad(text, lbox, words, sp)
                if quad is None:
                    continue
                d = Det(fidx, np.concatenate([quad.min(0), quad.max(0)]), float(score),
                        "text_pii", quad)
                d.meta = {"pii_label": sp.label, "text": sp.text, "line": text,
                          "line_box": lbox, "source": sp.source,
                          "surrogate": sp.surrogate or surrogate(sp.label, sp.text),
                          "surrogate_source": sp.surrogate_source or "local",
                          "surrogate_note": sp.surrogate_note,
                          "value_key16": _vk(sp.text)}
                out.append(d)
        return out


def _span_quad(text: str, lbox: np.ndarray, words, sp: Span) -> np.ndarray | None:
    """Quad (TL, TR, BR, BL) covering the characters [sp.start, sp.end) of a line."""
    if words:
        pos, hit = 0, []
        for w in words:
            wt, _, wq = w[0], w[1], np.asarray(w[2], np.float32)
            i = text.find(wt, pos)
            if i < 0:
                continue
            pos = i + len(wt)
            if i < sp.end and i + len(wt) > sp.start:
                # trim partially-covered word proportionally
                a = max(0, sp.start - i) / max(1, len(wt))
                b = min(len(wt), sp.end - i) / max(1, len(wt))
                tl, tr, br, bl = wq
                hit.append(np.array([tl + (tr - tl) * a, tl + (tr - tl) * b,
                                     bl + (br - bl) * b, bl + (br - bl) * a]))
        if hit:
            return np.array([hit[0][0], hit[-1][1], hit[-1][2], hit[0][3]], np.float32)
    # fallback: proportional split of the line box
    n = max(1, len(text))
    a, b = sp.start / n, sp.end / n
    tl, tr, br, bl = lbox
    return np.array([tl + (tr - tl) * a, tl + (tr - tl) * b, bl + (br - bl) * b,
                     bl + (br - bl) * a], np.float32)


def _vk(text: str) -> str:
    import hashlib
    return hashlib.sha256(value_key(text).encode()).hexdigest()[:16]
