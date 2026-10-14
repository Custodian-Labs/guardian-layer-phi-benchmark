"""CLI: custodian-video INPUT -o OUTPUT [--mode redact|transform] ..."""
from __future__ import annotations

import argparse

from .pipeline import Config, run


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="custodian-video",
        description="Video de-identification: faces, licence plates, on-screen text PII and "
                    "spoken PII. Fully local by default (--pii-backend local). Writes the masked "
                    "video and a JSON audit log.")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--audit", help="audit JSON path (default: <output>.audit.json)")
    ap.add_argument("--mode", choices=["redact", "transform"], default="redact",
                    help="redact = blur/pixelate/box + bleep; transform = realistic same-type surrogates")
    ap.add_argument("--channels", default="faces,plates,text,audio",
                    help="comma list of faces,plates,text,audio")
    ap.add_argument("--face-style", choices=["blur", "pixelate", "box"], default="blur")
    ap.add_argument("--text-style", choices=["blur", "pixelate", "box"], default="box")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--face-detector", choices=["egoblur", "centerface"], default="egoblur",
                    help="egoblur = Meta EgoBlur (Apache-2.0 weights, default); "
                         "centerface = v0.1 detector (WIDER-FACE-derived, research only)")
    ap.add_argument("--face-landmarks", choices=["yunet", "template", "none"], default="yunet",
                    help="5-point landmarks for EgoBlur face boxes (used by transform mode)")
    ap.add_argument("--plate-detector", choices=["egoblur", "yolov9"], default="egoblur")
    ap.add_argument("--face-threshold", type=float, help="default: EgoBlur 0.1 (its own default; recall first), CenterFace 0.35")
    ap.add_argument("--plate-threshold", type=float, help="default: EgoBlur 0.8, YOLOv9 0.35")
    ap.add_argument("--ocr-every", type=float, default=0.5, help="seconds between OCR samples")
    ap.add_argument("--llm-model", default="qwen2.5:7b", help="Ollama model for PII spans")
    ap.add_argument("--llm-url", default="http://localhost:11434")
    ap.add_argument("--no-llm", action="store_true", help="regex-only PII detection")
    ap.add_argument("--pii-backend", choices=["local", "guardian", "both", "guardian-replace"], default="local",
                    help="local = local LLM + regex (nothing leaves the machine); guardian = Custodian "
                         "Guardian Layer + regex net; both = Guardian Layer + local LLM + regex (union). "
                         "guardian-replace = spans from local LLM + regex (+ Guardian digit/@ spans), "
                         "replacement values from Guardian Layer when complete, local surrogate otherwise "
                         "(use with --mode transform).")
    ap.add_argument("--guardian-max-calls", type=int, default=500,
                    help="max Guardian Layer HTTP calls for this run (global cap: 2,000)")
    ap.add_argument("--guardian-cache", help="JSON cache of Guardian Layer results (stores rewritten text)")
    ap.add_argument("--scene-filter", choices=["vlm", "none"], default="vlm",
                    help="vlm = local VLM decides if flagged on-screen text is public signage (left "
                         "visible) or a document/screen/badge/whiteboard (masked)")
    ap.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--vlm-python", help="python with transformers>=4.57 for the VLM worker "
                                         "(default $CUSTODIAN_VLM_PYTHON or this interpreter)")
    ap.add_argument("--whisper-model", default="small.en")
    ap.add_argument("--tts-voice", help="Piper .onnx voice for spoken surrogates (transform mode)")
    ap.add_argument("--face-backend", default="sd15-keyframe", choices=["sd15-keyframe", "research-swap", "deepprivacy2"],
                    help="transform faces: sd15-keyframe (default) or research-swap (RESEARCH-ONLY, non-commercial "
                         "model licences: synthetic-identity HyperSwap/InsightFace face swap, see surrogate_face_swap.py)")
    ap.add_argument("--sd-strength", type=float, default=0.75)
    ap.add_argument("--face-mode", choices=["auto", "redact"], default="auto",
                    help="redact: always blur/box faces, even with --mode transform (text and speech surrogates only)")
    ap.add_argument("--surrogate-min-face", type=int, default=64,
                    help="transform: faces smaller than this (px, short side) are blurred instead")
    ap.add_argument("--surrogate-min-score", type=float, default=0.6,
                    help="transform: detector score below which a face is blurred instead")
    ap.add_argument("--surrogate-realism", choices=["v1", "v2"],
                    help="transform faces: v2 (default; attribute-conditioned ControlNet identity, "
                         "inner-face paste-back with colour/texture matching) or v1 (pre-2026-10); "
                         "default $CUSTODIAN_FACE_REALISM or v2")
    ap.add_argument("--face-recall-yunet", type=float, default=0.7,
                    help="fail-closed recall net: full-frame YuNet faces >= this score that EgoBlur missed "
                         "are blurred too (0 = off, the pre-2026-10-05 behaviour)")
    ap.add_argument("--small-face-extra", type=float, default=0.5,
                    help="extra mask padding for small faces (up to +X at <= 32 px, none at >= 80 px); "
                         "0 = the old fixed 1.25x face-box dilation without the anti-lag envelope")
    ap.add_argument("--face-fp-filter", choices=["person", "none"], default="person",
                    help="person-context filter for face false positives: a low-score face track with no "
                         "person head nearby (torchvision Faster R-CNN COCO person detector, full frame + "
                         "zoomed second look), off the frame edge, never seen by the YuNet recall net and "
                         "detected on >= 2 frames (or explained by an animal box) is left unmasked and "
                         "logged in the audit; anything uncertain stays masked")
    ap.add_argument("--face-fp-max-score", type=float, default=0.5,
                    help="face tracks scoring at/above this always stay masked (person filter)")
    ap.add_argument("--log-pii-text", action="store_true",
                    help="write raw PII strings into the audit log (default: hash + masked preview)")
    ap.add_argument("--no-audit-boxes", action="store_true")
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args(argv)
    cfg = Config(mode=a.mode, channels=tuple(c.strip() for c in a.channels.split(",") if c.strip()),
                 face_style=a.face_style, text_style=a.text_style, device=a.device,
                 face_detector=a.face_detector, face_landmarks=a.face_landmarks,
                 plate_detector=a.plate_detector,
                 face_threshold=a.face_threshold, plate_threshold=a.plate_threshold,
                 pii_backend=a.pii_backend, guardian_max_calls=a.guardian_max_calls,
                 guardian_cache=a.guardian_cache, scene_filter=a.scene_filter,
                 vlm_model=a.vlm_model, vlm_python=a.vlm_python,
                 ocr_every_s=a.ocr_every, llm_model=a.llm_model, llm_url=a.llm_url,
                 use_llm=not a.no_llm, whisper_model=a.whisper_model, tts_voice=a.tts_voice,
                 face_backend=a.face_backend, sd_strength=a.sd_strength,
                 face_mode=a.face_mode, surrogate_min_face=a.surrogate_min_face, surrogate_min_score=a.surrogate_min_score,
                 surrogate_realism=a.surrogate_realism, small_face_extra=a.small_face_extra,
                 face_recall_yunet=a.face_recall_yunet,
                 face_fp_filter=a.face_fp_filter, face_fp_max_score=a.face_fp_max_score,
                 log_pii_text=a.log_pii_text, audit_boxes=not a.no_audit_boxes,
                 max_frames=a.max_frames)
    run(a.input, a.output, cfg, a.audit)


if __name__ == "__main__":
    main()
