"""Quick quantitative checks of a de-identified video against its original.

Privacy  (lower is better)
  face_reid_rate      : share of original faces whose SFace embedding, taken at
                        the SAME location in the output (attacker knows where
                        the face is), still matches the original (cos > 0.363,
                        OpenCV-zoo SFace threshold).
  text_pii_leak       : ground-truth on-screen PII strings still OCR-readable
  audio_pii_leak      : ground-truth spoken PII tokens still in a re-transcript
Utility  (the video analogue of the text paper's "surrogate stays detectable")
  face_still_detected : share of original faces for which an independent face
                        detector (YuNet) still finds a face in the output
  surrogate_id_consistency : mean cos between output-face embeddings of the
                        same track across frames (temporal identity stability)
  text_non_pii_kept   : benign on-screen lines still readable
  text/audio_surrogate_detected : surrogate values re-detected as PII by the
                        same local-LLM detector (detectability preserved)
Also: mask coverage of YuNet reference faces (recall proxy) from the audit log.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

MODELS = Path.home() / ".cache" / "custodian_video"
SFACE_T = 0.363


def yunet(w, h, thr=0.6):
    d = cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2023mar.onnx"), "", (w, h),
                                  thr, 0.3, 5000)
    return d


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def frames(path, idxs):
    cap = cv2.VideoCapture(str(path))
    out, i, want = {}, 0, set(idxs)
    while True:
        ok, fr = cap.read()
        if not ok or i > max(idxs):
            break
        if i in want:
            out[i] = fr
        i += 1
    cap.release()
    return out


def face_eval(orig, outs, audit, step, min_face):
    cap = cv2.VideoCapture(str(orig)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
    idxs = list(range(0, n, step))
    fo = frames(orig, idxs)
    det = yunet(W, H)
    rec = cv2.FaceRecognizerSF.create(str(MODELS / "face_recognition_sface_2021dec.onnx"), "")
    boxes_by_frame = {}
    for d in audit["detections"]:
        if d.get("channel") == "faces":
            for b in d.get("boxes", []):
                boxes_by_frame.setdefault(b[0], []).append((b[1:5], d["id"]))
    ref = []
    for i, fr in fo.items():
        _, faces = det.detect(fr)
        for f in (faces if faces is not None else []):
            if min(f[2], f[3]) < min_face:
                continue
            ref.append((i, f))
    covered, cov_flag = 0, []
    for i, f in ref:
        cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
        c = any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b, _ in boxes_by_frame.get(i, []))
        covered += c
        cov_flag.append(c)
    res = {"reference_faces(YuNet)": len(ref),
           "mask_coverage_of_reference_faces": round(covered / max(1, len(ref)), 3)}
    for lab, path in outs.items():
        fx = frames(path, idxs)
        sims, still, per_track = [], 0, {}
        for i, f in ref:
            a = rec.feature(rec.alignCrop(fo[i], f))
            b = rec.feature(rec.alignCrop(fx[i], f))
            sims.append(float(rec.match(a, b, cv2.FaceRecognizerSF_FR_COSINE)))
            _, fo2 = det.detect(fx[i])
            bb = [f[0], f[1], f[0] + f[2], f[1] + f[3]]
            if fo2 is not None and any(iou(bb, [g[0], g[1], g[0] + g[2], g[1] + g[3]]) > 0.3 for g in fo2):
                still += 1
            tid = next((t for b, t in boxes_by_frame.get(i, [])
                        if b[0] <= f[0] + f[2] / 2 <= b[2] and b[1] <= f[1] + f[3] / 2 <= b[3]), None)
            if tid:
                per_track.setdefault(tid, []).append(b)
        # breakdown by face size + impostor baseline (orig face vs a DIFFERENT
        # person's orig face in the same frame): SFace is unreliable on tiny faces
        sz = np.array([min(f[2], f[3]) for _, f in ref])
        sims_arr = np.array(sims)
        by_size = {}
        for lo, hi in ((0, 32), (32, 64), (64, 10000)):
            m = (sz >= lo) & (sz < hi) & np.array(cov_flag, bool)
            if m.any():
                by_size[f"{lo}-{hi}px"] = {"n": int(m.sum()),
                                           "reid_rate": round(float((sims_arr[m] > SFACE_T).mean()), 3),
                                           "mean_cos": round(float(sims_arr[m].mean()), 3)}
        cons = []
        for tid, embs in per_track.items():
            for j in range(1, len(embs)):
                cons.append(float(rec.match(embs[0], embs[j], cv2.FaceRecognizerSF_FR_COSINE)))
        sims = np.array(sims) if sims else np.zeros(0)
        cf = np.array(cov_flag, bool)
        res[lab] = {"face_reid_rate_all_ref_faces": round(float((sims > SFACE_T).mean()) if len(sims) else 0, 3),
                    "face_reid_rate_masked_faces": round(float((sims[cf] > SFACE_T).mean()) if cf.any() else 0, 3),
                    "mean_cos_to_original_masked_faces": round(float(sims[cf].mean()) if cf.any() else 0, 3),
                    "face_still_detected": round(still / max(1, len(ref)), 3),
                    "output_identity_consistency": round(float(np.mean(cons)), 3) if cons else None,
                    "by_face_size_masked": by_size}
    # impostor baseline: different people, both ORIGINAL, same frame
    imp = {}
    feats = {}
    for i, f in ref:
        feats.setdefault(i, []).append((min(f[2], f[3]), rec.feature(rec.alignCrop(fo[i], f))))
    for i, L in feats.items():
        for a in range(len(L)):
            for b in range(a + 1, len(L)):
                key = "small(<64px)" if min(L[a][0], L[b][0]) < 64 else "large"
                imp.setdefault(key, []).append(float(rec.match(L[a][1], L[b][1],
                                                               cv2.FaceRecognizerSF_FR_COSINE)))
    res["impostor_baseline_orig_vs_other_person"] = {
        k: {"n_pairs": len(v), "mean_cos": round(float(np.mean(v)), 3),
            "false_match_rate": round(float((np.array(v) > SFACE_T).mean()), 3)} for k, v in imp.items()}
    return res


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def ocr_texts(path, step):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR
    ocr = RapidOCR(params={"Rec.lang_type": LangRec.EN, "Rec.ocr_version": OCRVersion.PPOCRV5,
                           "Rec.model_type": ModelType.MOBILE, "Det.ocr_version": OCRVersion.PPOCRV5,
                           "Det.lang_type": LangDet.CH, "Det.model_type": ModelType.MOBILE,
                           "Global.log_level": "error"})
    cap = cv2.VideoCapture(str(path)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    out = []
    for i, fr in frames(path, list(range(0, n, step))).items():
        r = ocr(fr)
        out.append(list(r.txts) if r.txts else [])
    return out


def fuzzy_in(needle, lines):
    n = norm(needle)
    for l in lines:
        L = norm(l)
        if n in L:
            return True
        if len(n) >= 6 and any(difflib.SequenceMatcher(None, n, L[k:k + len(n)]).ratio() > 0.85
                               for k in range(0, max(1, len(L) - len(n) + 1))):
            return True
    return False


def text_eval(orig, outs, gt, step, pii, audits):
    res = {}
    for lab, path in {"original": orig, **outs}.items():
        per = ocr_texts(path, step)
        leak = sum(any(fuzzy_in(p, ls) for ls in per) for p in gt["onscreen_pii"])
        kept = np.mean([np.mean([fuzzy_in(b, ls) for ls in per]) for b in gt["onscreen_non_pii"]])
        r = {"gt_pii_strings_readable": f"{leak}/{len(gt['onscreen_pii'])}",
             "non_pii_lines_readable_frac": round(float(kept), 3)}
        if lab != "original" and pii is not None:
            surr = [d["surrogate"] for d in audits[lab]["detections"]
                    if d.get("channel") == "text" and d.get("surrogate")]
            if surr:
                lines = sorted({l for ls in per for l in ls})
                spans = [s for sp in pii.detect_many(lines) for s in sp]
                found = [any(norm(s) in norm(x.text) or norm(x.text) in norm(s) for x in spans
                             if len(norm(x.text)) > 2) for s in surr]
                r["surrogates_redetected_as_pii"] = f"{sum(found)}/{len(surr)}"
        res[lab] = r
    return res


def audio_eval(orig, outs, gt, pii, audits, device):
    from faster_whisper import WhisperModel
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        m = WhisperModel("small.en", device=device, compute_type="float16" if device == "cuda" else "int8")
    except RuntimeError:
        m = WhisperModel("small.en", device="cpu", compute_type="int8")
    toks = ["margaret", "holloway", "415", "5550199", "0199", "88427139", "8842"]
    res = {}
    for lab, path in {"original": orig, **outs}.items():
        with tempfile.TemporaryDirectory() as td:
            w = os.path.join(td, "a.wav")
            subprocess.run([ff, "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", "16000", w],
                           check=True)
            text = " ".join(s.text for s in m.transcribe(w)[0])
        t = norm(text)
        leaked = [k for k in toks if k in t]
        r = {"transcript": text.strip(), "gt_pii_tokens_heard": f"{len(leaked)}/{len(toks)}",
             "leaked": leaked}
        if lab != "original" and pii is not None:
            surr = [d["surrogate"] for d in audits[lab]["detections"]
                    if d.get("channel") == "audio" and d.get("surrogate")]
            if surr:
                spans = pii.detect(text)
                r["pii_spans_in_retranscript"] = [s.label for s in spans]
                r["surrogate_events"] = len(surr)
        res[lab] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--outs", nargs="+", required=True, help="label=path (audit at .audit.json)")
    ap.add_argument("--gt", help="ground_truth.json (for text/audio checks)")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--min-face", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    outs = dict(o.split("=", 1) for o in a.outs)
    audits = {k: json.loads(Path(v).with_suffix(".audit.json").read_text()) for k, v in outs.items()}
    res = {"orig": a.orig, "faces": face_eval(a.orig, outs, next(iter(audits.values())), a.step, a.min_face)}
    if a.gt:
        gt = json.loads(Path(a.gt).read_text()).get(Path(a.orig).name)
        if gt:
            from custodian_video.pii import LocalLLMPII
            pii = LocalLLMPII()
            res["text"] = text_eval(a.orig, outs, gt, a.step, pii, audits)
            res["audio"] = audio_eval(a.orig, outs, gt, pii, audits, a.device)
    for k, v in audits.items():
        res.setdefault("throughput", {})[k] = {x: v["timing"].get(x) for x in
                                               ("fps_end_to_end", "fps_pass1_detect", "total_s",
                                                "faces_s", "plates_s", "text_ocr_llm_s", "audio_s",
                                                "face_surrogate_prep_s")}
        res.setdefault("audit_summary", {})[k] = v["summary"]
    s = json.dumps(res, indent=1)
    print(s)
    if a.json_out:
        Path(a.json_out).write_text(s)


if __name__ == "__main__":
    main()
