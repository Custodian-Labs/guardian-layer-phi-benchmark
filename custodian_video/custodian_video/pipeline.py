"""End-to-end local video de-identification pipeline.

  pass 1 (analyse): decode -> faces / plates every frame, OCR every `stride`
                    frames -> local-LLM PII spans; audio -> ASR -> PII spans
  link + densify   : tracks, gap interpolation, padding, smoothing, dilation
  (transform)      : per-track surrogate faces
  pass 2 (render)  : decode again -> apply masks -> H.264 via ffmpeg pipe
  mux + audit log  : processed audio + JSON log of every detection/action
By default nothing leaves the machine: models are local ONNX/TorchScript/CT2/
diffusers weights and the LLM is an Ollama server on localhost.  Only
`pii_backend=guardian|both` sends OCR lines / transcripts to the Custodian
Guardian Layer endpoint (hosted, or on-prem via CUSTODIAN_MASKING_BASE_URL).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .masking import redact_box, redact_quad, render_surrogate_text
from .pii import GuardianClient, PIIEngine, redact_preview, sha
from .tracking import Det, densify, link


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()   # static build with libx264
    except Exception:
        return "ffmpeg"


@dataclass
class Config:
    mode: str = "redact"                  # redact | transform
    channels: tuple = ("faces", "plates", "text", "audio")
    face_style: str = "blur"              # redact style for faces/plates: blur|pixelate|box
    text_style: str = "box"               # redact style for on-screen text
    device: str = "cuda"
    face_detector: str = "egoblur"       # egoblur (Apache-2.0) | centerface (research)
    face_landmarks: str = "yunet"        # landmarks for EgoBlur boxes: yunet | template | none
    plate_detector: str = "egoblur"      # egoblur (Apache-2.0) | yolov9
    face_threshold: float | None = None  # None -> detector default (EgoBlur 0.1, CenterFace 0.35)
    plate_threshold: float | None = None # None -> detector default (EgoBlur 0.8, YOLOv9 0.35)
    ocr_every_s: float = 0.5
    plate_every: int = 2
    ocr_device: str = "cpu"              # OCR runs every ~0.5 s only; CPU keeps GPU free
    llm_model: str = "qwen2.5:7b"
    llm_url: str = "http://localhost:11434"
    use_llm: bool = True
    pii_backend: str = "local"           # local | guardian | both | guardian-replace
    guardian_max_calls: int = 500        # per run (all runs together are capped at 2,000)
    guardian_cache: str | None = None    # optional on-disk cache (stores rewritten text!)
    scene_filter: str = "vlm"            # vlm | none : public-signage gate for on-screen text
    vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    vlm_python: str | None = None        # interpreter with transformers>=4.57 (else $CUSTODIAN_VLM_PYTHON)
    whisper_model: str = "small.en"
    tts_voice: str | None = None
    face_backend: str = "sd15-keyframe"
    sd_strength: float = 0.75
    face_mode: str = "auto"              # auto: follow --mode; redact: always mask faces (text/speech can still be transformed)
    surrogate_min_face: int = 64         # transform gate: smaller / lower-score faces are blurred
    surrogate_min_score: float = 0.6
    surrogate_realism: str | None = None # v1 | v2 (None -> $CUSTODIAN_FACE_REALISM or v2)
    face_recall_yunet: float = 0.7       # YuNet recall net for missed faces (blur only); 0 = off
    small_face_px: float = 80.0          # face boxes below this (px) get extra mask padding ...
    small_face_extra: float = 0.5        # ... up to +50 % (0 = old fixed 1.25x dilation)
    face_fp_filter: str = "person"       # person | none : person-context filter for face false positives
    face_fp_max_score: float = 0.5       # tracks scoring >= this always stay masked (person filter)
    log_pii_text: bool = False
    audit_boxes: bool = True
    crf: int = 20
    max_frames: int | None = None
    extra: dict = field(default_factory=dict)


class VideoWriter:
    def __init__(self, path, w, h, fps, crf=20):
        self.p = subprocess.Popen(
            [ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset",
             "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE)

    def write(self, fr):
        self.p.stdin.write(fr.tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def _sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(inp: str, out: str, cfg: Config, audit_path: str | None = None, verbose=True) -> dict:
    T = {}
    t_all = time.time()
    log = (lambda *a: print("[custodian-video]", *a, flush=True)) if verbose else (lambda *a: None)
    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        raise FileNotFoundError(inp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    gclient = None
    if cfg.pii_backend in ("guardian", "both", "guardian-replace"):
        gclient = GuardianClient(max_calls=cfg.guardian_max_calls, cache_path=cfg.guardian_cache)
    pii = PIIEngine(cfg.pii_backend, cfg.llm_model, cfg.llm_url, use_llm=cfg.use_llm, guardian=gclient)
    models = {"pii": pii.describe()}

    # ---------------- load detectors
    t0 = time.time()
    face_det = plate_det = text_det = None
    if "faces" in cfg.channels:
        from .detectors import make_face_detector
        face_det = make_face_detector(cfg.face_detector, cfg.device, cfg.face_threshold,
                                      cfg.face_landmarks, cfg.face_recall_yunet)
        models["faces"] = f"{face_det.name} on {face_det.provider}"
    if "plates" in cfg.channels:
        from .detectors import make_plate_detector
        plate_det = make_plate_detector(cfg.plate_detector, cfg.device, cfg.plate_threshold)
        models["plates"] = plate_det.name
    if "text" in cfg.channels:
        from .text_channel import OnScreenTextPII
        text_det = OnScreenTextPII(pii, cfg.ocr_device)
        models["text"] = text_det.name
    person_det, fp_params, persons, fp_info = None, None, {}, None
    if face_det is not None and cfg.face_fp_filter == "person":
        from .person_filter import FilterParams, PersonDetector
        fp_params = FilterParams(max_score=cfg.face_fp_max_score)
        fp_info = {"mode": "person", "params": dict(fp_params.__dict__)}
        try:
            person_det = PersonDetector(cfg.device, fp_params.person_threshold)
            fp_info["model"] = person_det.name
            models["face_fp_filter"] = person_det.name
        except Exception as e:  # fail closed: no filter -> every face track stays masked
            fp_info["error"] = f"person detector unavailable ({type(e).__name__}); all face tracks masked"
            log("face fp filter:", fp_info["error"])
    T["load_models_s"] = time.time() - t0

    # ---------------- pass 1: analyse
    stride = max(1, int(round(cfg.ocr_every_s * fps)))
    faces, plates, texts = {}, {}, {}
    text_frames = {}   # OCR frames with text-PII candidates (JPEG) for the scene gate
    t_face = t_plate = t_text = t_person = 0.0
    n = 0
    t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok or (cfg.max_frames and n >= cfg.max_frames):
            break
        if face_det:
            a = time.time(); faces[n] = face_det(fr, n); t_face += time.time() - a
            if person_det is not None and n % fp_params.every == 0:
                a = time.time()
                try:
                    persons[n] = person_det(fr)
                except Exception:   # frame missing from `persons` -> its tracks stay masked
                    pass
                t_person += time.time() - a
        if plate_det and n % cfg.plate_every == 0:
            a = time.time(); plates[n] = plate_det(fr, n); t_plate += time.time() - a
        if text_det and n % stride == 0:
            a = time.time(); texts[n] = text_det(fr, n); t_text += time.time() - a
            if texts[n] and cfg.scene_filter == "vlm":
                text_frames[n] = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
        n += 1
    cap.release()
    n_frames = n
    T.update(pass1_s=time.time() - t0, faces_s=t_face, plates_s=t_plate, text_ocr_llm_s=t_text)
    if person_det is not None:
        T["persons_s"] = t_person
    log(f"pass1: {n_frames} frames {W}x{H}@{fps:.2f} in {T['pass1_s']:.1f}s "
        f"(faces {t_face:.1f}s, plates {t_plate:.1f}s, text {t_text:.1f}s)")

    # free detector GPU memory (shared GPUs; SD/whisper need it next)
    import gc
    have_face, have_plate, have_text = face_det is not None, plate_det is not None, text_det is not None
    n_ocr = getattr(text_det, "n_ocr", 0)
    lm_stats = {k: getattr(face_det, k) for k in ("n_lm_yunet", "n_lm_template", "n_recall") if hasattr(face_det, k)}
    face_det = plate_det = text_det = None
    gc.collect()

    # ---------------- tracking / densify
    t0 = time.time()
    tracks = []
    kept_visible, scene_stats = [], None
    if have_face:
        ft = link(faces, iou_thr=0.2, max_gap=int(fps * 0.6))
        tracks += [densify(t, n_frames, max_interp_gap=int(fps * 0.6), pad_before=3, pad_after=3,
                           smooth_win=5, dilate=1.25, frame_wh=(W, H),
                           small_px=cfg.small_face_px if cfg.small_face_extra > 0 else 0.0,
                           small_extra=cfg.small_face_extra) for t in ft]
        if person_det is not None:
            a = time.time()
            ft_kept, fp_unmasked, fp_reason = _face_fp_filter(inp, ft, persons, person_det, fp_params,
                                                              (W, H), log)
            tracks = [t for t in tracks if t.label != "face"] + ft_kept
            fp_info.update(n_tracks=len(ft), n_unmasked=len(fp_unmasked), reasons=fp_reason["counts"],
                           unmasked=[_unmasked_entry(t, fp_reason["by_tid"][t.tid], fps) for t in fp_unmasked])
            for t in ft_kept:
                for d in t.dets.values():
                    d.meta["fp_filter"] = fp_reason["by_tid"][t.tid]
            T["face_fp_filter_s"] = time.time() - a
            log(f"face fp filter: {len(fp_unmasked)} of {len(ft)} face tracks left unmasked "
                f"({T['face_fp_filter_s']:.1f}s) {fp_reason['counts']}")
        person_det = None
    if have_plate:
        pt = link(plates, iou_thr=0.15, max_gap=int(fps * 0.6), start_id=10000)
        tracks += [densify(t, n_frames, max_interp_gap=int(fps * 0.6), pad_before=3, pad_after=3,
                           smooth_win=3, dilate=1.3, frame_wh=(W, H)) for t in pt]
    if have_text:
        key = lambda d: (d.meta["pii_label"], d.meta["text"].replace(" ", "").lower())
        tt = link(texts, iou_thr=0.3, max_gap=2 * stride + 1, key=key, start_id=20000)
        # scene-context gate: public signage (shop/street signs) stays visible
        if cfg.scene_filter == "vlm" and tt:
            from .scene_context import SceneContextGate
            a = time.time()
            gate = SceneContextGate(cfg.vlm_model, cfg.vlm_python)
            items = []
            for t in tt:
                d = max(t.dets.values(), key=lambda d: (d.score, -abs(d.frame - (t.first + t.last) / 2)))
                fr = cv2.imdecode(text_frames[d.frame], cv2.IMREAD_COLOR) if d.frame in text_frames else None
                if fr is not None:
                    items.append((str(t.tid), fr, d.meta.get("line_box", d.lm), d.meta.get("line", "")))
            decisions = gate.classify(items)
            for t in tt:
                dec = decisions.get(str(t.tid), {"category": "no_frame", "mask": True})
                for d in t.dets.values():
                    d.meta["scene"] = dec
            kept_visible = [t for t in tt if not t.dets[t.first].meta["scene"]["mask"]]
            tt = [t for t in tt if t.dets[t.first].meta["scene"]["mask"]]
            models["text_scene_gate"] = gate.name
            T["scene_gate_s"] = time.time() - a
            scene_stats = gate.stats
            log(f"scene gate: {len(kept_visible)} signage tracks left visible, {len(tt)} masked "
                f"({T['scene_gate_s']:.1f}s)")
        # conservative: extend a full OCR stride before/after (text may appear between samples)
        tracks += [densify(t, n_frames, max_interp_gap=2 * stride + 1, pad_before=stride,
                           pad_after=stride, smooth_win=1, dilate=1.08, frame_wh=(W, H)) for t in tt]
    T["tracking_s"] = time.time() - t0 - T.get("scene_gate_s", 0.0)

    # guardian-replace: a value first seen with the local fallback may have been
    # upgraded to a Guardian Layer surrogate on a later frame; give every text
    # track the final registry value, then freeze it so speech uses the same one
    if cfg.pii_backend == "guardian-replace":
        for t in tracks:
            if t.label == "text_pii":
                for d in t.dets.values():
                    fv = pii.final_value(d.meta["text"])
                    if fv:
                        d.meta["surrogate"], d.meta["surrogate_source"], d.meta["surrogate_note"] = fv
    pii.frozen = True

    # ---------------- transform prep
    face_backend = None
    if cfg.mode == "transform" and cfg.face_mode != "redact" and have_face:
        t0 = time.time()
        if cfg.face_backend == "sd15-keyframe":
            from .surrogate_face import SDKeyframeSurrogate
            face_backend = SDKeyframeSurrogate(cfg.device, strength=cfg.sd_strength,
                                               min_face=cfg.surrogate_min_face,
                                               min_score=cfg.surrogate_min_score,
                                               realism=cfg.surrogate_realism)
        elif cfg.face_backend == "research-swap":
            # RESEARCH-ONLY (non-commercial model licences); opt-in, see surrogate_face_swap.py
            from .surrogate_face_swap import ResearchSwapSurrogate
            face_backend = ResearchSwapSurrogate(cfg.device, min_face=cfg.surrogate_min_face,
                                                 min_score=cfg.surrogate_min_score)
        elif cfg.face_backend == "deepprivacy2":
            from .surrogate_face import DeepPrivacy2Backend
            face_backend = DeepPrivacy2Backend()
        face_backend.prepare(inp, [t for t in tracks if t.label == "face"])
        models["face_surrogate"] = face_backend.name
        T["face_surrogate_prep_s"] = time.time() - t0
        if hasattr(face_backend, "pipe"):
            face_backend.pipe = None
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
        log(f"surrogate faces generated for {len(face_backend.surr)} tracks "
            f"in {T['face_surrogate_prep_s']:.1f}s")

    # ---------------- pass 2: render
    per_frame: dict[int, list] = {}
    for t in tracks:
        for f, v in t.dense.items():
            per_frame.setdefault(f, []).append((t, v))
    order = {"text_pii": 0, "license_plate": 1, "face": 2}
    actions: dict[int, set] = {}
    tmpdir = tempfile.mkdtemp(prefix="cvg_")
    vid_tmp = os.path.join(tmpdir, "video.mp4")
    cap = cv2.VideoCapture(inp)
    vw = VideoWriter(vid_tmp, W, H, fps, cfg.crf)
    t0 = time.time()
    for f in range(n_frames):
        ok, fr = cap.read()
        if not ok:
            break
        for t, (box, lm, src) in sorted(per_frame.get(f, []), key=lambda x: order[x[0].label]):
            if t.label == "text_pii":
                if cfg.mode == "transform":
                    render_surrogate_text(fr, lm, t.dets[t.first].meta["surrogate"]); act = "surrogate_text"
                else:
                    redact_quad(fr, lm, cfg.text_style); act = cfg.text_style
            elif t.label == "face" and face_backend is not None and face_backend.apply(fr, t, box, lm):
                act = "surrogate_face"
            else:
                faces_redact = cfg.mode == "redact" or cfg.face_mode == "redact"
                style = cfg.face_style if faces_redact else "blur"
                redact_box(fr, box, style, "ellipse" if t.label == "face" else "rect")
                act = style if faces_redact else "blur_fallback(face too small for surrogate)"
            actions.setdefault(t.tid, set()).add(act)
        vw.write(fr)
    vw.close(); cap.release()
    T["pass2_render_s"] = time.time() - t0

    # ---------------- audio
    audio_events, transcript_len = [], 0
    audio_out = None
    ff = ffmpeg_exe()
    has_audio = False
    if "audio" in cfg.channels:
        from .audio_channel import AudioPII, extract_audio
        wav16, wavfull = os.path.join(tmpdir, "a16.wav"), os.path.join(tmpdir, "afull.wav")
        has_audio = extract_audio(ff, inp, wav16, 16000, mono=True) and extract_audio(ff, inp, wavfull)
        if has_audio:
            t0 = time.time()
            ap = AudioPII(pii, cfg.whisper_model, cfg.device,
                          cfg.tts_voice if cfg.mode == "transform" else None)
            words = ap.transcribe(wav16)
            audio_events = ap.find_events(words, cfg.mode)
            transcript_len = len(words)
            audio_out = os.path.join(tmpdir, "aout.wav")
            ap.apply(wavfull, audio_out, audio_events)
            models["audio_asr"] = ap.name
            if ap.tts is not None:
                models["audio_tts"] = "Piper TTS (piper-tts, GPL-3.0; optional plug-in)"
            T["audio_s"] = time.time() - t0
            log(f"audio: {transcript_len} words, {len(audio_events)} PII events in {T['audio_s']:.1f}s")
    elif _has_audio_stream(ff, inp):
        has_audio = True

    # ---------------- mux
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if audio_out:
        cmd = [ff, "-v", "error", "-y", "-i", vid_tmp, "-i", audio_out, "-map", "0:v", "-map", "1:a",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", out]
    elif has_audio and "audio" not in cfg.channels:
        cmd = [ff, "-v", "error", "-y", "-i", vid_tmp, "-i", inp, "-map", "0:v", "-map", "1:a?",
               "-c:v", "copy", "-c:a", "aac", out]
    else:
        cmd = [ff, "-v", "error", "-y", "-i", vid_tmp, "-c", "copy", out]
    subprocess.run(cmd, check=True)
    T["total_s"] = time.time() - t_all
    T["fps_end_to_end"] = n_frames / T["total_s"]
    T["fps_pass1_detect"] = n_frames / max(1e-6, T["pass1_s"])
    T["fps_pass2_render"] = n_frames / max(1e-6, T["pass2_render_s"])
    T["llm_calls"] = pii.n_calls
    T["llm_s"] = pii.llm_seconds

    # ---------------- audit log
    dets = []
    for t in tracks:
        fs = sorted(t.dense)
        srcs = [t.dense[f][2] for f in fs]
        e = {"id": f"{t.label}-{t.tid}", "channel": {"face": "faces", "license_plate": "plates",
                                                    "text_pii": "text"}[t.label],
             "type": t.label if t.label != "text_pii" else f"onscreen_{t.dets[t.first].meta['pii_label'].lower()}",
             "frame_start": fs[0], "frame_end": fs[-1],
             "t_start": round(fs[0] / fps, 3), "t_end": round((fs[-1] + 1) / fps, 3),
             "n_frames_masked": len(fs), "n_detected": srcs.count("det"),
             "n_interpolated": srcs.count("interp"), "n_padded": srcs.count("pad"),
             "max_score": round(max(d.score for d in t.dets.values()), 3),
             "action": sorted(actions.get(t.tid, []))}
        if t.label == "face":
            # found only by the YuNet recall net (EgoBlur missed it): always blurred
            e["recall_net_only"] = all(d.meta.get("src") == "yunet_recall" for d in t.dets.values())
            if fp_info is not None:
                e["fp_filter"] = t.dets[t.first].meta.get("fp_filter", "not_evaluated")
        if t.label == "text_pii":
            m = t.dets[t.first].meta
            e.update(pii_label=m["pii_label"], detector=m["source"], text_sha16=sha(m["text"]),
                     value_key16=m.get("value_key16"),
                     text_preview=m["text"] if cfg.log_pii_text else redact_preview(m["text"]),
                     surrogate=m["surrogate"] if cfg.mode == "transform" else None,
                     surrogate_planned=m["surrogate"], surrogate_source=m.get("surrogate_source"),
                     surrogate_note=m.get("surrogate_note"),
                     scene=m.get("scene"))
        if cfg.audit_boxes:
            e["boxes"] = [[f, *[int(round(v)) for v in t.dense[f][0]], t.dense[f][2]] for f in fs]
        dets.append(e)
    for t in kept_visible:   # logged for human review, NOT masked
        m = t.dets[t.first].meta
        fs = sorted(t.dets)
        dets.append({"id": f"text_pii-{t.tid}", "channel": "text",
                     "type": f"onscreen_{m['pii_label'].lower()}", "frame_start": fs[0],
                     "frame_end": fs[-1], "t_start": round(fs[0] / fps, 3),
                     "t_end": round((fs[-1] + 1) / fps, 3), "pii_label": m["pii_label"],
                     "detector": m["source"], "text_sha16": sha(m["text"]),
                     "text_preview": m["text"] if cfg.log_pii_text else redact_preview(m["text"]),
                     "scene": m.get("scene"), "action": ["kept_visible(scene_gate)"]})
    from .audio_channel import AudioPII as _A  # noqa
    dets += [_A.to_audit(ev, cfg.log_pii_text) for ev in audio_events]
    audit = {
        "tool": f"custodian_video {__version__}", "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "input": os.path.abspath(inp), "input_sha256": _sha_file(inp),
        "output": os.path.abspath(out), "output_sha256": _sha_file(out),
        "mode": cfg.mode,
        "network": ("local only (models on disk; LLM at %s)" % cfg.llm_url) if cfg.pii_backend == "local"
        else "OCR lines + transcript sent to Custodian Guardian Layer (%s); everything else local"
             % os.environ.get("CUSTODIAN_MASKING_BASE_URL", "hosted endpoint"),
        "video": {"fps": fps, "frames": n_frames, "width": W, "height": H,
                  "duration_s": round(n_frames / fps, 3), "has_audio": has_audio},
        "settings": {k: v for k, v in cfg.__dict__.items() if k != "extra"},
        "models": models,
        "summary": {
            "face_tracks": sum(t.label == "face" for t in tracks),
            "plate_tracks": sum(t.label == "license_plate" for t in tracks),
            "onscreen_text_pii_tracks": sum(t.label == "text_pii" for t in tracks),
            "onscreen_text_tracks_kept_visible_scene_gate": len(kept_visible),
            "audio_pii_events": len(audio_events),
            "face_tracks_unmasked_fp_filter": (fp_info or {}).get("n_unmasked", 0),
            "frames_with_any_mask": len(per_frame),
            "ocr_frames": n_ocr, "asr_words": transcript_len},
        "timing": {k: round(v, 3) if isinstance(v, float) else v for k, v in T.items()},
        "pii_engine": pii.audit(),
        "scene_gate": scene_stats,
        "face_landmarks": lm_stats or None,
        # face tracks the person filter left UNMASKED (geometry only, for human review);
        # deliberately not in "detections", which lists only masked regions
        "face_fp_filter": fp_info,
        "detections": dets,
    }
    audit_path = audit_path or str(Path(out).with_suffix(".audit.json"))
    Path(audit_path).write_text(json.dumps(audit, indent=1))
    log(f"done: {out} ({T['fps_end_to_end']:.2f} fps end-to-end); audit -> {audit_path}")
    return audit


def _face_fp_filter(inp, ft, persons, person_det, p, frame_wh, log):
    """Person-context filter (person_filter.py). Fails closed: any error keeps the track."""
    from .person_filter import look_frames, prescreen, second_look
    by_tid, cands = {}, []
    # confident faces (frame -> [(tid, raw box)]): they explain their person's head region
    conf_all = {}
    for t in ft:
        for f, d in t.dets.items():
            if d.score >= p.conf_face and d.meta.get("src") != "yunet_recall":
                conf_all.setdefault(f, []).append((t.tid, np.asarray(d.box, np.float32)[:4]))

    def conf_for(tid):
        return {f: np.array([b for k, b in L if k != tid], np.float32).reshape(-1, 4) for f, L in conf_all.items()}
    for t in ft:
        det = {f: v[0] for f, v in t.dense.items() if v[2] == "det"}
        raw = {f: np.asarray(t.dets[f].box, np.float32)[:4] for f in det if f in t.dets}
        recall = any(d.meta.get("src") == "yunet_recall" for d in t.dets.values())
        ms = max(d.score for d in t.dets.values())
        try:
            cf = conf_for(t.tid)
            ok, why = prescreen(det, persons, frame_wh, ms, recall, p, raw_boxes=raw, conf_faces=cf)
        except Exception as e:  # noqa: BLE001
            ok, why = False, f"error:{type(e).__name__}"
        by_tid[t.tid] = why
        if ok:
            cands.append((t, det, raw, ms, cf))
    frames = {}
    need = {f for _, det, *_ in cands for f in look_frames(det, p)}
    if need:
        cap = cv2.VideoCapture(inp)
        f, last = 0, max(need)
        while f <= last:
            ok, fr = cap.read()
            if not ok:
                break
            if f in need:
                frames[f] = fr
            f += 1
        cap.release()
    unmasked = []
    for t, det, raw, ms, cf in cands:
        try:
            un, why = second_look(det, frames, person_det, p, max_score=ms, raw_boxes=raw, conf_faces=cf)
        except Exception as e:  # noqa: BLE001
            un, why = False, f"error:{type(e).__name__}"
        by_tid[t.tid] = why
        if un:
            unmasked.append(t)
    gone = {t.tid for t in unmasked}
    counts = {}
    for t in ft:
        k = "unmasked" if t.tid in gone else by_tid[t.tid].split(";")[0]
        counts[k] = counts.get(k, 0) + 1
    return [t for t in ft if t.tid not in gone], unmasked, {"by_tid": by_tid, "counts": counts}


def _unmasked_entry(t, reason, fps):
    fs = sorted(t.dets)
    return {"id": f"face-{t.tid}", "frame_start": fs[0], "frame_end": fs[-1],
            "t_start": round(fs[0] / fps, 3), "t_end": round((fs[-1] + 1) / fps, 3),
            "n_detected": len(fs), "max_score": round(max(d.score for d in t.dets.values()), 3),
            "reason": reason, "action": ["kept_visible(face_fp_filter)"],
            "det_boxes": [[f, *[int(round(v)) for v in t.dense[f][0]]] for f in fs if f in t.dense]}


def _has_audio_stream(ff: str, inp: str) -> bool:
    r = subprocess.run([ff, "-i", inp], capture_output=True, text=True)
    return "Audio:" in r.stderr
