"""Face-only transform run (static vs re-enacted surrogates) for experiment face_d.

Mirrors pipeline.py's face path (CenterFace every frame -> link -> densify with
the same parameters -> backend.prepare -> per-frame backend.apply / blur) without
importing pipeline.py, and writes <out>.mp4 + <out>.audit.json (format read by
demo/evaluate.py) + <out>.timing.json.

  python experiments/face_d/run_face_d.py demo/inputs/clip1_interview_synthpii.mp4 \
      -o experiments/face_d/out/clip1_reenact.mp4 --motion reenact
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from custodian_video.masking import redact_box  # noqa: E402
from custodian_video.tracking import densify, link  # noqa: E402


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def detect_tracks(inp, cache, device="cuda", max_frames=None):
    if cache and Path(cache).exists():
        return pickle.loads(Path(cache).read_bytes())
    from custodian_video.detectors import FaceDetector
    det = FaceDetector(device, 0.35)
    cap = cv2.VideoCapture(inp)
    fps = cap.get(cv2.CAP_PROP_FPS); W, H = int(cap.get(3)), int(cap.get(4))
    faces, n, t0 = {}, 0, time.time()
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        faces[n] = det(fr, n); n += 1
    cap.release()
    ft = link(faces, iou_thr=0.2, max_gap=int(fps * 0.6))
    tracks = [densify(t, n, max_interp_gap=int(fps * 0.6), pad_before=3, pad_after=3,
                      smooth_win=5, dilate=1.25, frame_wh=(W, H)) for t in ft]
    res = dict(tracks=tracks, n=n, fps=fps, W=W, H=H, det_s=time.time() - t0)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_bytes(pickle.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--motion", choices=["static", "reenact"], default="reenact")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sd-strength", type=float, default=0.75)
    ap.add_argument("--lip-gain", type=float, default=1.0)
    ap.add_argument("--drive", default="hybrid", choices=["hybrid", "absolute", "relative"])
    ap.add_argument("--mask-grow", type=float, default=1.10)
    ap.add_argument("--guard-thr", type=float, default=0.30, help="identity guard; <=0 disables")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--cache", help="pickle of face tracks (shared across runs)")
    a = ap.parse_args()
    from custodian_video.surrogate_face import SDKeyframeSurrogate
    D = detect_tracks(a.input, a.cache, a.device, a.max_frames)
    tracks, n, fps, W, H = D["tracks"], D["n"], D["fps"], D["W"], D["H"]
    tracks = [t for t in tracks if t.label == "face"]
    T = {"pass1_face_det_s": D["det_s"]}
    t0 = time.time()
    be = SDKeyframeSurrogate(a.device, strength=a.sd_strength, motion=a.motion, lip_gain=a.lip_gain,
                             drive=a.drive, mask_grow=a.mask_grow,
                             guard_thr=a.guard_thr if a.guard_thr > 0 else None)
    be.prepare(a.input, tracks)
    be.pipe = None
    T["prepare_s"] = time.time() - t0
    per = {}
    for t in tracks:
        for f, v in t.dense.items():
            per.setdefault(f, []).append((t, v))
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen([ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
                          "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(out)],
                         stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(a.input)
    acts = {}
    t0 = time.time(); t_face = 0.0; n_face = 0
    for f in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        for t, (box, lm, src) in per.get(f, []):
            s = time.time()
            if be.apply(fr, t, box, lm):
                act = "surrogate_face"
            else:
                redact_box(fr, box, "blur", "ellipse"); act = "blur_fallback"
            t_face += time.time() - s; n_face += 1
            acts.setdefault(t.tid, set()).add(act)
        p.stdin.write(fr.tobytes())
    p.stdin.close(); p.wait(); cap.release()
    T["render_s"] = time.time() - t0
    T["face_apply_s"] = t_face
    T["face_frames"] = n_face
    T["face_apply_fps"] = n_face / max(t_face, 1e-6)
    T["render_fps_incl_decode_encode"] = n / T["render_s"]
    T["backend_stats"] = be.stats
    T["backend"] = be.name
    T["motion_effective"] = be.motion
    dets = [{"id": f"face-{t.tid}", "channel": "faces", "type": "face",
             "action": sorted(acts.get(t.tid, [])),
             "boxes": [[int(f)] + [round(float(x), 1) for x in v[0]] for f, v in sorted(t.dense.items())]}
            for t in tracks]
    audit = {"summary": {"faces": len(tracks), "motion": be.motion, "stats": be.stats},
             "detections": dets, "timing": T}
    out.with_suffix(".audit.json").write_text(json.dumps(audit))
    out.with_suffix(".timing.json").write_text(json.dumps(T, indent=1))
    print(json.dumps(T, indent=1))


if __name__ == "__main__":
    main()
