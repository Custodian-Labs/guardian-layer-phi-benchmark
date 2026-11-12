"""Face-only transform run for experiment face_g (v2 vs research-only synthetic-identity face swap).

Same face path as pipeline.py (EgoBlur Gen1 @0.10 + YuNet landmarks -> link ->
densify -> SDKeyframeSurrogate.prepare -> per-frame apply / blur), with the
face tracks cached so the before/after variants see identical tracks.  Writes
<out>.mp4, <out>.audit.json (demo/evaluate.py format) and <out>.timing.json.

  python experiments/face_e/run_face_e.py ../data/video_demo/inputs/clip1_interview_synthpii.mp4 \
      -o experiments/face_f/out/clip1_v2.mp4 --cache experiments/face_f/out/clip1_tracks.pkl --realism v2
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from custodian_video.masking import redact_box  # noqa: E402
from custodian_video.tracking import densify, link  # noqa: E402


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def detect_tracks(inp, cache, device="cuda", small_extra=0.0):
    if cache and Path(cache).exists():
        res = pickle.loads(Path(cache).read_bytes())
        if small_extra > 0:   # re-densify the cached raw detections with small-face padding
            fps = res["fps"]
            res["tracks"] = [densify(t, res["n"], max_interp_gap=int(fps * 0.6), pad_before=3, pad_after=3,
                                     smooth_win=5, dilate=1.25, frame_wh=(res["W"], res["H"]),
                                     small_px=80.0, small_extra=small_extra) for t in res["tracks"]]
        return res
    from custodian_video.detectors import make_face_detector
    det = make_face_detector("egoblur", device, None, "yunet")
    cap = cv2.VideoCapture(inp)
    fps = cap.get(cv2.CAP_PROP_FPS); W, H = int(cap.get(3)), int(cap.get(4))
    faces, n, t0 = {}, 0, time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        faces[n] = det(fr, n); n += 1
    cap.release()
    ft = link(faces, iou_thr=0.2, max_gap=int(fps * 0.6))
    tracks = [densify(t, n, max_interp_gap=int(fps * 0.6), pad_before=3, pad_after=3,
                      smooth_win=5, dilate=1.25, frame_wh=(W, H),
                      small_px=80.0, small_extra=small_extra) for t in ft]
    res = dict(tracks=tracks, n=n, fps=fps, W=W, H=H, det_s=time.time() - t0)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_bytes(pickle.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--realism", default="v2", help="v1 (old surrogate) | v2 (realism pipeline)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-face", type=int, default=32)
    ap.add_argument("--cache", help="pickle of face tracks (shared across runs)")
    ap.add_argument("--kw", default="{}", help="extra SDKeyframeSurrogate kwargs as JSON")
    ap.add_argument("--backend", default="v2", help="v2 (current default backend) | v3 (surrogate_face_v3) | swap (surrogate_face_swap, research-only)")
    ap.add_argument("--small-extra", type=float, default=0.0,
                    help="small-face mask padding (pipeline default 0.5; 0 = before 2026-10-05)")
    a = ap.parse_args()
    from custodian_video.surrogate_face import SDKeyframeSurrogate
    if a.backend == "v3":
        from custodian_video.surrogate_face_v3 import SDKeyframeSurrogateV3 as SDKeyframeSurrogate
    if a.backend == "swap":
        from custodian_video.surrogate_face_swap import ResearchSwapSurrogate as SDKeyframeSurrogate
    D = detect_tracks(a.input, a.cache, a.device, a.small_extra)
    tracks, n, fps, W, H = D["tracks"], D["n"], D["fps"], D["W"], D["H"]
    tracks = [t for t in tracks if t.label == "face"]
    T = {"pass1_face_det_s": D["det_s"]}
    t0 = time.time()
    be = SDKeyframeSurrogate(a.device, min_face=a.min_face, realism=a.realism, **json.loads(a.kw))
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
                          "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(out)],
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
    T["v3_stats"] = getattr(be, "v3_stats", None)
    T["surrogate_tracks"] = len(be.src) or len(be.surr)
    T["gen_info"] = {str(k): v for k, v in getattr(be, "gen_info", {}).items()}
    T["frame_state"] = {str(k): {str(f): round(w, 3) for f, w in v.items()}
                        for k, v in getattr(be, "frame_state", {}).items()}
    dets = [{"id": f"face-{t.tid}", "channel": "faces", "type": "face",
             "action": sorted(acts.get(t.tid, [])),
             "boxes": [[int(f)] + [round(float(x), 1) for x in v[0]] for f, v in sorted(t.dense.items())]}
            for t in tracks]
    audit = {"summary": {"faces": len(tracks), "stats": be.stats}, "detections": dets, "timing": T}
    out.with_suffix(".audit.json").write_text(json.dumps(audit))
    out.with_suffix(".timing.json").write_text(json.dumps(T, indent=1))
    print(json.dumps(T, indent=1))


if __name__ == "__main__":
    main()
