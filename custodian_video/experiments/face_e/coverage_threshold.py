"""Coverage of YuNet reference faces vs EgoBlur face threshold (+ small-face padding).
  python coverage_threshold.py IN.mp4 --thr 0.1 0.05 0.03 --extra 0 0.5"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from custodian_video.detectors import make_face_detector  # noqa: E402
from custodian_video.tracking import densify, link  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("input")
ap.add_argument("--thr", nargs="+", type=float, default=[0.1, 0.05]); ap.add_argument("--extra", nargs="+", type=float, default=[0.0, 0.5])
a = ap.parse_args()
lo = min(a.thr)
det = make_face_detector("egoblur", "cuda", lo, "none")
cap = cv2.VideoCapture(a.input); fps = cap.get(5); W, H = int(cap.get(3)), int(cap.get(4))
yn = cv2.FaceDetectorYN.create(str(Path.home() / ".cache/custodian_video/face_detection_yunet_2023mar.onnx"), "", (W, H), 0.6, 0.3, 5000)
dets, refs, n = {}, [], 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    dets[n] = det(fr, n)
    if n % 2 == 0:
        _, fs = yn.detect(fr)
        refs += [(n, r) for r in (fs if fs is not None else []) if min(r[2], r[3]) >= 24]
    n += 1
for t in a.thr:
    faces = {f: [d for d in L if d.score >= t] for f, L in dets.items()}
    for ex in a.extra:
        tr = [densify(x, n, max_interp_gap=int(fps * 0.6), pad_before=3, pad_after=3, smooth_win=5, dilate=1.25,
                      frame_wh=(W, H), small_px=80.0 if ex > 0 else 0.0, small_extra=ex)
              for x in link(faces, iou_thr=0.2, max_gap=int(fps * 0.6))]
        bx = {}
        for x in tr:
            for f, v in x.dense.items():
                bx.setdefault(f, []).append(v[0])
        cov = np.array([any(b[0] <= r[0] + r[2] / 2 <= b[2] and b[1] <= r[1] + r[3] / 2 <= b[3] for b in bx.get(f, []))
                        for f, r in refs])
        sc = np.array([r[14] for _, r in refs])
        area = np.mean([sum((b[2] - b[0]) * (b[3] - b[1]) for b in bx.get(f, [])) / (W * H) for f in range(n)])
        print(f"thr {t:.2f} extra {ex}: tracks {len(tr)}  coverage all {cov.mean():.3f}  score>=0.7 {cov[sc >= 0.7].mean() if (sc>=0.7).any() else float('nan'):.3f}  "
              f"score>=0.8 {cov[sc >= 0.8].mean() if (sc>=0.8).any() else float('nan'):.3f}  masked area/frame {area:.3%}")
