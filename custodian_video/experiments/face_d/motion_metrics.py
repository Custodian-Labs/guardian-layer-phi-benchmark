"""Expression / pose / temporal-stability metrics for face_d (independent of
LivePortrait: uses MediaPipe Face Landmarker, Apache-2.0, run in its own venv).

For every frame and every face track (boxes from the run's .audit.json) it runs
the landmarker on a crop of the ORIGINAL and of each OUTPUT video and reports,
per output, the Pearson correlation with the original over time of
  mouth_open  : inner-lip gap / mouth width (landmarks 13-14 / 78-308)
  jawOpen     : MediaPipe blendshape
  eyeBlink    : mean(eyeBlinkLeft, eyeBlinkRight) blendshape
  yaw, pitch  : from the facial transformation matrix
plus MAE of mouth_open, landmarker detection rate on the output, and a flicker
score = mean |I(t+1) - 2 I(t) + I(t-1)| inside the face box (grey, 0-255),
reported raw and as a ratio to the original video's value.

  mpvenv/bin/python motion_metrics.py --orig in.mp4 --outs static=a.mp4 reenact=b.mp4 \
      --audit b.audit.json --model face_landmarker.task --json-out m.json
"""
import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision


def landmarker(model):
    o = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=model),
                                     output_face_blendshapes=True,
                                     output_facial_transformation_matrixes=True, num_faces=1)
    return vision.FaceLandmarker.create_from_options(o)


def crop(fr, box, k=1.7):
    H, W = fr.shape[:2]
    x1, y1, x2, y2 = box
    c = np.array([(x1 + x2) / 2, (y1 + y2) / 2]); s = max(x2 - x1, y2 - y1) * k / 2
    a = np.clip(np.r_[c - s, c + s], 0, [W, H, W, H]).astype(int)
    return fr[a[1]:a[3], a[0]:a[2]], a


def measure(L, img):
    r = L.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not r.face_landmarks:
        return None
    p = np.array([[q.x, q.y] for q in r.face_landmarks[0]]) * [img.shape[1], img.shape[0]]
    bs = {b.category_name: b.score for b in r.face_blendshapes[0]}
    M = np.array(r.facial_transformation_matrixes[0])[:3, :3]
    yaw = np.degrees(np.arctan2(-M[2, 0], np.hypot(M[2, 1], M[2, 2])))
    pitch = np.degrees(np.arctan2(M[2, 1], M[2, 2]))
    return dict(mouth_open=np.linalg.norm(p[13] - p[14]) / (np.linalg.norm(p[78] - p[308]) + 1e-6),
                jawOpen=bs.get("jawOpen", 0), eyeBlink=(bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2,
                yaw=yaw, pitch=pitch)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True); ap.add_argument("--outs", nargs="+", required=True)
    ap.add_argument("--audit", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--json-out"); ap.add_argument("--series-out")
    a = ap.parse_args()
    outs = dict(o.split("=", 1) for o in a.outs)
    aud = json.loads(Path(a.audit).read_text())
    boxes = {}
    for d in aud["detections"]:
        if d["channel"] == "faces":
            for b in d["boxes"]:
                boxes.setdefault(b[0], []).append((d["id"], b[1:5]))
    vids = {"original": a.orig, **outs}
    caps = {k: cv2.VideoCapture(v) for k, v in vids.items()}
    L = {k: landmarker(a.model) for k in vids}
    ser = {k: {} for k in vids}          # k -> tid -> list of (f, metrics|None)
    flick = {k: [] for k in vids}
    hist = {k: {} for k in vids}         # tid -> last 2 grey crops (fixed geometry per triple)
    n = int(caps["original"].get(cv2.CAP_PROP_FRAME_COUNT))
    frames_buf = {k: [] for k in vids}
    for f in range(n):
        frs = {}
        for k, c in caps.items():
            ok, fr = c.read()
            frs[k] = fr if ok else None
        if any(v is None for v in frs.values()):
            break
        for k in vids:
            frames_buf[k] = (frames_buf[k] + [frs[k]])[-3:]
        for tid, box in boxes.get(f, []):
            for k in vids:
                im, ab = crop(frs[k], box)
                if im.size == 0:
                    continue
                ser[k].setdefault(tid, []).append((f, measure(L[k], im)))
                # flicker on the previous 3 frames with the CURRENT box geometry
                if len(frames_buf[k]) == 3 and all(any(t == tid for t, _ in boxes.get(g, []))
                                                   for g in (f - 1, f - 2)):
                    x1, y1, x2, y2 = ab
                    g = [cv2.cvtColor(F[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
                         for F in frames_buf[k]]
                    flick[k].append(float(np.abs(g[2] - 2 * g[1] + g[0]).mean()))
    res = {}
    keys = ("mouth_open", "jawOpen", "eyeBlink", "yaw", "pitch")
    for k in outs:
        r = {"per_track": {}}
        allx = {m: ([], []) for m in keys}
        det = tot = 0
        for tid, s in ser[k].items():
            o = dict(ser["original"][tid])
            pairs = [(o[f], m) for f, m in s if o.get(f) is not None]
            tot += len(pairs); det += sum(m is not None for _, m in pairs)
            pairs = [(x, y) for x, y in pairs if y is not None]
            tr = {"n": len(pairs)}
            for m in keys:
                xs, ys = [x[m] for x, _ in pairs], [y[m] for _, y in pairs]
                tr[f"corr_{m}"] = corr(xs, ys)
                allx[m][0].extend(xs); allx[m][1].extend(ys)
            tr["mae_mouth_open"] = round(float(np.mean([abs(x["mouth_open"] - y["mouth_open"]) for x, y in pairs])), 4) if pairs else None
            tr["std_mouth_open_orig"] = round(float(np.std([x["mouth_open"] for x, _ in pairs])), 4) if pairs else None
            tr["std_mouth_open_out"] = round(float(np.std([y["mouth_open"] for _, y in pairs])), 4) if pairs else None
            r["per_track"][tid] = tr
        for m in keys:
            r[f"corr_{m}"] = corr(*allx[m])
        r["landmarker_detect_rate_on_output"] = round(det / max(tot, 1), 3)
        r["flicker_2nd_diff"] = round(float(np.mean(flick[k])), 3)
        r["flicker_ratio_vs_original"] = round(float(np.mean(flick[k]) / np.mean(flick["original"])), 3)
        res[k] = r
    res["original"] = {"flicker_2nd_diff": round(float(np.mean(flick["original"])), 3)}
    print(json.dumps(res, indent=1))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=1))
    if a.series_out:
        Path(a.series_out).write_text(json.dumps({k: {t: [(f, m) for f, m in s] for t, s in v.items()}
                                                  for k, v in ser.items()}, default=float))


if __name__ == "__main__":
    main()
