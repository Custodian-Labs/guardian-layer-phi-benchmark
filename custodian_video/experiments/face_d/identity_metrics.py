"""Dense (every-frame) SFace identity check for face_d.

For every frame and every YuNet face in the ORIGINAL that lies inside a masked
face-track box, align with YuNet's 5 points and compare SFace embeddings of the
original and the output at the same location (attacker knows where the face is;
same protocol as demo/evaluate.py but on all frames instead of every 5th).
Also reports identity consistency (mean cos of each frame's embedding to the
track's mean embedding) for the original and every output, and cross-identity:
whether the two surrogates of different people look alike.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MODELS = Path.home() / ".cache" / "custodian_video"
T = 0.363


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True); ap.add_argument("--outs", nargs="+", required=True)
    ap.add_argument("--audit", required=True); ap.add_argument("--json-out")
    ap.add_argument("--min-face", type=int, default=24)
    a = ap.parse_args()
    outs = dict(o.split("=", 1) for o in a.outs)
    aud = json.loads(Path(a.audit).read_text())
    boxes = {}
    for d in aud["detections"]:
        if d["channel"] == "faces":
            for b in d["boxes"]:
                boxes.setdefault(b[0], []).append((d["id"], b[1:5]))
    caps = {"original": cv2.VideoCapture(a.orig), **{k: cv2.VideoCapture(v) for k, v in outs.items()}}
    W, H = int(caps["original"].get(3)), int(caps["original"].get(4))
    det = cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2023mar.onnx"), "", (W, H), 0.6, 0.3, 5000)
    rec = cv2.FaceRecognizerSF.create(str(MODELS / "face_recognition_sface_2021dec.onnx"), "")
    sims = {k: [] for k in outs}
    emb = {k: {} for k in caps}
    f = 0
    while True:
        frs = {k: c.read()[1] for k, c in caps.items()}
        if any(v is None for v in frs.values()):
            break
        _, faces = det.detect(frs["original"])
        for fc in (faces if faces is not None else []):
            if min(fc[2], fc[3]) < a.min_face:
                continue
            cx, cy = fc[0] + fc[2] / 2, fc[1] + fc[3] / 2
            tid = next((t for t, b in boxes.get(f, []) if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]), None)
            if tid is None:
                continue
            e = {k: rec.feature(rec.alignCrop(fr, fc)) for k, fr in frs.items()}
            for k in caps:
                emb[k].setdefault(tid, []).append(e[k][0] / np.linalg.norm(e[k][0]))
            for k in outs:
                sims[k].append((f, tid, float(rec.match(e["original"], e[k], cv2.FaceRecognizerSF_FR_COSINE))))
        f += 1
    res = {"frames": f, "pairs": len(next(iter(sims.values())))}
    for k in caps:
        cons, means = [], {}
        for tid, E in emb[k].items():
            E = np.stack(E); m = E.mean(0); m /= np.linalg.norm(m); means[tid] = m
            cons.append(float((E @ m).mean()))
        r = {"identity_consistency_to_track_mean": round(float(np.mean(cons)), 3)}
        ids = sorted(means)
        if len(ids) > 1:
            r["cross_track_cos_of_track_means"] = round(float(means[ids[0]] @ means[ids[1]]), 3)
        if k in outs:
            s = np.array([x[2] for x in sims[k]])
            r.update(reid_rate=round(float((s > T).mean()), 4), n=int(len(s)), mean_cos=round(float(s.mean()), 3),
                     p95_cos=round(float(np.percentile(s, 95)), 3), max_cos=round(float(s.max()), 3),
                     reid_frames=[(fr, t, round(c, 3)) for fr, t, c in sims[k] if c > T][:20])
        res[k] = r
    print(json.dumps(res, indent=1))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
