"""Zoomed face crops (original | variants...) at given times -> one jpg.
  python crops.py --orig in.mp4 --outs v1=a.mp4 v2=b.mp4 --audit b.audit.json --times 6 10 --dst x.jpg"""
import argparse, json
from pathlib import Path
import cv2, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True); ap.add_argument("--outs", nargs="+", required=True)
    ap.add_argument("--audit", required=True); ap.add_argument("--times", nargs="+", type=float)
    ap.add_argument("--dst", required=True); ap.add_argument("--tile", type=int, default=420)
    ap.add_argument("--k", type=float, default=1.9)
    a = ap.parse_args()
    vids = {"original": a.orig, **dict(o.split("=", 1) for o in a.outs)}
    boxes = {}
    for d in json.loads(Path(a.audit).read_text())["detections"]:
        if d["channel"] == "faces":
            for b in d["boxes"]:
                boxes.setdefault(b[0], []).append(b[1:5])
    rows = []
    for t in a.times:
        tiles = []
        for name, v in vids.items():
            c = cv2.VideoCapture(v); fps = c.get(5); f = int(round(t * fps))
            c.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = c.read()
            bs = boxes.get(f) or [[0, 0, fr.shape[1], fr.shape[0]]]
            b = max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; s = max(b[2] - b[0], b[3] - b[1]) * a.k / 2
            H, W = fr.shape[:2]
            x1, y1, x2, y2 = int(max(0, cx - s)), int(max(0, cy - s)), int(min(W, cx + s)), int(min(H, cy + s))
            z = cv2.resize(fr[y1:y2, x1:x2], (a.tile, a.tile), interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(z, (0, 0), (a.tile, 26), (0, 0, 0), -1)
            cv2.putText(z, f"{name} t={t:.1f}s", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(z)
        rows.append(np.hstack(tiles))
    cv2.imwrite(a.dst, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
