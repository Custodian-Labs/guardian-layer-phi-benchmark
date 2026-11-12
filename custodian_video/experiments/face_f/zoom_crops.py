"""Side-by-side keyframes + zoomed region crops (original | variants ...).

For each time t: row 1 = head crop (1.9x the face box); then the three regions
the realism review complained about, each zoomed x(tile/region):
  forehead/hairline : top 45 % of the box, extended 30 % above it
  eyes/glasses      : 18-58 % of the box height
  edges             : left and right face outline (box edge -12 % .. +30 % of the box width)
  python zoom_crops.py --orig in.mp4 --outs v2=a.mp4 new=b.mp4 --audit a.audit.json --times 1 4 --dst x.jpg
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np


def label(z, s):
    cv2.rectangle(z, (0, 0), (z.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(z, s, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return z


def cut(fr, x1, y1, x2, y2, size):
    H, W = fr.shape[:2]
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))
    return cv2.resize(fr[y1:y2, x1:x2], size, interpolation=cv2.INTER_CUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True); ap.add_argument("--outs", nargs="+", required=True)
    ap.add_argument("--audit", required=True); ap.add_argument("--times", nargs="+", type=float)
    ap.add_argument("--dst", required=True); ap.add_argument("--tile", type=int, default=360)
    a = ap.parse_args()
    vids = {"original": a.orig, **dict(o.split("=", 1) for o in a.outs)}
    boxes = {}
    for d in json.loads(Path(a.audit).read_text())["detections"]:
        if d["channel"] == "faces":
            for b in d["boxes"]:
                boxes.setdefault(b[0], []).append(b[1:5])
    T = a.tile
    blocks = []
    for t in a.times:
        frs = {}
        for name, v in vids.items():
            c = cv2.VideoCapture(v); fps = c.get(5); f = int(round(t * fps))
            c.set(cv2.CAP_PROP_POS_FRAMES, f); frs[name] = c.read()[1]; c.release()
        b = max(boxes.get(f, [[0, 0, 10, 10]]), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2 = b; w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; s = max(w, h) * 0.95
        regs = [("head", (cx - s, cy - s, cx + s, cy + s), (T, T)),
                ("forehead/hair", (cx - 0.6 * w, y1 - 0.3 * h, cx + 0.6 * w, y1 + 0.45 * h), (T, int(T * 0.75 / 1.2))),
                ("eyes/glasses", (cx - 0.55 * w, y1 + 0.18 * h, cx + 0.55 * w, y1 + 0.58 * h), (T, int(T * 0.4 / 1.1))),
                ("L edge", (x1 - 0.12 * w, y1 + 0.05 * h, x1 + 0.3 * w, y2 - 0.1 * h), (T // 2, int(T // 2 * 0.85 / 0.42))),
                ("R edge", (x2 - 0.3 * w, y1 + 0.05 * h, x2 + 0.12 * w, y2 - 0.1 * h), (T // 2, int(T // 2 * 0.85 / 0.42)))]
        cols = []
        for name, fr in frs.items():
            ts = [label(cut(fr, *r, sz), f"{name} {rn} t={t:g}s") for rn, r, sz in regs]
            edge = np.hstack([ts[3], ts[4]])
            cols.append(np.vstack([ts[0], ts[1], ts[2], edge]))
        blk = np.hstack([np.pad(c, ((0, 0), (0, 4), (0, 0)), constant_values=255) for c in cols])
        blocks.append(np.pad(blk, ((0, 6), (0, 0), (0, 0)), constant_values=255))
    cv2.imwrite(a.dst, np.vstack(blocks), [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    main()
