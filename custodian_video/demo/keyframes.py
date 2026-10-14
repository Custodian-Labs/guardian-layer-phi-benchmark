"""Extract before/after keyframes: one PNG per timestamp with
[original | redact | transform] side by side (whichever outputs exist)."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def grab(path, t):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--outs", nargs="+", required=True, help="label=path")
    ap.add_argument("--times", nargs="+", type=float, required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--crop", help="x1,y1,x2,y2 crop (orig pixel coords) for zoomed panels")
    a = ap.parse_args()
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    cols = [("original", a.orig)] + [tuple(o.split("=", 1)) for o in a.outs]
    for t in a.times:
        tiles = []
        for lab, p in cols:
            fr = grab(p, t)
            if fr is None:
                continue
            if a.crop:
                x1, y1, x2, y2 = map(int, a.crop.split(","))
                fr = fr[y1:y2, x1:x2]
            s = a.width / fr.shape[1]
            fr = cv2.resize(fr, (a.width, int(fr.shape[0] * s)), interpolation=cv2.INTER_AREA)
            cv2.rectangle(fr, (0, 0), (a.width, 34), (0, 0, 0), -1)
            cv2.putText(fr, f"{lab}  t={t:.1f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(fr)
        out = f"{a.dst}_t{t:05.1f}.png"
        cv2.imwrite(out, np.hstack(tiles))
        print(out)


if __name__ == "__main__":
    main()
