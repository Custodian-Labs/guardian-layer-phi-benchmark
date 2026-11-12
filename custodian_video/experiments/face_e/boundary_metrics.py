"""Boundary-artifact metric for pasted surrogate faces (no mask needed).

Per frame and face-track box: the EDITED region C = pixels where the output
differs from the original by > --thr (max over BGR, 3x3 median-filtered), largest
component, holes filled (outer boundary), inside the box grown 1.6x.  Along C's boundary B we measure
  * dE_excess : local CIELAB colour step across the boundary in the OUTPUT
                (mean of the 5 px inner ring vs the 5 px outer ring, normalised
                convolution, sampled on B) minus the same step in the ORIGINAL
                at the same place (natural edges cancel), positive part only;
                lower = less visible seam
  * grad_ratio: mean |grad L| on B in the output / in the original (1 = seam
                indistinguishable from the original image structure)
  * seam_frac : fraction of B pixels whose output colour step exceeds the
                original's by > 6 dE (visible-seam pixels)
  * dE_flat / seam_flat : PRIMARY. Output colour step (and fraction > 6 dE) on the
                boundary pixels where the ORIGINAL has no edge (step < 3 dE:
                flat skin, flat background); there any step is a seam
Frames whose edited area is < 1% of the box are skipped (no paste).

  python boundary_metrics.py --orig in.mp4 --outs v1=a.mp4 v2=b.mp4 --audit b.audit.json --json-out x.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def ring_means(lab, inner, outer, s=3.0):
    gi = cv2.GaussianBlur(inner, (0, 0), s) + 1e-4
    go = cv2.GaussianBlur(outer, (0, 0), s) + 1e-4
    mi = cv2.GaussianBlur(lab * inner[..., None], (0, 0), s) / gi[..., None]
    mo = cv2.GaussianBlur(lab * outer[..., None], (0, 0), s) / go[..., None]
    return np.linalg.norm(mi - mo, axis=-1)


def measure(o, x, thr):
    d = np.abs(o.astype(np.int16) - x.astype(np.int16)).max(-1).astype(np.uint8)
    d = cv2.medianBlur(d, 3)
    C = (d > thr).astype(np.uint8)
    C = cv2.morphologyEx(C, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    C = cv2.morphologyEx(C, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(C)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    C = (lab == k).astype(np.uint8)
    # the edit's OUTER boundary: fill holes (colour-matched interior pixels can fall
    # under the threshold; they are not a boundary)
    C = cv2.morphologyEx(C, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cs, _ = cv2.findContours(C, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    C = np.zeros_like(C)
    cv2.drawContours(C, [max(cs, key=cv2.contourArea)], -1, 1, -1)
    if C.sum() < 0.01 * C.size:
        return None
    k5 = np.ones((11, 11), np.uint8)
    inner = (C - cv2.erode(C, k5)).astype(np.float32)
    outer = (cv2.dilate(C, k5) - C).astype(np.float32)
    B = (cv2.dilate(C, np.ones((3, 3), np.uint8)) - cv2.erode(C, np.ones((3, 3), np.uint8))) > 0
    B &= (outer + inner > 0)
    if B.sum() < 20:
        return None
    lo = cv2.cvtColor(o, cv2.COLOR_BGR2LAB).astype(np.float32)
    lx = cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32)
    eo, ex = ring_means(lo, inner, outer)[B], ring_means(lx, inner, outer)[B]
    g = lambda L: np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))
    go, gx = g(lo[..., 0])[B], g(lx[..., 0])[B]
    flat = eo < 3.0      # boundary pixels where the ORIGINAL has no edge: any step there is a seam
    if flat.sum() < 10:
        return None
    return dict(dE_flat=float(ex[flat].mean()), seam_flat=float((ex[flat] > 6).mean()),
                dE_excess=float(np.maximum(ex - eo, 0).mean()), grad_ratio=float(gx.mean() / (go.mean() + 1e-3)),
                seam_frac=float(((ex - eo) > 6).mean()), edited_frac=float(C.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True); ap.add_argument("--outs", nargs="+", required=True)
    ap.add_argument("--audit", required=True); ap.add_argument("--json-out")
    ap.add_argument("--thr", type=int, default=14); ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--min-box", type=int, default=48)
    a = ap.parse_args()
    outs = dict(o.split("=", 1) for o in a.outs)
    boxes = {}
    for d in json.loads(Path(a.audit).read_text())["detections"]:
        if d["channel"] == "faces":
            for b in d["boxes"]:
                boxes.setdefault(b[0], []).append(b[1:5])
    caps = {"original": cv2.VideoCapture(a.orig), **{k: cv2.VideoCapture(v) for k, v in outs.items()}}
    acc = {k: [] for k in outs}
    f = 0
    while True:
        frs = {k: c.read()[1] for k, c in caps.items()}
        if any(v is None for v in frs.values()):
            break
        if f % a.step == 0:
            H, W = frs["original"].shape[:2]
            for b in boxes.get(f, []):
                if min(b[2] - b[0], b[3] - b[1]) < a.min_box:
                    continue
                c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); s = max(b[2] - b[0], b[3] - b[1]) * 0.8
                x1, y1 = np.clip(c - s, 0, [W, H]).astype(int); x2, y2 = np.clip(c + s, 0, [W, H]).astype(int)
                o = frs["original"][y1:y2, x1:x2]
                for k in outs:
                    r = measure(o, frs[k][y1:y2, x1:x2], a.thr)
                    if r:
                        acc[k].append(r)
        f += 1
    res = {}
    for k, L in acc.items():
        if not L:
            res[k] = {"n": 0}
            continue
        res[k] = {"n": len(L), **{m: round(float(np.mean([r[m] for r in L])), 3)
                                  for m in ("dE_flat", "seam_flat", "dE_excess", "grad_ratio", "seam_frac", "edited_frac")},
                  "dE_excess_p90": round(float(np.percentile([r["dE_excess"] for r in L], 90)), 3)}
    print(json.dumps(res, indent=1))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
