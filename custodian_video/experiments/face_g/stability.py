"""Swap/blur stability and landmark jitter of research-swap outputs, per face track.

State per frame: from <out>.timing.json "frame_state" (harness runs: weight 1 = swap,
0 < w < 1 = cross-fade, 0 = blur), or, with --blur-ref, by comparing the output with
the Blur output inside the track box (mean abs diff > 4 = face changed = swapped/fading).
  transitions   number of blur <-> non-blur switches inside a track
  swap_share    frames with a (fully or partly) swapped face / track frames
  longest_blur  longest run of blurred frames; blur_runs / short_swap_runs (< 12 frames)
Jitter (frames where the face is swapped on 3 consecutive frames): YuNet 5 points on
the output inside the box; mean |2nd temporal difference| / inter-ocular distance,
also on the original at the same frames; ratio = output / original (1 = as steady
as the original face).
  python stability.py --orig in.mp4 --audit a.audit.json --outs B0=x.mp4 S1=y.mp4 [--blur-ref blur.mp4] --json-out j
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np

YN = str(Path.home() / ".cache/custodian_video/face_detection_yunet_2023mar.onnx")


def yunet_pts(img, box, det_cache={}):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box; bw, bh = x2 - x1, y2 - y1
    X1, Y1, X2, Y2 = int(max(0, x1 - .4 * bw)), int(max(0, y1 - .4 * bh)), int(min(W, x2 + .4 * bw)), int(min(H, y2 + .4 * bh))
    c = img[Y1:Y2, X1:X2]
    if c.size == 0:
        return None
    k = max(1.0, 160 / max(c.shape[:2])); c = cv2.resize(c, None, fx=k, fy=k) if k > 1 else c
    d = det_cache.setdefault("d", cv2.FaceDetectorYN.create(YN, "", (c.shape[1], c.shape[0]), 0.5, 0.3, 50))
    d.setInputSize((c.shape[1], c.shape[0])); _, f = d.detect(c)
    if f is None or not len(f):
        return None
    f = max(f, key=lambda r: r[14])
    return f[4:14].reshape(5, 2) / k + [X1, Y1]


def runs(states):
    out, cur, n = [], None, 0
    for s in states:
        if s == cur:
            n += 1
        else:
            if cur is not None:
                out.append((cur, n))
            cur, n = s, 1
    if cur is not None:
        out.append((cur, n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True); ap.add_argument("--audit", required=True)
    ap.add_argument("--outs", nargs="+", required=True); ap.add_argument("--blur-ref")
    ap.add_argument("--min-face", type=int, default=80); ap.add_argument("--json-out")
    a = ap.parse_args()
    outs = dict(o.split("=", 1) for o in a.outs)
    tracks = {}
    for d in json.loads(Path(a.audit).read_text())["detections"]:
        if d["channel"] == "faces":
            bx = {b[0]: b[1:5] for b in d["boxes"]}
            if bx and max(min(b[2] - b[0], b[3] - b[1]) for b in bx.values()) >= a.min_face:
                tracks[d["id"]] = bx
    caps = {"o": cv2.VideoCapture(a.orig), **{k: cv2.VideoCapture(v) for k, v in outs.items()}}
    if a.blur_ref:
        caps["_blur"] = cv2.VideoCapture(a.blur_ref)
    st = {}
    for k, v in outs.items():
        tj = Path(v).with_suffix(".timing.json")
        if not a.blur_ref and tj.exists():
            fs = json.loads(tj.read_text()).get("frame_state", {})
            st[k] = {f"face-{tid}": {int(f): w for f, w in m.items()} for tid, m in fs.items()}
        else:
            st[k] = {tid: {} for tid in tracks}
    P = {k: {tid: {} for tid in tracks} for k in ["o", *outs]}
    f = 0
    while True:
        frs = {k: c.read()[1] for k, c in caps.items()}
        if any(x is None for x in frs.values()):
            break
        for tid, bx in tracks.items():
            if f not in bx:
                continue
            b = bx[f]
            x1, y1, x2, y2 = [int(max(0, v)) for v in b]
            for k in ["o", *outs]:
                P[k][tid][f] = yunet_pts(frs[k], b)
            if a.blur_ref:
                for k in outs:
                    dd = np.abs(frs[k][y1:y2, x1:x2].astype(int) - frs["_blur"][y1:y2, x1:x2].astype(int)).mean()
                    st[k][tid][f] = 1.0 if dd > 4 else 0.0
        f += 1
    res = {}
    for k in outs:
        R = {}
        for tid, bx in tracks.items():
            S = st[k].get(tid) or {}
            fs = sorted(bx)
            if not S:
                continue
            states = ["swap" if S.get(g, 0) > 0 else "blur" for g in fs]
            rr = runs(states)
            jit_o, jit_s = [], []
            for g in fs:
                trio = [g - 1, g, g + 1]
                if all(S.get(h, 0) >= 1.0 for h in trio) and all(P[k][tid].get(h) is not None and P["o"][tid].get(h) is not None for h in trio):
                    for src, L in (("o", jit_o), (k, jit_s)):
                        p = [P[src][tid][h] for h in trio]
                        iod = np.linalg.norm(p[1][0] - p[1][1]) + 1e-6
                        L.append(np.linalg.norm(p[0] - 2 * p[1] + p[2], axis=1).mean() / iod)
            R[tid] = {"frames": len(fs), "swap_share": round(sum(s == "swap" for s in states) / len(fs), 3),
                      "fade_frames": sum(0 < S.get(g, 0) < 1 for g in fs),
                      "transitions": max(0, len(rr) - 1),
                      "blur_runs": sum(1 for s, n in rr if s == "blur"),
                      "longest_blur": max([n for s, n in rr if s == "blur"] or [0]),
                      "short_swap_runs_lt12": sum(1 for i, (s, n) in enumerate(rr) if s == "swap" and n < 12 and len(rr) > 1),
                      "jitter_out": round(float(np.mean(jit_s)), 4) if jit_s else None,
                      "jitter_orig": round(float(np.mean(jit_o)), 4) if jit_o else None,
                      "jitter_ratio": round(float(np.mean(jit_s) / np.mean(jit_o)), 3) if jit_s else None,
                      "jitter_n": len(jit_s)}
        res[k] = R
    print(json.dumps(res, indent=1))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
