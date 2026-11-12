"""Per-frame re-identification with THREE recognizers (face_e/identity2.py + glintr100).

For every frame, every YuNet face (>= --min-face px) in the ORIGINAL whose centre
lies in a face-track box is aligned with YuNet's 5 points, and the embedding of
the original is compared with the embedding of each output at the SAME
location (worst case: the attacker knows where the face is).

  * SFace (OpenCV zoo, Apache-2.0)  -- also the pipeline's own guard; threshold 0.363
  * ArcFace-R50 w600k (InsightFace buffalo_l) -- EVALUATION ONLY, non-commercial
    research licence, never used by the pipeline; threshold 0.217 = FAR 1e-3
    calibrated in custodian_video/eval (summary.md).

  python identity2.py --orig in.mp4 --outs v1=a.mp4 v2=b.mp4 --audit b.audit.json --json-out x.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.common import FaceModels  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from glint import Glint, threshold  # noqa: E402

T_SFACE, T_ARC = 0.363, 0.217


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True); ap.add_argument("--outs", nargs="+", required=True)
    ap.add_argument("--audit", required=True); ap.add_argument("--json-out")
    ap.add_argument("--min-face", type=int, default=24)
    ap.add_argument("--dump", help="write every (frame, cosine) per output and recogniser here")
    a = ap.parse_args()
    outs = dict(o.split("=", 1) for o in a.outs)
    boxes = {}
    for d in json.loads(Path(a.audit).read_text())["detections"]:
        if d["channel"] == "faces":
            for b in d["boxes"]:
                boxes.setdefault(b[0], []).append((d["id"], b[1:5]))
    fm = FaceModels(gpu=False); gl = Glint(); T_GL = threshold()
    caps = {"original": cv2.VideoCapture(a.orig), **{k: cv2.VideoCapture(v) for k, v in outs.items()}}
    sims = {k: {"sface": [], "arc": [], "glint": []} for k in outs}
    f = 0
    while True:
        frs = {k: c.read()[1] for k, c in caps.items()}
        if any(v is None for v in frs.values()):
            break
        rows = [r for r in fm.detect(frs["original"]) if min(r[2], r[3]) >= a.min_face and any(
            b[0] <= r[0] + r[2] / 2 <= b[2] and b[1] <= r[1] + r[3] / 2 <= b[3] for _, b in boxes.get(f, []))]
        if rows:
            rows = np.stack(rows)
            so, ao, go = fm.sface_embed(frs["original"], rows), fm.arc_embed(frs["original"], rows), gl.embed(frs["original"], rows)
            for k in outs:
                s, ar, gg = fm.sface_embed(frs[k], rows), fm.arc_embed(frs[k], rows), gl.embed(frs[k], rows)
                sims[k]["glint"] += [(f, float(x)) for x in (go * gg).sum(1)]
                sims[k]["sface"] += [(f, float(x)) for x in (so * s).sum(1)]
                sims[k]["arc"] += [(f, float(x)) for x in (ao * ar).sum(1)]
        f += 1
    if a.dump:
        Path(a.dump).write_text(json.dumps({k: {m: v for m, v in d.items()} for k, d in sims.items()}))
    res = {"frames": f}
    for k in outs:
        r = {}
        for m, T in (("sface", T_SFACE), ("arc", T_ARC), ("glint", T_GL)):
            s = np.array([x for _, x in sims[k][m]])
            if len(s) == 0:
                continue
            r[m] = dict(n=int(len(s)), reid_rate=round(float((s > T).mean()), 4), mean=round(float(s.mean()), 3),
                        p95=round(float(np.percentile(s, 95)), 3), max=round(float(s.max()), 3),
                        frames_over=[(fr, round(c, 3)) for fr, c in sims[k][m] if c > T][:10])
        res[k] = r
    print(json.dumps(res, indent=1))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
