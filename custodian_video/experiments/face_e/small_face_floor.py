"""Clip6 small-face privacy floor: re-ID (SFace / ArcFace-eval) of the ORIGINAL
faces against (a) existing outputs and (b) blur simulated on the original with
the track boxes scaled by s (redact_box, same style as the pipeline), to see how
much identity the region around the masked ellipse (hair, head outline) carries.

  python small_face_floor.py --orig in.mp4 --audit a.audit.json --outs v2=a.mp4 box=b.mp4 --scales 1.0 1.4 1.8
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.common import FaceModels  # noqa: E402
from custodian_video.masking import redact_box  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--orig", required=True); ap.add_argument("--audit", required=True)
ap.add_argument("--outs", nargs="*", default=[]); ap.add_argument("--scales", nargs="*", type=float, default=[])
ap.add_argument("--json-out")
a = ap.parse_args()
outs = dict(o.split("=", 1) for o in a.outs)
boxes = {}
for d in json.loads(Path(a.audit).read_text())["detections"]:
    if d["channel"] == "faces":
        for b in d["boxes"]:
            boxes.setdefault(b[0], []).append(b[1:5])
fm = FaceModels(gpu=False)
caps = {"o": cv2.VideoCapture(a.orig), **{k: cv2.VideoCapture(v) for k, v in outs.items()}}
keys = list(outs) + [f"blur_x{s}" for s in a.scales]
S = {k: {"sface": [], "arc": []} for k in keys}
f = 0
while True:
    fr = {k: c.read()[1] for k, c in caps.items()}
    if any(v is None for v in fr.values()):
        break
    o = fr["o"]
    rows = [r for r in fm.detect(o) if min(r[2], r[3]) >= 24 and any(
        b[0] <= r[0] + r[2] / 2 <= b[2] and b[1] <= r[1] + r[3] / 2 <= b[3] for b in boxes.get(f, []))]
    if rows:
        rows = np.stack(rows)
        imgs = {k: fr[k] for k in outs}
        for s in a.scales:
            x = o.copy()
            for b in boxes.get(f, []):
                c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); h = np.array([b[2] - b[0], b[3] - b[1]]) * s / 2
                redact_box(x, [*(c - h), *(c + h)], "blur", "ellipse")
            imgs[f"blur_x{s}"] = x
        so, ao = fm.sface_embed(o, rows), fm.arc_embed(o, rows)
        for k, x in imgs.items():
            S[k]["sface"] += list((so * fm.sface_embed(x, rows)).sum(1))
            S[k]["arc"] += list((ao * fm.arc_embed(x, rows)).sum(1))
    f += 1
res = {}
for k, v in S.items():
    s, r = np.array(v["sface"]), np.array(v["arc"])
    res[k] = dict(n=len(s), sface_reid=round(float((s > 0.363).mean()), 4), sface_max=round(float(s.max()), 3),
                  arc_reid=round(float((r > 0.217).mean()), 4), arc_max=round(float(r.max()), 3))
    print(k, res[k])
if a.json_out:
    Path(a.json_out).write_text(json.dumps(res, indent=1))
