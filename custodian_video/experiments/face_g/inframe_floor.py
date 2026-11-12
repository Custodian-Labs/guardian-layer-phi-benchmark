"""In-frame impostor floor for multi-person clips (clip6): every ORIGINAL face (YuNet,
>= --min-face, every --step frames) vs every OTHER person's face in the same output frame,
aligned at that other face's location.  Share over the threshold = how often a recogniser
'matches' a masked face to a DIFFERENT person in the same scene (context, not identity)."""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval.common import FaceModels  # noqa
from glint import Glint, threshold  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--orig"); ap.add_argument("--outs", nargs="+"); ap.add_argument("--step", type=int, default=2)
ap.add_argument("--min-face", type=int, default=24); ap.add_argument("--json-out"); a = ap.parse_args()
fm, gl = FaceModels(gpu=False), Glint(); T = {"sface": 0.363, "arc": 0.217, "glint": threshold()}
outs = dict(o.split("=", 1) for o in a.outs)
caps = {"o": cv2.VideoCapture(a.orig), **{k: cv2.VideoCapture(v) for k, v in outs.items()}}
S = {k: {m: [] for m in T} for k in outs}; same = {k: {m: [] for m in T} for k in outs}; f = 0
while True:
    frs = {k: c.read()[1] for k, c in caps.items()}
    if any(v is None for v in frs.values()): break
    if f % a.step == 0:
        rows = [r for r in fm.detect(frs["o"]) if min(r[2], r[3]) >= a.min_face]
        if len(rows) >= 2:
            rows = np.stack(rows)
            eo = {"sface": fm.sface_embed(frs["o"], rows), "arc": fm.arc_embed(frs["o"], rows), "glint": gl.embed(frs["o"], rows)}
            for k in outs:
                e = {"sface": fm.sface_embed(frs[k], rows), "arc": fm.arc_embed(frs[k], rows), "glint": gl.embed(frs[k], rows)}
                for m in T:
                    M = eo[m] @ e[m].T
                    S[k][m] += M[~np.eye(len(rows), dtype=bool)].tolist(); same[k][m] += np.diag(M).tolist()
    f += 1
res = {k: {m: {"same_reid": round(float((np.array(same[k][m]) > T[m]).mean()), 4), "inframe_floor": round(float((np.array(S[k][m]) > T[m]).mean()), 4),
               "n_same": len(same[k][m]), "n_pairs": len(S[k][m])} for m in T} for k in outs}
print(json.dumps(res, indent=1)); a.json_out and Path(a.json_out).write_text(json.dumps(res, indent=1))
