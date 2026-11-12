"""Calibrate the glintr100 threshold at FAR 1e-3 on the VoxCeleb2 gallery clips (same
frames / detector as custodian_video/eval/metrics.py gallery_embeddings)."""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.common import FaceModels, CLIPS  # noqa
from glint import Glint, THR_FILE  # noqa
fm, g = FaceModels(gpu=False), Glint()
ids, E = [], []
for p in sorted((CLIPS / "vox2_gallery").glob("*.mp4")):
    c = cv2.VideoCapture(str(p)); n = int(c.get(7))
    for f in np.linspace(0, n - 1, 16).astype(int):
        c.set(1, f); ok, fr = c.read()
        if not ok:
            continue
        rows = fm.detect(fr, 0.7)
        if len(rows) == 0:
            continue
        r = rows[np.argmax(rows[:, 2] * rows[:, 3])][None]
        ids.append(p.stem.split("_")[1]); E.append(g.embed(fr, r)[0])
ids, E = np.array(ids), np.stack(E)
S = E @ E.T; same = ids[:, None] == ids[None]; iu = np.triu_indices(len(ids), 1)
imp, gen = S[iu][~same[iu]], S[iu][same[iu]]
thr = float(np.quantile(imp, 1 - 1e-3))
res = {"thr_far1e-3": round(thr, 4), "n_impostor_pairs": int(len(imp)), "n_genuine_pairs": int(len(gen)),
       "tar_at_thr": round(float((gen >= thr).mean()), 4), "n_ids": int(len(set(ids))), "n_frames": int(len(ids))}
THR_FILE.parent.mkdir(exist_ok=True, parents=True); THR_FILE.write_text(json.dumps(res, indent=1)); print(res)
