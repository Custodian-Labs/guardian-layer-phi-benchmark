"""Cross-clip impostor floor for the eval-only ArcFace / SFace re-ID numbers.

Clips 4 and 5 hold one person each, so the in-frame impostor floor of
face_e/privacy_report.py is undefined there.  Instead: every original face of
clip A (YuNet, every --step frames, aligned at the original's location) is compared
with every face of variant V of clip B != A (different people).  floor = share of
such pairs over the re-ID threshold; this is how often the recognizer "matches"
a surrogate to a DIFFERENT real person, i.e. the rate a reported re-ID must exceed
before it means anything.
  python impostor_floor.py --clips clip1=orig.mp4,v2.mp4,v3.mp4 ... --names v2 v3 --json-out x.json
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.common import FaceModels  # noqa: E402

T_SFACE, T_ARC = 0.363, 0.217


def embs(fm, orig, outs, step, min_face=64):
    caps = [cv2.VideoCapture(p) for p in [orig] + outs]
    E = [{"sface": [], "arc": []} for _ in caps]
    f = 0
    while True:
        frs = [c.read()[1] for c in caps]
        if any(x is None for x in frs):
            break
        if f % step == 0:
            rows = [r for r in fm.detect(frs[0]) if min(r[2], r[3]) >= min_face]
            if rows:
                rows = np.stack(rows)
                for i, fr in enumerate(frs):
                    E[i]["sface"].append(fm.sface_embed(fr, rows)); E[i]["arc"].append(fm.arc_embed(fr, rows))
        f += 1
    return [{k: np.concatenate(v) if v else np.zeros((0, 1)) for k, v in e.items()} for e in E]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True); ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--step", type=int, default=5); ap.add_argument("--json-out")
    a = ap.parse_args()
    fm = FaceModels(gpu=False)
    D = {}
    for c in a.clips:
        k, v = c.split("=", 1); ps = v.split(",")
        D[k] = embs(fm, ps[0], ps[1:], a.step)
    res = {}
    for j, name in enumerate(a.names):
        r = {}
        for m, T in (("sface", T_SFACE), ("arc", T_ARC)):
            s = [(D[x][0][m] @ D[y][j + 1][m].T).ravel() for x in D for y in D if x != y]
            s = np.concatenate(s)
            r[m] = dict(pairs=int(len(s)), floor=round(float((s > T).mean()), 4), p99=round(float(np.percentile(s, 99)), 3),
                        max=round(float(s.max()), 3))
        res[name] = r
    print(json.dumps(res, indent=1))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
