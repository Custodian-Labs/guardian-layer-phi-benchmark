"""Face privacy report over the web-demo outputs (every clip x config).

Reference faces = YuNet (independent of the pipeline's EgoBlur) on the ORIGINAL,
every --step-th frame, short side >= --min-face.  Per config:
  coverage      share of reference faces whose centre lies in one of the
                config's masked face boxes (audit log) -- demo/evaluate.py's
                recall proxy; < 95 % is flagged
  sface_reid    share of reference faces whose SFace embedding at the SAME
                location in the output still matches the original (cos > 0.363)
  arc_reid      same with ArcFace-R50 (InsightFace buffalo_l, non-commercial,
                EVALUATION ONLY) at its FAR-1e-3 threshold 0.217
  *_masked      the same restricted to covered faces; split by face size
`box` (solid ellipse, no face pixels left) is the protocol's floor: whatever
it still "re-identifies" comes from context (hair, head outline, clothing) or
partial coverage, not from the face.  impostor_*: original face i vs the output at
ANOTHER reference face j of the same frame (a different person): the rate at
which the protocol "matches" the wrong person.

  python privacy_report.py --root ../data/video_demo --step 2 --json-out x.json --md-out x.md
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONFIGS = ["blur", "box", "surrogate_static", "surrogate_animated", "surrogate_recommended", "gen2"]
T_S, T_A = 0.363, 0.217


def clip_report(args):
    root, clip, cfgs, step, min_face = args
    from eval.common import FaceModels
    fm = FaceModels(gpu=False)
    inp = Path(root) / "inputs" / f"{clip}.mp4"
    outs = {c: Path(root) / "outputs" / clip / f"{c}.mp4" for c in cfgs}
    outs = {c: p for c, p in outs.items() if p.exists() and p.with_suffix(".audit.json").exists()}
    boxes, prim = {}, {}
    for c, p in outs.items():
        bb, pp = {}, {}
        for d in json.loads(p.with_suffix(".audit.json").read_text())["detections"]:
            if d.get("channel") == "faces":
                for b in d.get("boxes", []):
                    bb.setdefault(b[0], []).append(b[1:5])
                    if not d.get("recall_net_only"):
                        pp.setdefault(b[0], []).append(b[1:5])
        boxes[c], prim[c] = bb, pp
    caps = {"o": cv2.VideoCapture(str(inp)), **{c: cv2.VideoCapture(str(p)) for c, p in outs.items()}}
    acc = {c: [] for c in outs}       # (size, covered, sface, arc, yunet score, covered by EgoBlur tracks)
    imp = {c: [] for c in outs}
    f = 0
    while True:
        frs = {k: cap.read()[1] for k, cap in caps.items()}
        if any(v is None for v in frs.values()):
            break
        if f % step == 0:
            o = frs["o"]
            rows = [r for r in fm.detect(o) if min(r[2], r[3]) >= min_face]
            if rows:
                rows = np.stack(rows)
                so, ao = fm.sface_embed(o, rows), fm.arc_embed(o, rows)
                for c in outs:
                    xs, xa = fm.sface_embed(frs[c], rows), fm.arc_embed(frs[c], rows)
                    s = (so * xs).sum(1)
                    a = (ao * xa).sum(1)
                    if len(rows) > 1:   # impostor floor: original face i vs output at ANOTHER face j
                        js = np.roll(np.arange(len(rows)), 1)
                        imp[c] += list(zip((so * xs[js]).sum(1), (ao * xa[js]).sum(1)))
                    for r, si, ai in zip(rows, s, a):
                        cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
                        inside = lambda B: any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in B.get(f, []))
                        acc[c].append((min(r[2], r[3]), inside(boxes[c]), float(si), float(ai), float(r[14]),
                                       inside(prim[c])))
        f += 1
    res = {}
    for c, L in acc.items():
        if not L:
            res[c] = {"n_ref": 0}
            continue
        A = np.array(L, float)
        sz, cov, s, a, sc, cp = A[:, 0], A[:, 1] > 0, A[:, 2], A[:, 3], A[:, 4], A[:, 5] > 0
        conf = sc >= 0.8
        r = {"n_ref": int(len(A)), "coverage": round(float(cov.mean()), 4),
             "sface_reid": round(float((s > T_S).mean()), 4), "arc_reid": round(float((a > T_A).mean()), 4),
             "sface_reid_masked": round(float((s[cov] > T_S).mean()), 4) if cov.any() else None,
             "arc_reid_masked": round(float((a[cov] > T_A).mean()), 4) if cov.any() else None,
             "sface_max_masked": round(float(s[cov].max()), 3) if cov.any() else None,
             "n_ref_confident": int(conf.sum()),
             "coverage_confident": round(float(cov[conf].mean()), 4) if conf.any() else None,
             "coverage_confident_egoblur_only": round(float(cp[conf].mean()), 4) if conf.any() else None,
             "sface_reid_confident": round(float((s[conf] > T_S).mean()), 4) if conf.any() else None,
             "arc_reid_confident": round(float((a[conf] > T_A).mean()), 4) if conf.any() else None,
             "by_size": {}}
        for lo, hi in ((0, 48), (48, 96), (96, 10000)):
            m = (sz >= lo) & (sz < hi)
            if m.any():
                r["by_size"][f"{lo}-{hi}px"] = {"n": int(m.sum()), "coverage": round(float(cov[m].mean()), 3),
                                                "sface_reid": round(float((s[m] > T_S).mean()), 3),
                                                "arc_reid": round(float((a[m] > T_A).mean()), 3)}
        if imp[c]:
            I = np.array(imp[c], float)
            r["impostor_sface"] = round(float((I[:, 0] > T_S).mean()), 4)
            r["impostor_arc"] = round(float((I[:, 1] > T_A).mean()), 4)
            r["n_impostor_pairs"] = int(len(I))
        res[c] = r
    return clip, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--min-face", type=int, default=24); ap.add_argument("--configs", nargs="*", default=CONFIGS)
    ap.add_argument("--clips", nargs="*"); ap.add_argument("--json-out"); ap.add_argument("--md-out")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    clips = a.clips or sorted(p.stem for p in (Path(a.root) / "inputs").glob("*.mp4"))
    with ProcessPoolExecutor(a.workers) as ex:
        R = dict(ex.map(clip_report, [(a.root, c, a.configs, a.step, a.min_face) for c in clips]))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(R, indent=1))
    pc = lambda v: "n/a" if v is None else f"{v:.1%}"
    L = ["| clip | config | ref faces (all / conf>=0.8) | coverage all | coverage conf (EgoBlur alone) | SFace re-ID all / masked / conf | ArcFace re-ID all / masked / conf (eval) | SFace max masked | impostor floor SFace / ArcFace |",
         "|---|---|---|---|---|---|---|---|---|"]
    for clip in clips:
        for c in a.configs:
            r = R[clip].get(c)
            if not r or not r.get("n_ref"):
                continue
            flag = " **<95%**" if r["coverage"] < 0.95 else ""
            fc = " **<95%**" if (r["coverage_confident"] or 1) < 0.95 else ""
            L.append(f"| {clip} | {c} | {r['n_ref']} / {r['n_ref_confident']} | {pc(r['coverage'])}{flag} | "
                     f"{pc(r['coverage_confident'])}{fc} ({pc(r['coverage_confident_egoblur_only'])}) | "
                     f"{pc(r['sface_reid'])} / {pc(r['sface_reid_masked'])} / {pc(r['sface_reid_confident'])} | "
                     f"{pc(r['arc_reid'])} / {pc(r['arc_reid_masked'])} / {pc(r['arc_reid_confident'])} | "
                     f"{r['sface_max_masked']} | {pc(r.get('impostor_sface'))} / {pc(r.get('impostor_arc'))} |")
    md = "\n".join(L) + "\n"
    print(md)
    if a.md_out:
        Path(a.md_out).write_text(md)


if __name__ == "__main__":
    main()
