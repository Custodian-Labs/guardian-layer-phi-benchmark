"""Old vs new person-context filter on dumped pass-1 data (dump_fp_inputs.py).
Writes out/<clip>.fpcmp.json and crops of every track whose decision changed.
Usage: eval_fp_filter.py <clip> [...]"""
import json
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import person_filter_old as OLD  # noqa: E402
from custodian_video import pipeline as P  # noqa: E402
from custodian_video import person_filter as NEW  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def old_filter(inp, ft, persons, det, p, wh):
    by, cands = {}, []
    for t in ft:
        d = {f: v[0] for f, v in t.dense.items() if v[2] == "det"}
        rec = any(x.meta.get("src") == "yunet_recall" for x in t.dets.values())
        ok, why = OLD.prescreen(d, persons, wh, max(x.score for x in t.dets.values()), rec, p)
        by[t.tid] = why
        if ok:
            cands.append((t, d))
    need = {f for _, d in cands for f in OLD.look_frames(d, p)}
    frames = read(inp, need)
    un = set()
    for t, d in cands:
        u, why = OLD.second_look(d, frames, det, p)
        by[t.tid] = why
        if u:
            un.add(t.tid)
    return un, by


def read(inp, need):
    frames = {}
    if need:
        cap = cv2.VideoCapture(inp)
        f = 0
        while f <= max(need):
            ok, fr = cap.read()
            if not ok:
                break
            if f in need:
                frames[f] = fr
            f += 1
    return frames


def main(clips):
    det = NEW.PersonDetector("cuda", NEW.FilterParams().person_threshold)
    for c in clips:
        D = pickle.load(open(f"{HERE}/out/{c}.fpin.pkl", "rb"))
        inp, ft, persons, wh = D["inp"], D["ft"], D["persons"], D["frame_wh"]
        po = OLD.FilterParams()
        un_old, why_old = old_filter(inp, ft, persons, det, po, wh)
        kept, unm, info = P._face_fp_filter(inp, ft, persons, det, NEW.FilterParams(), wh, print)
        un_new = {t.tid for t in unm}
        rows = []
        for t in ft:
            fs = sorted(t.dets)
            rows.append(dict(tid=t.tid, f0=fs[0], f1=fs[-1], n=len(fs), max=round(max(x.score for x in t.dets.values()), 3),
                             old=("unmasked" if t.tid in un_old else "masked"), old_why=why_old[t.tid],
                             new=("unmasked" if t.tid in un_new else "masked"), new_why=info["by_tid"][t.tid]))
        chg = [r for r in rows if r["old"] != r["new"]]
        json.dump({"rows": rows, "changed": chg, "n_tracks": len(ft), "old_unmasked": len(un_old),
                   "new_unmasked": len(un_new)}, open(f"{HERE}/out/{c}.fpcmp.json", "w"), indent=1)
        print(c, "tracks", len(ft), "unmasked old", len(un_old), "new", len(un_new), "changed", len(chg))
        for r in chg:
            print("  ", r)
        # crops of changed tracks (from the original, at up to 6 detected frames)
        tr = {t.tid: t for t in ft}
        for r in chg:
            t = tr[r["tid"]]
            fs = sorted(t.dets)
            sel = [fs[int(i)] for i in np.linspace(0, len(fs) - 1, min(6, len(fs)))]
            frames = read(inp, set(sel))
            tiles = []
            for f in sel:
                x1, y1, x2, y2 = [int(v) for v in t.dets[f].box[:4]]
                w, h = x2 - x1, y2 - y1
                fr = frames[f]
                X1, Y1, X2, Y2 = max(0, x1 - w), max(0, y1 - h), min(fr.shape[1], x2 + w), min(fr.shape[0], y2 + h)
                cr = fr[Y1:Y2, X1:X2].copy()
                cv2.rectangle(cr, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), (0, 0, 255), 2)
                k = 240 / max(cr.shape[:2])
                cr = cv2.resize(cr, None, fx=k, fy=k)
                cr = cv2.copyMakeBorder(cr, 0, 240 - cr.shape[0], 0, 240 - cr.shape[1], 0)
                cv2.putText(cr, str(f), (4, 18), 0, 0.6, (0, 255, 255), 2)
                tiles.append(cr)
            os.makedirs(f"{HERE}/out/changed", exist_ok=True)
            cv2.imwrite(f"{HERE}/out/changed/{c}_face{t.tid}.jpg", np.hstack(tiles))


if __name__ == "__main__":
    main(sys.argv[1:])
