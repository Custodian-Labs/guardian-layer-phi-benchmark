"""Identity selection of the research-swap backend on chosen tracks of a dumped clip
(dump_fp_inputs.py pickle), without the rest of the pipeline.  Writes the synthetic
portraits (CUSTODIAN_RESEARCH_DUMP_DIR) and a strip of swapped frames.
Usage: try_identity.py <clip> <tid,...> <tag>"""
import os
import pickle
import sys

import cv2
import numpy as np

from custodian_video.surrogate_face_swap import ResearchSwapSurrogate

HERE = os.path.dirname(os.path.abspath(__file__))


def main(clip, tids, tag):
    D = pickle.load(open(f"{HERE}/out/{clip}.fpin.pkl", "rb"))
    want = {int(x) for x in tids.split(",")}
    tr = [t for t in D["ft"] if t.tid in want]
    S = ResearchSwapSurrogate("cuda", n_cand=int(os.environ.get("TRY_NCAND", "8")))
    D["inp"] = os.path.abspath(D["inp"])
    S.prepare(D["inp"], tr)
    print("gen_info", S.gen_info)
    cap = cv2.VideoCapture(D["inp"])
    tiles = []
    for t in tr:
        fs = sorted(S.frame_state.get(t.tid, {}))
        for f in [fs[int(i)] for i in np.linspace(0, len(fs) - 1, 6)] if fs else []:
            cap.set(1, f); fr = cap.read()[1]
            box, lm, _ = t.dense[f]
            out = fr.copy()
            ok = S.apply(out, t, box, lm)
            x1, y1, x2, y2 = [int(v) for v in box]
            w, h = x2 - x1, y2 - y1
            X1, Y1, X2, Y2 = max(0, x1 - w // 3), max(0, y1 - h // 3), x2 + w // 3, y2 + h // 3
            pair = np.hstack([cv2.resize(fr[Y1:Y2, X1:X2], (240, 300)), cv2.resize(out[Y1:Y2, X1:X2], (240, 300))])
            cv2.putText(pair, f"{f} {'swap' if ok else 'blur'} w{S.frame_state[t.tid].get(f, 0):.1f}", (4, 18), 0, 0.55, (0, 255, 255), 2)
            tiles.append(pair)
    if tiles:
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
        cv2.imwrite(f"{HERE}/out/try_{clip}_{tag}.jpg", np.vstack(rows))


if __name__ == "__main__":
    main(*sys.argv[1:4])
