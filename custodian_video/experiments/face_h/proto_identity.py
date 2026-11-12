"""Prototype of the synthetic-identity generator (RealVisXL + Qwen3-VL attribute check
+ quality ranking) on the keyframe faces of the demo clips; uses the backend's own
_portraits / _verify_candidates.  Writes a sheet per track to out/proto_<gen>/.
Usage: proto_identity.py <generator> <n> clip:frame:x1,y1,x2,y2 ..."""
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np

from custodian_video.face_attributes import FaceAttributes
from custodian_video.surrogate_face_swap import ResearchSwapSurrogate

HERE = os.path.dirname(os.path.abspath(__file__))


def main(gen, n, specs):
    od = f"{HERE}/out/proto_{gen}"
    os.makedirs(od, exist_ok=True)
    crops = {}
    for sp in specs:
        clip, f, box = sp.split(":")
        x1, y1, x2, y2 = map(int, box.split(","))
        cap = cv2.VideoCapture(f"../data/video_demo/inputs/{clip}.mp4")
        cap.set(1, int(f)); fr = cap.read()[1]
        cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 0.95
        crops[f"{clip}_{f}"] = fr[int(max(0, cy - s)):int(cy + s), int(max(0, cx - s)):int(cx + s)].copy()
    attrs = FaceAttributes()(crops)
    S = ResearchSwapSurrogate("cuda", generator=gen)
    S.t2i, S._gen_dev = S._sd_txt2img()
    ports = {k: S._portraits(attrs[k], S._seed("proto", k) & 0x7FFFFFFF, n) for k in crops}
    todo = [SimpleNamespace(tid=k) for k in crops]
    ports = S._verify_candidates("proto", todo, attrs, ports)
    for k, L in ports.items():
        q = S.cand_quality.get(k, {})
        order = sorted(range(len(L)), key=lambda i: -q.get(i, 0))
        tiles = [cv2.resize(crops[k], (256, 256))]
        for i in order:
            t = cv2.resize(L[i][0], (256, 256))
            cv2.putText(t, f"cand {i} q{q.get(i, 0):.2f}", (4, 20), 0, 0.6, (0, 255, 0), 2)
            tiles.append(t)
        cv2.imwrite(f"{od}/{k}.jpg", np.hstack(tiles))
        print(k, "info", S.cand_info.get(k), "order", order)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
