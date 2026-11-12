"""Media for experiment face_h:
  1. clip-1 ending strip (original | before | after) for surrogate_research and blur;
  2. identity contact sheet clips 1/4/5: original | previous page output | new page output | new synthetic portrait.
Writes into experiments/face_g/media/ (as requested)."""
import glob
import json
import os

import cv2
import numpy as np

R = os.environ.get("VIDEO_DEMO_DIR", "data/video_demo")
HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "..", "face_g", "media")


def frame(path, f):
    cap = cv2.VideoCapture(path); cap.set(1, f); ok, fr = cap.read()
    return fr


def label(im, t):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(im, t, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return im


def strip(clip, cfg, frames, out):
    rows = []
    for f in frames:
        o = frame(f"{R}/inputs/{clip}.mp4", f)
        b = frame(f"{R}/outputs_pre_face_h/{clip}/{cfg}.mp4", f)
        a = frame(f"{R}/outputs/{clip}/{cfg}.mp4", f)
        rows.append(np.hstack([label(cv2.resize(x, (640, 360)), f"{t} {cfg} f{f} ({f / 25:.1f}s)")
                               for x, t in ((o, "original"), (b, "before"), (a, "after"))]))
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def face_box(clip, tid_hint=None, frame_idx=None):
    a = json.load(open(f"{R}/outputs/{clip}/surrogate_research.audit.json"))
    for e in a["detections"]:
        if e.get("type") == "face" and e["max_score"] >= 0.9 and (tid_hint is None or e["id"] == tid_hint):
            d = {b[0]: b[1:5] for b in e["boxes"]}
            return d[frame_idx] if frame_idx in d else list(d.values())[len(d) // 2]


def sheet(items, out):
    rows = []
    for clip, tid, f in items:
        x1, y1, x2, y2 = face_box(clip, f"face-{tid}", f)
        w, h = x2 - x1, y2 - y1
        X1, Y1, X2, Y2 = max(0, x1 - w // 4), max(0, y1 - h // 4), x2 + w // 4, y2 + h // 4
        tiles = []
        for p, t in ((f"{R}/inputs/{clip}.mp4", "original"),
                     (f"{R}/outputs_pre_face_h/{clip}/surrogate_research.mp4", "previous (SD 1.5)"),
                     (f"{R}/outputs/{clip}/surrogate_research.mp4", "new (RealVisXL, checked)")):
            fr = frame(p, f)
            tiles.append(label(cv2.resize(fr[Y1:Y2, X1:X2], (300, 360)), t))
        por = sorted(glob.glob(f"{HERE}/out/portraits/{clip}/track{tid}_*_chosen_*.jpg"))
        p = cv2.imread(por[0]) if por else np.zeros((512, 512, 3), np.uint8)
        tiles.append(label(cv2.resize(p, (300, 360)), "new synthetic portrait"))
        rows.append(label(np.hstack(tiles), "") if False else np.hstack(tiles))
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    os.makedirs(MEDIA, exist_ok=True)
    c1 = "clip1_interview_synthpii"
    strip(c1, "surrogate_research", [380, 398, 420, 445], f"{MEDIA}/page_clip1_ending_original_before_after_surrogate_research.jpg")
    strip(c1, "blur", [380, 398, 420, 445], f"{MEDIA}/page_clip1_ending_original_before_after_blur.jpg")
    sheet([(c1, 0, 186), (c1, 5, 420), ("clip4_closeup_caption", 0, 180), ("clip5_presenter_badge", 0, 180)],
          f"{MEDIA}/identities_clip1_4_5_original_previous_new.jpg")
    print("ok")
