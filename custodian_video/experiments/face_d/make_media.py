"""Before/after keyframes + side-by-side mp4 for face_d.

  python make_media.py --orig IN.mp4 --outs static=a.mp4 reenact=b.mp4 \
      --audit b.audit.json --times 2.0 5.0 16.5 --clip 1.0 7.0 14.9 18.1 --dst media/
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def zoom_box(boxes, f, W, H, k=2.2):
    bs = boxes.get(f)
    if not bs:
        return None
    b = max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); s = max(b[2] - b[0], b[3] - b[1]) * k / 2
    x1, y1 = np.clip(c - s, 0, [W, H]).astype(int); x2, y2 = np.clip(c + s, 0, [W, H]).astype(int)
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True); ap.add_argument("--outs", nargs="+", required=True)
    ap.add_argument("--audit", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--times", nargs="+", type=float, default=[2.0, 5.0, 16.5])
    ap.add_argument("--clip", nargs="+", type=float, default=[1.0, 7.0, 14.9, 18.1],
                    help="pairs of start end seconds for the side-by-side mp4")
    a = ap.parse_args()
    dst = Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
    vids = {"original": a.orig, **dict(o.split("=", 1) for o in a.outs)}
    boxes = {}
    for d in json.loads(Path(a.audit).read_text())["detections"]:
        if d["channel"] == "faces":
            for b in d["boxes"]:
                boxes.setdefault(b[0], []).append(b[1:5])
    caps = {k: cv2.VideoCapture(v) for k, v in vids.items()}
    fps = caps["original"].get(cv2.CAP_PROP_FPS)
    W, H = int(caps["original"].get(3)), int(caps["original"].get(4))
    n = int(caps["original"].get(cv2.CAP_PROP_FRAME_COUNT))
    want_kf = {int(round(t * fps)) for t in a.times}
    ranges = [(int(s * fps), min(n, int(e * fps))) for s, e in zip(a.clip[::2], a.clip[1::2])]
    import imageio_ffmpeg
    tile = 480
    sbs_w = tile * len(vids)
    p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-s", f"{sbs_w}x{tile}", "-r", f"{fps}", "-i", "-",
                          "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                          str(dst / "side_by_side_face_zoom.mp4")], stdin=subprocess.PIPE)
    last_zb = None
    for f in range(n):
        frs = {k: c.read()[1] for k, c in caps.items()}
        if any(v is None for v in frs.values()):
            break
        zb = zoom_box(boxes, f, W, H) or last_zb
        last_zb = zb
        if f in want_kf:
            t = f / fps
            full = np.hstack([label(cv2.resize(v, (640, 360)), k) for k, v in frs.items()])
            cv2.imwrite(str(dst / f"keyframe_full_t{t:05.1f}.jpg"), full, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if zb:
                x1, y1, x2, y2 = zb
                z = np.hstack([label(cv2.resize(v[y1:y2, x1:x2], (512, 512)), k) for k, v in frs.items()])
                cv2.imwrite(str(dst / f"keyframe_face_t{t:05.1f}.jpg"), z, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if zb and any(s <= f < e for s, e in ranges):
            x1, y1, x2, y2 = zb
            row = np.hstack([label(cv2.resize(v[y1:y2, x1:x2], (tile, tile)), k) for k, v in frs.items()])
            p.stdin.write(row.tobytes())
    p.stdin.close(); p.wait()
    # mux the original audio for the same ranges is skipped: ranges are concatenated
    print("wrote", sorted(x.name for x in dst.iterdir()))


if __name__ == "__main__":
    main()
