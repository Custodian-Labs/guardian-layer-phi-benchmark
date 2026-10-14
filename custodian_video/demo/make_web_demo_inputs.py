"""Build the extra input clips for the web video demo (web/video.html).

New clips (sources + licences in web/video/SOURCES.md; all Wikimedia Commons; clips 4-5 CC BY-SA 4.0, clip 6 CC BY 3.0):
  clip4_closeup_caption.mp4 : 12 s close-up of a speaker talking to camera. The real
      name caption of the source is fully covered by a SYNTHETIC lower-third with a
      fake name, member ID and e-mail. Original speech kept (no PII in it).
  clip5_presenter_badge.mp4 : 12 s medium shot of a presenter. The real name caption is
      cropped away and a SYNTHETIC visitor badge (fake name, ID, phone) is pinned to
      her shirt, following her face position. Original speech kept (no PII in it).
  clip6_meeting_adults.mp4  : 12 s hand-held shot of eight adults around a meeting table
      (faces from ~30 to ~80 px, profiles, occlusion, zoom-out). Original audio kept.
      Source: Wikimedia Commons, Biswarup Ganguly, CC BY 3.0.

Every PII value is invented; 555-01xx is a reserved fictional phone range and
example.org is a reserved domain.

Usage:
  python demo/make_web_demo_inputs.py --raw-dir <dir with downloaded .webm> --out <dir>
     raw files: cebu1.webm (Wikiclub_Cebu_Pagkat-on_sa_Sugbo_Interview.webm),
                cebu2.webm (Wiki_Pagkat-on_sa_Sugbo_sahring_experience.webm),
                kolkata.webm (Wikimedia_Meetup_-_St_Johns_Church_-_Kolkata_2016-09-10_9382.webm)
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
MODELS = Path.home() / ".cache" / "custodian_video"

CAPTION = {"name": "Dana Whitfield",
           "line2": "Member ID 30418-77  |  dana.whitfield@example.org"}
BADGE = [("VISITOR PASS", False), ("Priya Raman", True), ("ID V-558214", True),
         ("Tel (212) 555-0143", True)]


def _font(size: int, bold=True):
    names = (["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if bold
             else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
    for d in ["/usr/share/fonts/dejavu/", "/usr/share/fonts/truetype/dejavu/",
              "/usr/share/fonts/liberation/"]:
        for n in names:
            if Path(d + n).exists():
                return ImageFont.truetype(d + n, size)
    return ImageFont.load_default(size=size)


def ff(*args):
    subprocess.run([_FFMPEG, "-v", "error", "-y", *args], check=True)


def read_all(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    cap.release()
    return fr, fps


def write_mux(frames, fps, audio_src, dst):
    tmp = Path(str(dst) + ".tmp.mp4")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    ff("-i", str(tmp), "-i", str(audio_src), "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
       "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
       "-shortest", str(dst))
    tmp.unlink()


def lower_third(frame):
    """Opaque synthetic caption band covering the source's own caption (y 560-705)."""
    h, w = frame.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 556, w, 708], fill=(18, 32, 64, 255))
    d.rectangle([0, 556, 14, 708], fill=(16, 185, 129, 255))
    d.text((44, 572), CAPTION["name"], font=_font(46), fill=(255, 255, 255))
    d.text((46, 636), CAPTION["line2"], font=_font(30, bold=False), fill=(220, 230, 245))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def render_badge(w=250, h=170):
    img = Image.new("RGB", (w, h), (250, 250, 247))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w - 1, 38], fill=(190, 30, 45))
    d.text((12, 7), BADGE[0][0], font=_font(22), fill=(255, 255, 255))
    y = 48
    for (t, _), s in zip(BADGE[1:], (26, 22, 22)):
        d.text((12, y), t, font=_font(s), fill=(15, 15, 15))
        y += s + 13
    d.rectangle([0, 0, w - 1, h - 1], outline=(60, 60, 60), width=2)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def face_centres(frames):
    h, w = frames[0].shape[:2]
    det = cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2023mar.onnx"), "",
                                    (w, h), 0.6, 0.3, 50)
    cs = []
    for f in frames:
        _, faces = det.detect(f)
        if faces is not None and len(faces):
            x, y, bw, bh = max(faces, key=lambda r: r[2] * r[3])[:4]
            cs.append((x + bw / 2, y + bh / 2))
        else:
            cs.append(None)
    # fill gaps, then smooth (moving average over 9 frames)
    last = next(c for c in cs if c is not None)
    filled = []
    for c in cs:
        last = c if c is not None else last
        filled.append(last)
    a = np.array(filled, float)
    k = 9
    pad = np.pad(a, ((k // 2, k // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], np.ones(k) / k, mode="valid") for j in range(2)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw, out = Path(a.raw_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # clip4: close-up, synthetic lower third over the real caption
    t4 = out / "_c4.mp4"
    ff("-ss", "48.3", "-t", "12", "-i", str(raw / "cebu1.webm"), "-c:v", "libx264", "-crf", "16",
       "-preset", "veryfast", "-c:a", "aac", str(t4))
    fr, fps = read_all(t4)
    write_mux([lower_third(f) for f in fr], fps, t4, out / "clip4_closeup_caption.mp4")
    t4.unlink()

    # clip5: crop the caption away (16:9 window above it), scale to 720p, pin badge
    t5 = out / "_c5.mp4"
    ff("-ss", "34", "-t", "12", "-i", str(raw / "cebu2.webm"), "-vf",
       "crop=1048:590:116:0,scale=1280:720:flags=lanczos", "-c:v", "libx264", "-crf", "16",
       "-preset", "veryfast", "-c:a", "aac", str(t5))
    fr, fps = read_all(t5)
    cs = face_centres(fr)
    badge = render_badge()
    bh, bw = badge.shape[:2]
    res = []
    for f, (cx, cy) in zip(fr, cs):
        # badge sits over the shirt sticker: ~ +75 px right, +330 px below the face centre
        x0 = int(round(cx + 75 - bw / 2))
        y0 = int(round(cy + 330 - bh / 2))
        g = f.copy()
        g[y0:y0 + bh, x0:x0 + bw] = badge
        cv2.rectangle(g, (x0 + bw // 2 - 18, y0 - 10), (x0 + bw // 2 + 18, y0 + 4), (90, 90, 90), -1)
        res.append(g)
    write_mux(res, fps, t5, out / "clip5_presenter_badge.mp4")
    t5.unlink()

    # clip6: adults around a meeting table (adults only), trimmed and scaled to 720p
    ff("-ss", "66.0", "-t", "12.0", "-i", str(raw / "kolkata.webm"), "-vf",
       "scale=1280:720:flags=lanczos", "-c:v", "libx264", "-crf", "18",
       "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
       str(out / "clip6_meeting_adults.mp4"))

    gt = {
        "clip4_closeup_caption.mp4": {
            "onscreen_pii": [CAPTION["name"], "30418-77", "dana.whitfield@example.org"],
            "onscreen_non_pii": ["Member ID"], "spoken_pii": []},
        "clip5_presenter_badge.mp4": {
            "onscreen_pii": [t.split(" ", 1)[1] if t.startswith(("ID ", "Tel ")) else t
                             for t, p in BADGE if p],
            "onscreen_non_pii": [t for t, p in BADGE if not p], "spoken_pii": []},
    }
    (out / "ground_truth_web.json").write_text(json.dumps(gt, indent=2))
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
