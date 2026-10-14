"""Build the demo input clips from openly-licensed Wikimedia Commons footage.

Clips (sources + licences are recorded in demo/inputs/SOURCES.md):
  clip1_interview_synthpii.mp4 : 18 s of a CC BY 3.0 Royal Society interview
      (includes one shot cut / second person) with a SYNTHETIC "screen" panel
      carrying fake PII rendered onto it, and the audio track REPLACED by a
      local Piper-TTS voice reading a fake name / phone / account number.
  clip2_street_faces.mp4       : 9 s CC BY-SA 4.0 Tokyo street (many small faces)
  clip3_traffic_plates.mp4     : 12 s CC BY 3.0 Ljubljana street (vans + plates + people)

Usage:
  python demo/make_demo_inputs.py --raw-dir <dir with downloaded .webm> --tts <tts.wav>
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import imageio_ffmpeg
_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
HERE = Path(__file__).resolve().parent
OUT = HERE / "inputs"

# Fake PII -- all invented; 555-01xx is a reserved fictional phone range.
PANEL_LINES = [
    ("PATIENT INTAKE - Flood Clinic", False),
    ("Name: Margaret Holloway", True),
    ("Phone: (415) 555-0199", True),
    ("Acct #: 88427139", True),
    ("Topic: rainfall & river levels", False),
    ("Room 3 - follow-up next week", False),
]
SPOKEN_PII = ["Margaret Holloway", "four one five, five five five, zero one nine nine",
              "8 8 4 2 7 1 3 9"]


def _font(size: int):
    for p in ["/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def render_panel(w=600, h=330) -> np.ndarray:
    img = Image.new("RGB", (w, h), (245, 245, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w - 1, 52], fill=(30, 70, 140))
    f_title, f_body = _font(28), _font(30)
    d.text((14, 10), PANEL_LINES[0][0], font=f_title, fill=(255, 255, 255))
    y = 70
    for text, _ in PANEL_LINES[1:]:
        d.text((16, y), text, font=f_body, fill=(20, 20, 20))
        y += 50
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def overlay_panel(frame: np.ndarray, panel: np.ndarray, quad: np.ndarray) -> np.ndarray:
    h, w = panel.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(panel, M, (frame.shape[1], frame.shape[0]))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (frame.shape[1], frame.shape[0]))
    # monitor bezel
    cv2.polylines(frame, [quad.astype(np.int32)], True, (15, 15, 15), 14, cv2.LINE_AA)
    m = (mask > 0)[..., None]
    return np.where(m, warped, frame)


def ff(*args):
    subprocess.run([_FFMPEG, "-v", "error", "-y", *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--tts", required=True)
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    # clip1: trim, overlay panel, replace audio
    tmp = OUT / "_clip1_trim.mp4"
    ff("-ss", "51", "-t", "18.2", "-i", str(raw / "cloke.webm"), "-an", "-c:v", "libx264",
       "-preset", "veryfast", "-crf", "18", str(tmp))
    cap = cv2.VideoCapture(str(tmp))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))
    panel = render_panel()
    quad = np.array([[40, 610], [610, 630], [600, 930], [45, 960]])  # slight perspective
    tmp2 = OUT / "_clip1_panel.mp4"
    vw = cv2.VideoWriter(str(tmp2), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(overlay_panel(fr, panel, quad))
    cap.release(); vw.release()
    ff("-i", str(tmp2), "-i", a.tts, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset",
       "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
       str(OUT / "clip1_interview_synthpii.mp4"))
    tmp.unlink(); tmp2.unlink()

    ff("-i", str(raw / "dog.webm"), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
       "-pix_fmt", "yuv420p", "-c:a", "aac", str(OUT / "clip2_street_faces.mp4"))
    ff("-ss", "22", "-t", "12", "-i", str(raw / "traffic.webm"), "-c:v", "libx264", "-preset",
       "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac",
       str(OUT / "clip3_traffic_plates.mp4"))

    gt = {"clip1_interview_synthpii.mp4": {
        "onscreen_pii": [t.split(": ", 1)[1] for t, p in PANEL_LINES if p],
        "onscreen_non_pii": [t for t, p in PANEL_LINES if not p],
        "spoken_pii": SPOKEN_PII,
        "panel_quad": quad.tolist()}}
    (OUT / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
