#!/usr/bin/env python3
"""Side-by-side download videos for the web demo: original (left) | surrogate face (right).

  python custodian_video/demo/make_compare_videos.py            # -> web/video/compare/<clip>_compare.mp4

Each half is scaled to 960 px wide, a label bar is drawn above each half, the audio is the
PROCESSED track (replaced names/numbers are heard as in the right-hand video).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "data" / "video_demo"
OUT = ROOT / "web" / "video" / "compare"
FF = imageio_ffmpeg.get_ffmpeg_exe()
FONTS = ["/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
         "/data/qbao775/miniconda3/envs/gemma4-rl/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/liberation-mono/LiberationMono-Bold.ttf"]
HALF_W, BAR_H = 960, 64
CFG = "surrogate_research"
LEFT = "Original"
RIGHT = "Surrogate face (research models, non-commercial)"


def font(size):
    for p in FONTS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def bar(path: Path):
    im = Image.new("RGB", (2 * HALF_W, BAR_H), (17, 24, 39))
    d = ImageDraw.Draw(im)
    f = font(28)
    for i, (txt, col) in enumerate([(LEFT, (255, 255, 255)), (RIGHT, (52, 211, 153))]):
        w = d.textlength(txt, font=f)
        d.text((i * HALF_W + (HALF_W - w) / 2, (BAR_H - 34) / 2), txt, font=f, fill=col)
    d.line([(HALF_W, 0), (HALF_W, BAR_H)], fill=(75, 85, 99), width=2)
    im.save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clips = [c["id"] for c in json.loads((ROOT / "web" / "video" / "manifest.json").read_text())["clips"]]
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "bar.png"
        bar(b)
        for cid in clips:
            orig, proc = DEMO / "inputs" / f"{cid}.mp4", DEMO / "outputs" / cid / f"{CFG}.mp4"
            if not proc.exists():
                print("skip", cid); continue
            dst = OUT / f"{cid}_compare.mp4"
            fc = (f"[0:v]scale={HALF_W}:-2,setsar=1[l];[1:v]scale={HALF_W}:-2,setsar=1[r];"
                  f"[l][r]hstack=inputs=2[v];[2:v][v]vstack=inputs=2[o]")
            subprocess.run([FF, "-v", "error", "-y", "-i", str(orig), "-i", str(proc), "-loop", "1", "-i", str(b),
                            "-filter_complex", fc, "-map", "[o]", "-map", "1:a?", "-shortest",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dst)], check=True)
            print(f"{dst.relative_to(ROOT)} {dst.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
