"""Per-item "planted PII caught" checklist for the web video demo.

For each clip with planted synthetic PII (ground truth) and each processed output:
  on-screen item caught  = the GT string is NOT OCR-readable in any sampled frame
  spoken item caught     = none of its GT tokens appear in a Whisper re-transcript
The original is checked too (every item should be readable / audible there).
Writes only item labels and booleans (no PII values) to --json-out.

  CUDA_VISIBLE_DEVICES=3 python demo/web_pii_checklist.py --root ../data/video_demo \
      --json-out ../data/video_demo/checklist.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import fuzzy_in, norm, ocr_texts  # noqa: E402

HERE = Path(__file__).resolve().parent

# item label (shown on the page) -> how to check it.  Values are synthetic.
ITEMS = {
    "clip1_interview_synthpii": [
        ("screen", "Fake patient name on screen", "Margaret Holloway"),
        ("screen", "Fake phone number on screen", "(415) 555-0199"),
        ("screen", "Fake account number on screen", "88427139"),
        ("speech", "Fake name in speech", ["margaret", "holloway"]),
        ("speech", "Fake phone number in speech", ["5550199", "0199"]),
        # the web demo uses the first 12 s only (2026-10-09); the spoken account number starts at 12.5 s
    ],
    "clip4_closeup_caption": [
        ("screen", "Fake name in caption", "Dana Whitfield"),
        ("screen", "Fake member ID in caption", "30418-77"),
        ("screen", "Fake e-mail in caption", "dana.whitfield@example.org"),
    ],
    "clip5_presenter_badge": [
        ("screen", "Fake name on badge", "Priya Raman"),
        ("screen", "Fake badge ID", "V-558214"),
        ("screen", "Fake phone on badge", "(212) 555-0143"),
    ],
}
BENIGN = {  # non-PII lines that should stay readable
    "clip1_interview_synthpii": ["PATIENT INTAKE - Flood Clinic", "Topic: rainfall & river levels",
                                 "Room 3 - follow-up next week"],
    "clip5_presenter_badge": ["VISITOR PASS"],
}


def transcript(path, model):
    import imageio_ffmpeg
    with tempfile.TemporaryDirectory() as td:
        w = os.path.join(td, "a.wav")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(path),
                        "-ac", "1", "-ar", "16000", w], check=True)
        return " ".join(s.text for s in model.transcribe(w)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--configs", default="",
                    help="comma-separated configs to (re)check; others are kept from an existing --json-out")
    a = ap.parse_args()
    root = Path(a.root)
    only = {c for c in a.configs.split(",") if c}
    from faster_whisper import WhisperModel
    wm = WhisperModel("small.en", device="cuda", compute_type="float16")
    res = json.loads(Path(a.json_out).read_text()) if only and Path(a.json_out).exists() else {}
    for clip, items in ITEMS.items():
        vids = {"original": root / "inputs" / f"{clip}.mp4"}
        vids.update({p.stem: p for p in sorted((root / "outputs" / clip).glob("*.mp4"))})
        if only:
            vids = {k: v for k, v in vids.items() if k in only}
        res.setdefault(clip, {})
        for cfg, path in vids.items():
            per = ocr_texts(path, a.step)
            text = norm(transcript(path, wm)) if any(k == "speech" for k, *_ in items) else ""
            rows = []
            for kind, label, gt in items:
                if kind == "screen":
                    present = any(fuzzy_in(gt, ls) for ls in per)
                else:
                    present = any(t in text for t in gt)
                rows.append({"channel": kind, "item": label, "present": bool(present),
                             "caught": not present})
            ben = BENIGN.get(clip, [])
            kept = [any(fuzzy_in(b, ls) for ls in per) for b in ben]
            res[clip][cfg] = {"items": rows,
                              "benign_lines_readable": f"{sum(kept)}/{len(ben)}" if ben else None}
            print(clip, cfg, [(r["item"], r["caught"]) for r in rows], flush=True)
    Path(a.json_out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
