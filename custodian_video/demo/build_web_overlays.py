#!/usr/bin/env python3
"""Per-output "what was changed" overlays for the web demo (web/video.html).

Reads each output's .audit.json (data/video_demo/outputs/<clip>/<cfg>.audit.json)
and writes web/video/clips/<clip>/<cfg>.marks.json with, per track, the frames and
boxes that were masked/replaced and a short label ("surrogate face", "blurred face",
"text replaced", ...), plus speech-PII time spans. Boxes are normalised to 0..1000.
Only geometry, frame indices and PII *types* are exported -- never PII strings (the
audit itself only holds hashes/previews; those fields are not read).

  python custodian_video/demo/build_web_overlays.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "video_demo" / "outputs"
WEB = ROOT / "web" / "video" / "clips"


def face_label(actions: list[str], cfg: str) -> str:
    if any(a.startswith("surrogate") for a in actions):
        # fail-closed backends blur the frames where the identity check failed
        return "synthetic face (blurred where unverified)" if any("blur" in a for a in actions) else "surrogate face"
    return "face (solid box)" if cfg == "box" else "face (blurred)"


def label(d: dict, cfg: str, guardian: bool = False) -> tuple[str, str]:
    """guardian: the run used --pii-backend guardian-replace, so every replaced
    value says whether Custodian Guardian Layer or the local fallback made it."""
    acts = d["action"] if isinstance(d["action"], list) else [d["action"]]
    kind = d.get("type", "").replace("onscreen_", "").replace("speech_", "")
    if d["channel"] == "faces":
        return "face", face_label(acts, cfg)
    if d["channel"] == "plates":
        return "plate", "licence plate (blurred)"
    replaced = any("surrogate" in a for a in acts)
    by_g = d.get("surrogate_source") == "guardian"
    if d["channel"] == "text":
        if guardian and replaced:
            return "text", f"on-screen {kind}: " + ("replaced by Guardian Layer" if by_g else "replaced (local fallback)")
        return "text", f"on-screen {kind} ({'replaced' if replaced else 'masked'})"
    if guardian and replaced:
        return "speech", f"spoken {kind}: " + ("replaced by Guardian Layer (TTS)" if by_g else
                                                "replaced (local fallback, TTS)")
    return "speech", f"spoken {kind} ({'replaced by TTS' if replaced else 'bleeped'})"


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_web_assets import CONFIGS  # only configs shown on the page
    shown = {c["id"] for c in CONFIGS}
    n = 0
    for clip_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        for ap in sorted(clip_dir.glob("*.audit.json")):
            cfg = ap.name[: -len(".audit.json")]
            if cfg not in shown:
                continue
            dst_dir = WEB / clip_dir.name
            if not dst_dir.exists():
                continue
            a = json.loads(ap.read_text())
            W, H = a["video"]["width"], a["video"]["height"]
            tracks, speech = [], []
            guardian = (a.get("settings") or {}).get("pii_backend") == "guardian-replace"
            for d in a["detections"]:
                ch, lab = label(d, cfg, guardian)
                if ch == "speech":
                    speech.append({"t0": round(d["t_start"], 2), "t1": round(d["t_end"], 2), "label": lab})
                    continue
                fr, bx = [], []
                for b in d.get("boxes") or []:
                    f, x1, y1, x2, y2 = b[:5]
                    fr.append(int(f))
                    bx.append([round(1000 * x1 / W), round(1000 * y1 / H),
                               round(1000 * x2 / W), round(1000 * y2 / H)])
                if fr:
                    tr = {"ch": ch, "label": lab, "fr": fr, "bx": bx}
                    if guardian and ch == "text":   # colour key: guardian / local fallback
                        tr["src"] = "guardian" if d.get("surrogate_source") == "guardian" else "local"
                    tracks.append(tr)
            (dst_dir / f"{cfg}.marks.json").write_text(json.dumps(
                {"fps": a["video"]["fps"], "w": W, "h": H, "tracks": tracks, "speech": speech},
                separators=(",", ":")))
            n += 1
    print(f"wrote {n} overlay files")


if __name__ == "__main__":
    main()
