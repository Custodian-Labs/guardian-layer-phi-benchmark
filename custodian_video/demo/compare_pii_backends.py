"""Compare PII backends (local | guardian | both) on the demo clips from their
audit logs (+ optional evaluate.py JSON for OCR / re-transcript leakage).

  python demo/compare_pii_backends.py --runs local=demo/outputs/v02 guardian=... \
      --gt demo/inputs/ground_truth.json --md out.md

For each backend directory it expects clip{1,2,3}_{redact,transform}.audit.json
and (optional) clip1_eval.json.  Reports, without printing any raw PII:
  planted PII caught (clip1 on-screen + spoken), false positives (text tracks /
  audio events that are not planted PII; clip2/3 have no planted text PII),
  screen<->audio surrogate consistency (same value -> same surrogate),
  Guardian Layer HTTP calls / cache hits / latency, and Guardian-specific
  diagnostics (spans missed by Guardian, non-PII rewrites ignored, ...).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def vk(s: str) -> str:
    return hashlib.sha256(re.sub(r"[\W_]", "", s.lower()).encode()).hexdigest()[:16]


def norm(s):
    return re.sub(r"[\W_]", "", (s or "").lower())


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def clip1_stats(a: dict, gt: dict) -> dict:
    gt_screen = {vk(x) for x in gt["onscreen_pii"]}
    # spoken GT as Whisper writes it (digits) — matched through the screen values
    text = [d for d in a["detections"] if d["channel"] == "text" and "kept_visible" not in str(d["action"])]
    audio = [d for d in a["detections"] if d["channel"] == "audio"]
    t_keys = {d.get("value_key16") for d in text}
    a_keys = {d.get("value_key16") for d in audio}
    caught_screen = len(gt_screen & t_keys)
    caught_audio = len(gt_screen & a_keys)
    fp_text = len({d.get("value_key16") for d in text} - gt_screen)
    fp_audio = len(a_keys - gt_screen)
    # consistency: same value on screen and in speech -> same surrogate (format-free)
    sur_t = {d["value_key16"]: d.get("surrogate_planned") for d in text if d.get("value_key16")}
    sur_a = {d["value_key16"]: d.get("surrogate_planned") for d in audio if d.get("value_key16")}
    both = set(sur_t) & set(sur_a)
    same = sum(norm(sur_t[k]) == norm(sur_a[k]) for k in both)
    src = sorted({d.get("surrogate_source") for d in text + audio if d.get("surrogate_source")})
    return {"planted_onscreen_caught": f"{caught_screen}/{len(gt_screen)}",
            "planted_spoken_caught": f"{caught_audio}/{len(gt_screen)}",
            "fp_text_values": fp_text, "fp_audio_values": fp_audio,
            "screen_audio_surrogate_consistent": f"{same}/{len(both)}",
            "surrogate_sources": src}


def run(dirs: dict[str, Path], gt: dict) -> dict:
    res = {}
    for name, d in dirs.items():
        r = {}
        calls = hits = 0
        lat = 0.0
        diag = {}
        for clip in ("clip1", "clip2", "clip3"):
            for mode in ("redact", "transform"):
                a = load(d / f"{clip}_{mode}.audit.json")
                if a is None:
                    continue
                pe = a.get("pii_engine") or {}
                g = pe.get("guardian_layer") or {}
                calls += g.get("http_calls", 0)
                hits += g.get("cache_hits", 0)
                lat += g.get("latency_s", 0.0)
                if mode == "transform":
                    for k in ("guardian_spans", "local_spans", "union_spans", "guardian_only_spans",
                              "local_only_spans(guardian_missed)", "guardian_edits",
                              "guardian_edits_used", "guardian_edits_ignored_non_pii",
                              "guardian_sensitive_left_verbatim", "guardian_flagged_raw", "guardian_only_dropped_unchanged", "guardian_only_dropped_inflection", "surrogates_from_guardian",
                              "surrogates_local_fallback"):
                        if k in g:
                            diag[k] = diag.get(k, 0) + g[k]
                s = a["summary"]
                key = f"{clip}_{mode}"
                r[key] = {"text_tracks_masked": s["onscreen_text_pii_tracks"],
                          "text_tracks_left_visible_by_scene_gate":
                              s.get("onscreen_text_tracks_kept_visible_scene_gate", 0),
                          "audio_events": s["audio_pii_events"],
                          "text_ocr_llm_s": a["timing"].get("text_ocr_llm_s"),
                          "audio_s": a["timing"].get("audio_s"),
                          "total_s": a["timing"].get("total_s")}
                if clip == "clip1":
                    r[key].update(clip1_stats(a, gt))
        ev = load(d / "clip1_eval.json")
        if ev:
            r["clip1_eval"] = {k: {"onscreen_gt_readable": v.get("gt_pii_strings_readable"),
                                   "non_pii_kept": v.get("non_pii_lines_readable_frac"),
                                   "surrogates_redetected": v.get("surrogates_redetected_as_pii")}
                               for k, v in ev.get("text", {}).items()}
            for k, v in ev.get("audio", {}).items():
                r["clip1_eval"].setdefault(k, {})["spoken_gt_tokens_heard"] = v.get("gt_pii_tokens_heard")
        r["guardian_http_calls"] = calls
        r["guardian_cache_hits"] = hits
        r["guardian_latency_s"] = round(lat, 2)
        r["guardian_diagnostics(transform runs)"] = diag
        res[name] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=dir")
    ap.add_argument("--gt", default=str(Path(__file__).parent / "inputs" / "ground_truth.json"))
    ap.add_argument("--json-out")
    a = ap.parse_args()
    gt = json.loads(Path(a.gt).read_text())["clip1_interview_synthpii.mp4"]
    res = run({k: Path(v) for k, v in (x.split("=", 1) for x in a.runs)}, gt)
    s = json.dumps(res, indent=1)
    print(s)
    if a.json_out:
        Path(a.json_out).write_text(s)


if __name__ == "__main__":
    main()
