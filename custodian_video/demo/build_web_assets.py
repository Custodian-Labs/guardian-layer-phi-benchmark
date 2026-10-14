"""Transcode the precomputed web-demo outputs into web/video/ and write manifest.json.

  python demo/build_web_assets.py --root ../data/video_demo --web ../web/video

Per output: H.264 (libx264, yuv420p, +faststart), max 1280x720, CRF 27, AAC 96 kb/s.
One poster JPG per clip. The manifest carries clip metadata (source, author, licence),
config descriptions, file paths, durations, and an audit summary per output with
COUNTS ONLY: no OCR text, transcript, preview, hash or surrogate value is copied.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
MAX_BYTES = 25 * 1024 * 1024

CLIPS = [
    {"id": "clip1_interview_synthpii", "title": "Interview + fake intake screen",
     "blurb": "Interview (first 11 s). A synthetic 'patient intake' screen with a fake name, phone "
              "and account number is composited in, and the audio is a TTS voice reading the fake "
              "name and phone number.",
     "tests": ["faces", "text", "speech"], "planted": True,
     "source": {"file": "Interview on extreme weather with physical geographer Hannah Cloke – The Royal Society.webm",
                "url": "https://commons.wikimedia.org/wiki/File:Interview_on_extreme_weather_with_physical_geographer_Hannah_Cloke_%E2%80%93_The_Royal_Society.webm",
                "author": "The Royal Society", "licence": "CC BY 3.0",
                "licence_url": "https://creativecommons.org/licenses/by/3.0/",
                "changes": "Trimmed to 51.0–69.2 s; synthetic screen panel with fake PII added; audio replaced by Piper TTS reading fake PII."}},
    {"id": "clip2_street_faces", "title": "Busy street, many small faces",
     "blurb": "Hand-held walk through a Tokyo street: dozens of small, partly defocused faces.",
     "tests": ["faces"], "planted": False,
     "source": {"file": "Pro dog walker - Tokyo - 2024 Nov 1.webm",
                "url": "https://commons.wikimedia.org/wiki/File:Pro_dog_walker_-_Tokyo_-_2024_Nov_1.webm",
                "author": "Nesnad", "licence": "CC BY-SA 4.0",
                "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "changes": "Re-encoded to H.264."}},
    {"id": "clip3_traffic_plates", "title": "Traffic with licence plates",
     "blurb": "Street traffic in Ljubljana: vans, cars with readable plates, pedestrians and shop signs "
              "(signage should stay visible).",
     "tests": ["faces", "plates", "text"], "planted": False,
     "source": {"file": "Road traffic on Stritarjeva street.webm",
                "url": "https://commons.wikimedia.org/wiki/File:Road_traffic_on_Stritarjeva_street.webm",
                "author": "Sounds of Changes", "licence": "CC BY 3.0",
                "licence_url": "https://creativecommons.org/licenses/by/3.0/",
                "changes": "Trimmed to 22–34 s."}},
    {"id": "clip4_closeup_caption", "title": "Close-up speaker + name caption",
     "blurb": "A speaker talking to camera in close-up. The source's real name caption is fully covered "
              "by a synthetic lower-third with a fake name, member ID and e-mail. Real speech, no PII in it.",
     "tests": ["faces", "text", "speech"], "planted": True,
     "source": {"file": "Wikiclub Cebu Pagkat-on sa Sugbo Interview.webm",
                "url": "https://commons.wikimedia.org/wiki/File:Wikiclub_Cebu_Pagkat-on_sa_Sugbo_Interview.webm",
                "author": "Lady01v", "licence": "CC BY-SA 4.0",
                "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "changes": "Trimmed to 48.3–60.3 s; original caption covered by a synthetic lower-third with fake PII."}},
    {"id": "clip5_presenter_badge", "title": "Presenter wearing an ID badge",
     "blurb": "A presenter in a medium shot. The source caption is cropped away and a synthetic visitor "
              "badge (fake name, ID and phone) is pinned to her shirt and follows her movement. Real speech.",
     "tests": ["faces", "text", "speech"], "planted": True,
     "source": {"file": "Wiki Pagkat-on sa Sugbo sahring experience.webm",
                "url": "https://commons.wikimedia.org/wiki/File:Wiki_Pagkat-on_sa_Sugbo_sahring_experience.webm",
                "author": "Lady01v", "licence": "CC BY-SA 4.0",
                "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "changes": "Trimmed to 34–46 s; cropped to remove the caption and scaled to 720p; synthetic badge with fake PII added."}},
    {"id": "clip6_meeting_adults", "title": "Meeting table, eight adults",
     "blurb": "Hand-held shot of eight adults around a meeting table while the camera zooms out: faces "
              "from about 30 to 80 px, profiles, a person seen from behind, occlusion by hands and objects.",
     "tests": ["faces"], "planted": False,
     "source": {"file": "Wikimedia Meetup - St Johns Church - Kolkata 2016-09-10 9382.webm",
                "url": "https://commons.wikimedia.org/wiki/File:Wikimedia_Meetup_-_St_Johns_Church_-_Kolkata_2016-09-10_9382.webm",
                "author": "Biswarup Ganguly", "licence": "CC BY 3.0",
                "licence_url": "https://creativecommons.org/licenses/by/3.0/",
                "changes": "Trimmed to 66.0–78.0 s and scaled to 1280×720; original audio kept."}},
]

CONFIGS = [
    {"id": "original", "label": "Original", "family": "none",
     "desc": "The unprocessed input clip.", "licence": ""},
    {"id": "surrogate_research", "label": "Surrogate face · research models (experimental)", "family": "transform",
     "guardian": True, "research": True, "default": True,
     "desc": "Default (experimental). Every person whose face is large enough gets one synthetic identity for the "
             "whole clip: a photorealistic face of nobody, generated by an SDXL portrait model (RealVisXL) from "
             "coarse attributes (age band, gender presentation, skin tone, hair), never from the person's face and "
             "never a real person. A local vision-language model checks every generated portrait and rejects any "
             "whose apparent gender presentation, age band or skin tone does not match the person, then ranks the "
             "rest by how natural and good-looking they are. The chosen face is swapped onto each frame with a "
             "research face-swap model, so the head pose, gaze, blinks and lighting are the original's. The "
             "original mouth and lips are kept so the speech still matches, and hair, ears and background are "
             "never replaced. Fail closed: faces smaller than 64 px (for example the passers-by on clip 2 and the "
             "meeting on clip 6) and uncertain faces are blurred, and so is any frame in which a face recogniser "
             "can still match the face to the original. Among the best-looking candidates, the identity is the "
             "one that passes that check; a failing frame is first retried with a stronger change; the change "
             "between synthetic face and blur is cross-faded over a few frames, and short synthetic runs between "
             "blurred ones are blurred too, so faces do not flicker. "
             "Plates are blurred; on-screen and spoken personal data are replaced as in Blur + Custodian "
             "Guardian Layer. Uses research-only models with non-commercial licences.",
     "licence": "Research-only models (non-commercial licences), not for commercial use: HyperSwap face swap "
                "(FaceFusion, ResearchRAIL), InsightFace ArcFace identity embedding (non-commercial), BiSeNet face "
                "parsing (MIT code, non-commercial CelebAMask-HQ training data), XSeg occlusion mask (GPL-3.0), "
                "RealVisXL V5.0 SDXL portrait model (CreativeML OpenRAIL++-M) with the SDXL-VAE fp16 fix (MIT), "
                "Qwen3-VL attributes and portrait check (Apache-2.0); plus the Blur + Guardian Layer components."},
    {"id": "blur", "label": "Blur", "family": "redact", "recommended": True,
     "desc": "Safest. Redact: EgoBlur finds faces and plates and blurs them; only the PII words on screen are "
             "boxed; spoken PII is bleeped. Licence-safe components only.",
     "licence": "Licence-safe (Apache-2.0 / MIT components)"},
    {"id": "box", "label": "Solid box", "family": "redact",
     "desc": "Redact like Blur, but faces get a solid elliptical box instead of a blur.",
     "licence": "Licence-safe (Apache-2.0 / MIT components)"},
    {"id": "blur_fake_pii", "label": "Blur + fake text/speech", "family": "redact",
     "desc": "Faces and plates are blurred exactly as in Blur; on-screen personal data is re-typeset with a "
             "realistic fake value of the same type and spoken personal data is replaced by a TTS fake value "
             "(the same fake value on screen and in speech).",
     "licence": "Licence-safe (Apache-2.0 / MIT components; Piper TTS GPL-3.0 optional plug-in)"},
    {"id": "blur_guardian", "label": "Blur + Custodian Guardian Layer", "family": "redact", "guardian": True,
     "desc": "Faces and plates are blurred exactly as in Blur. On-screen and spoken personal data are replaced "
             "with values from the Custodian Guardian Layer, Custodian's own transform that rewrites personal "
             "data into realistic values of the same type. The local detector (LLM + regex) and Guardian Layer "
             "together decide which words are personal data, and only those words are changed. A Guardian Layer "
             "value is used only when it replaces the whole value (not, say, just the area code of a phone "
             "number); otherwise a local fake value of the same type fills in. The same value is used on screen "
             "and in speech. For this configuration only, OCR lines and the transcript are sent to the "
             "Guardian Layer API.",
     "licence": "Licence-safe local components (Apache-2.0 / MIT; Piper TTS GPL-3.0 optional plug-in) + "
                "Custodian Guardian Layer API"},
]


def ff(*args):
    subprocess.run([FF, "-v", "error", "-y", *args], check=True)


def probe(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    return {"fps": round(fps, 3), "frames": int(n), "width": w, "height": h,
            "duration_s": round(n / fps, 2) if fps else None}


def transcode(src, dst, crf):
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff("-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
       "-vf", "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease,"
              "scale=trunc(iw/2)*2:trunc(ih/2)*2",
       "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
       "-profile:v", "high", "-g", "50", "-movflags", "+faststart",
       "-c:a", "aac", "-b:a", "96k", "-ac", "2", str(dst))
    if dst.stat().st_size > MAX_BYTES:
        transcode(src, dst, crf + 3)


def audit_summary(a: dict) -> dict:
    """Counts only. Never copies text, previews, hashes or surrogate values."""
    det = a.get("detections", [])
    faces = [d for d in det if d.get("channel") == "faces"]
    acts = lambda d: " ".join(d.get("action", []) if isinstance(d.get("action"), list)
                              else [str(d.get("action", ""))])
    text = [d for d in det if d.get("channel") == "text"]
    text_masked = [d for d in text if "kept_visible" not in acts(d)]
    audio = [d for d in det if d.get("channel") == "audio"]
    s = a.get("summary", {})
    t = a.get("timing", {})
    g = (a.get("pii_engine") or {}).get("guardian_layer") or {}
    out = {
        "face_tracks": len(faces),
        "face_tracks_surrogate": sum("surrogate_face" in acts(d) for d in faces),
        "face_tracks_blur_only": sum("surrogate_face" not in acts(d) and
                                     ("blur" in acts(d) or "pixel" in acts(d)) for d in faces),
        "face_tracks_box": sum(acts(d).strip() == "box" for d in faces),
        "plate_tracks": s.get("plate_tracks", sum(d.get("channel") == "plates" for d in det)),
        "text_pii_tracks_masked": len(text_masked),
        "text_pii_tracks_surrogate": sum("surrogate_text" in acts(d) for d in text_masked),
        "text_tracks_kept_visible_scene_gate": len(text) - len(text_masked),
        "text_pii_types": sorted({d.get("pii_label", "?") for d in text_masked}),
        "speech_pii_events": len(audio),
        "speech_actions": sorted({acts(d) for d in audio}),
        "speech_pii_types": sorted({d.get("pii_label", "?") for d in audio}),
        "frames_with_any_mask": s.get("frames_with_any_mask"),
        "frames": (a.get("video") or {}).get("frames"),
        "fps": (a.get("video") or {}).get("fps"),
        "processing_fps": t.get("fps_end_to_end"),
        "processing_s": t.get("total_s"),
        "pii_backend": (a.get("settings") or {}).get("pii_backend"),
        "face_detector": (a.get("models") or {}).get("faces", "").split(" (")[0],
        "surrogate_min_face": (a.get("settings") or {}).get("surrogate_min_face"),
    }
    if g:
        out["guardian"] = {k: g.get(k) for k in ("http_calls", "cache_hits", "guardian_edits_used",
                                                 "guardian_edits_ignored_non_pii",
                                                 "surrogates_from_guardian", "surrogates_local_fallback")
                           if k in g}
    if out["pii_backend"] == "guardian-replace":
        # who produced each replacement VALUE: per track/event and per distinct value (counts only)
        rep = {}
        for ch, items in (("screen", [d for d in text_masked if "surrogate_text" in acts(d)]),
                          ("speech", [d for d in audio if "tts" in acts(d)])):
            src = lambda d: "guardian" if d.get("surrogate_source") == "guardian" else "local"
            rep[ch] = {"guardian": sum(src(d) == "guardian" for d in items),
                       "local": sum(src(d) == "local" for d in items)}
            vals = {}
            for d in items:
                vals.setdefault(d.get("value_key16") or d.get("id"), set()).add(src(d))
            rep[ch + "_values"] = {"guardian": sum(v == {"guardian"} for v in vals.values()),
                                   "local": sum(v != {"guardian"} for v in vals.values())}
        allv = {}
        for d in [*text_masked, *audio]:
            if "surrogate" in acts(d) or "tts" in acts(d):
                allv.setdefault(d.get("value_key16") or d.get("id"), set()).add(
                    "guardian" if d.get("surrogate_source") == "guardian" else "local")
        rep["values"] = {"guardian": sum(v == {"guardian"} for v in allv.values()),
                         "local": sum(v != {"guardian"} for v in allv.values())}
        notes = {}
        for d in [*text_masked, *audio]:
            if d.get("surrogate_note"):
                notes[d["surrogate_note"]] = notes.get(d["surrogate_note"], 0) + 1
        rep["fallback_reasons"] = notes
        out["replacements"] = rep
    return out


PII_LIKE = re.compile(r"(margaret|holloway|whitfield|priya|raman|555-?01|88427139|30418|558214)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--web", required=True)
    ap.add_argument("--crf", type=int, default=27)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    root, web = Path(a.root), Path(a.web)
    checklist = {}
    if (root / "checklist.json").exists():
        checklist = json.loads((root / "checklist.json").read_text())
    # optional: experiments/face_e/privacy_report.py --json-out <root>/face_privacy.json
    privacy = {}
    if (root / "face_privacy.json").exists():
        privacy = json.loads((root / "face_privacy.json").read_text())
    # optional: experiments/face_g/page_eval.sh -> re-ID (3 recognisers) / floors / motion of surrogate_research
    surr_eval = {}
    if (root / "surrogate_research_eval.json").exists():
        surr_eval = json.loads((root / "surrogate_research_eval.json").read_text()).get("clips", {})
    man = {"generated_by": "custodian_video/demo/build_web_assets.py", "configs": CONFIGS, "clips": []}
    for c in CLIPS:
        cid = c["id"]
        src = root / "inputs" / f"{cid}.mp4"
        d = web / "clips" / cid
        d.mkdir(parents=True, exist_ok=True)
        info = probe(src)
        poster = d / "poster.jpg"
        if a.force or not poster.exists():
            ff("-ss", str(min(2.0, info["duration_s"] / 3)), "-i", str(src), "-frames:v", "1",
               "-vf", "scale=640:-2", "-q:v", "4", str(poster))
        entry = {**c, "source_video": info, "poster": f"video/clips/{cid}/poster.jpg", "outputs": {}}
        for cfg in CONFIGS:
            k = cfg["id"]
            s = src if k == "original" else root / "outputs" / cid / f"{k}.mp4"
            if not s.exists():
                entry["outputs"][k] = {"available": False}
                continue
            dst = d / f"{k}.mp4"
            if a.force or not dst.exists() or dst.stat().st_mtime < s.stat().st_mtime:
                transcode(s, dst, a.crf)
            o = {"available": True, "src": f"video/clips/{cid}/{k}.mp4",
                 "bytes": dst.stat().st_size, "duration_s": probe(dst)["duration_s"]}
            if k != "original":
                o["audit"] = audit_summary(json.loads(s.with_suffix(".audit.json").read_text()))
            pr = privacy.get(cid, {}).get(k)
            if pr and pr.get("n_ref"):
                o["face_check"] = {x: pr.get(x) for x in (
                    "n_ref", "coverage", "n_ref_confident", "coverage_confident",
                    "coverage_confident_egoblur_only", "sface_reid_masked", "sface_reid_confident",
                    "arc_reid_masked")}
            if k == "surrogate_research" and surr_eval.get(cid):
                o["surrogate_eval"] = surr_eval[cid]
            if checklist.get(cid, {}).get(k):
                o["checklist"] = checklist[cid][k]
            entry["outputs"][k] = o
        man["clips"].append(entry)
    txt = json.dumps(man, indent=1, ensure_ascii=False)
    # Belt and braces: the manifest must not carry planted values or surrogate strings.
    hits = PII_LIKE.findall(txt)
    assert not hits, f"PII-like strings in manifest: {set(hits)}"
    (web / "manifest.json").write_text(txt)
    tot = sum(p.stat().st_size for p in web.rglob("*") if p.is_file())
    big = max((p for p in web.rglob("*.mp4")), key=lambda p: p.stat().st_size)
    print(f"web/video total {tot / 2**20:.1f} MiB; largest {big} {big.stat().st_size / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()
