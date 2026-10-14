"""Accuracy evidence for the web video demo (web/video.html -> web/video/accuracy.json).

Every number is measured against an INDEPENDENT reference, never against the
pipeline's own detections:

faces   Reference faces = YuNet (not used for masking decisions in the redact
        path; EgoBlur is the masking detector) on the ORIGINAL clip, every
        --step-th frame, short side >= --min-face, two score levels:
        "all" >= 0.6 and "confident" >= 0.8.  A reference face is COVERED when
        >= 50 % of its box area lies inside the union of the output's masked
        face boxes for that frame (audit log).  Every uncovered reference face
        is cropped from the original and judged by a local VLM (Qwen3-VL-8B):
        real face / head from behind / not a head.  Real-face recall =
        covered / (covered + misses judged a real face).
precision  Up to --n-prec masked face boxes per clip x config, sampled round-robin
        across face tracks (fixed seed), cropped from the ORIGINAL frame and
        judged by the same VLM: face (A) / head from behind (B) / not a head (C).
        Precision = (A + B) / n with a 95 % Wilson interval; C items are listed
        by object type.
plates  Reference plates = open-image-models YOLOv9-t (MIT) on the original, every
        2nd frame (conf >= 0.35); covered as for faces (>= 50 % of the box).
text    Planted values: web_pii_checklist.py (OCR of the output).  Masked text
        tracks: the masked region is cropped from the ORIGINAL at a detected
        frame, OCR'd and classified (planted PII / other personal data / not
        personal) by the local LLM; only counts and types leave this script.
speech  Planted values: web_pii_checklist.py (Whisper re-transcript).  False
        bleeps: Whisper word timestamps on the original; each speech-PII event of
        the audit is matched to the original words it covers and classified
        (planted PII / personal / not personal).  A word-level diff of original
        vs output transcripts counts words changed inside the event spans
        (altered by the pipeline) and outside them (ASR run-to-run variation).

Raw text, crops and transcripts stay in --work (outside git).  The JSON written
to --out contains counts, rates and PII TYPES only.

  python demo/web_accuracy.py faces   --root ../data/video_demo   # CPU
  python demo/web_accuracy.py plates  --root ../data/video_demo   # CPU
  python demo/web_accuracy.py vlm     --root ../data/video_demo   # GPU (Qwen3-VL via $CUSTODIAN_VLM_PYTHON)
  python demo/web_accuracy.py text    --root ../data/video_demo   # CPU OCR + Ollama
  python demo/web_accuracy.py speech  --root ../data/video_demo   # GPU (Whisper) + Ollama
  python demo/web_accuracy.py assemble --root ../data/video_demo --out ../web/video/accuracy.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS = ["blur", "box", "blur_fake_pii", "blur_guardian", "surrogate_research"]
MODELS = Path.home() / ".cache" / "custodian_video"
VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
VLM_PROMPT = ("This is a crop from a video frame. Look at the CENTRE of the crop. Is there a human head there?\n"
              "A) yes - a human face is visible (frontal, profile, partly hidden, small or blurry)\n"
              "B) yes - a human head seen from behind or from above; the face itself is not visible\n"
              "C) no human head - something else\n"
              "Reply with the letter. If C, add 1-3 words naming what it is, e.g. 'C: poster'.")
PAIR_PROMPT = ("Left: a crop from an ORIGINAL video frame. Right: the same place in the de-identified "
               "version of that frame. Is a real person's face from the left image still recognisably visible "
               "and unaltered in the right image?\n"
               "A) yes - the same face is still clearly visible, not blurred, covered or replaced\n"
               "B) no - in the right image that face is blurred, covered, replaced by a different face, or "
               "there is no face in the left image\n"
               "Reply with the letter only.")
PLATE_PROMPT = ("This is a crop from a street video. Is there a vehicle licence plate in the centre of the crop?\n"
                "A) yes - a licence plate\nB) no - something else\n"
                "Reply with the letter. If B, add 1-3 words naming what it is.")


def _write_stage(a, path: Path, res: dict, nested: bool):
    """Write a stage result.  With --configs (a subset), merge into the existing file so
    the other configurations' numbers stay exactly as they were measured."""
    if getattr(a, "configs", None) and path.exists():
        old = json.loads(path.read_text())
        for clip, r in res.items():
            if clip not in old:
                old[clip] = r
            elif nested:
                # clip-level reference data (refs, n_frames, ...) is recomputed from the input on
                # every run: take the fresh one, or a changed input (e.g. a trimmed clip) keeps stale refs
                old[clip].update({k: v for k, v in r.items() if k != "configs"})
                old[clip].setdefault("configs", {}).update(r.get("configs", {}))
            else:
                old[clip].update(r)
        res = old
    return res


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def clips_of(root: Path):
    return sorted(p.stem for p in (root / "inputs").glob("*.mp4"))


def audit_of(root, clip, cfg):
    p = root / "outputs" / clip / f"{cfg}.audit.json"
    return json.loads(p.read_text()) if p.exists() else None


def boxes_by_frame(audit, channel):
    bb = {}
    for d in audit["detections"]:
        if d.get("channel") == channel:
            for b in d.get("boxes") or []:
                bb.setdefault(int(b[0]), []).append([float(v) for v in b[1:5]])
    return bb


def covered_frac(ref, masks):
    """Share of the reference box area inside the union of the mask boxes."""
    import numpy as np
    x1, y1, x2, y2 = (int(math.floor(ref[0])), int(math.floor(ref[1])),
                      int(math.ceil(ref[2])), int(math.ceil(ref[3])))
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    g = np.zeros((h, w), bool)
    for m in masks:
        a1, b1 = max(0, int(m[0]) - x1), max(0, int(m[1]) - y1)
        a2, b2 = min(w, int(math.ceil(m[2])) - x1), min(h, int(math.ceil(m[3])) - y1)
        if a2 > a1 and b2 > b1:
            g[b1:b2, a1:a2] = True
    return float(g.mean())


def read_frames(path, wanted):
    import cv2
    cap = cv2.VideoCapture(str(path))
    out, f, wanted = {}, 0, set(wanted)
    last = max(wanted) if wanted else -1
    while f <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if f in wanted:
            out[f] = fr
        f += 1
    cap.release()
    return out


def save_crop(frame, box, path, pad=0.35, min_side=192):
    import cv2
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    s = max(bw, bh) * (1 + 2 * pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X1, Y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    X2, Y2 = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0:
        return False
    z = max(1.0, min_side / max(c.shape[:2]))
    if z > 1:
        c = cv2.resize(c, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


def save_pair(orig, out, box, path, pad=0.35, side=224):
    import cv2
    import numpy as np
    tmp = []
    for k, fr in enumerate((orig, out)):
        q = path.with_name(path.stem + f"_{k}.jpg")
        save_crop(fr, box, q, pad=pad, min_side=side)
        im = cv2.imread(str(q))
        q.unlink()
        tmp.append(cv2.resize(im, (side, side)))
    gap = np.full((side, 12, 3), 255, np.uint8)
    cv2.imwrite(str(path), np.hstack([tmp[0], gap, tmp[1]]), [cv2.IMWRITE_JPEG_QUALITY, 92])


# ------------------------------------------------------------------ faces
def _face_boxes(au):
    if au is None:
        return None
    return sorted(json.dumps(d.get("boxes")) for d in au["detections"] if d.get("channel") == "faces")


def stage_faces(a):
    import cv2
    root, work = Path(a.root), Path(a.work)
    res = {}
    for clip in clips_of(root):
        inp = root / "inputs" / f"{clip}.mp4"
        cap = cv2.VideoCapture(str(inp))
        W, H = int(cap.get(3)), int(cap.get(4))
        det = cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2023mar.onnx"), "", (W, H), 0.6, 0.3, 5000)
        refs, f = [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if f % a.step == 0:
                _, faces = det.detect(fr)
                for r in ([] if faces is None else faces):
                    if min(r[2], r[3]) >= a.min_face:
                        refs.append([f, float(r[0]), float(r[1]), float(r[0] + r[2]), float(r[1] + r[3]),
                                     round(float(r[14]), 3)])
            f += 1
        cap.release()
        res[clip] = {"n_frames": f, "refs": refs, "configs": {}}
        need = {}   # frame -> {key: box}
        for cfg in CONFIGS:
            au = audit_of(root, clip, cfg)
            if au is None:
                continue
            mb = boxes_by_frame(au, "faces")
            cov, misses = [], []
            for fr_i, *box, sc in refs:
                c = covered_frac(box, mb.get(fr_i, []))
                cov.append(round(c, 3))
                if c < 0.5:
                    key = f"{clip}_f{fr_i}_{int(box[0])}_{int(box[1])}_{int(box[2])}_{int(box[3])}"
                    misses.append(key)
                    need.setdefault(fr_i, {})[key] = box
            # precision sample: round-robin across face tracks, fixed seed.  A config
            # whose face masks are IDENTICAL to Blur's (same tracks, same boxes) gets
            # Blur's seed, i.e. the same sample, so identical masks get identical numbers.
            same = cfg != "blur" and _face_boxes(au) == _face_boxes(audit_of(root, clip, "blur"))
            rng = random.Random(f"{clip}|{'blur' if same else cfg}")
            tracks = [list(d.get("boxes") or []) for d in au["detections"] if d.get("channel") == "faces"]
            tracks = [t for t in tracks if t]
            for t in tracks:
                rng.shuffle(t)
            rng.shuffle(tracks)
            lens = [len(t) for t in tracks]
            sample, src, i = [], [], 0
            while len(sample) < a.n_prec and any(tracks):
                t = tracks[i % len(tracks)]
                if t:
                    b = t.pop()
                    key = f"{clip}_m{int(b[0])}_{int(b[1])}_{int(b[2])}_{int(b[3])}_{int(b[4])}"
                    sample.append(key)
                    src.append(i % len(tracks))
                    need.setdefault(int(b[0]), {})[key] = [float(v) for v in b[1:5]]
                i += 1
            # weight of a sample = masked frames of its track / samples drawn from that track
            weight = [round(lens[j] / src.count(j), 3) for j in src]
            res[clip]["configs"][cfg] = {"coverage": cov, "misses": misses, "prec_sample": sample,
                                         "prec_weight": weight, "n_face_tracks": len(tracks)}
        frames = read_frames(inp, need.keys())
        for fr_i, items in need.items():
            for key, box in items.items():
                p = work / "crops" / f"{key}.jpg"
                if not p.exists() and fr_i in frames:
                    save_crop(frames[fr_i], box, p, pad=0.35 if key[len(clip):].startswith("_f") else 0.1)
        # residual check: original | output side by side at every missed reference face
        for cfg, r in res[clip]["configs"].items():
            want = {}
            for key in r["misses"]:
                fr_i = int(re.search(r"_f(\d+)_", key[len(clip):])[1])
                want.setdefault(fr_i, []).append(key)
            if not want:
                continue
            outf = read_frames(root / "outputs" / clip / f"{cfg}.mp4", want.keys())
            for fr_i, keys in want.items():
                for key in keys:
                    p = work / "crops" / f"pair_{key}__{cfg}.jpg"
                    if p.exists() or fr_i not in outf or fr_i not in frames:
                        continue
                    box = need[fr_i][key]
                    save_pair(frames[fr_i], outf[fr_i], box, p)
        n_m = {c: len(v["misses"]) for c, v in res[clip]["configs"].items()}
        print(clip, "refs", len(refs), "misses", n_m, flush=True)
    (work / "faces.json").write_text(json.dumps(_write_stage(a, work / "faces.json", res, True)))


# ------------------------------------------------------------------ plates
def stage_plates(a):
    sys.path.insert(0, str(HERE.parent))
    from custodian_video.detectors import YoloPlateDetector
    import cv2
    root, work = Path(a.root), Path(a.work)
    det = YoloPlateDetector(device="cpu", threshold=0.35)
    res = {}
    for clip in clips_of(root):
        if "plates" not in clip:
            continue
        inp = root / "inputs" / f"{clip}.mp4"
        cap = cv2.VideoCapture(str(inp))
        refs, f = [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if f % 2 == 0:
                for d in det(fr, f):
                    refs.append([f, *[float(v) for v in d.box], round(d.score, 3)])
            f += 1
        cap.release()
        res[clip] = {"refs": refs, "configs": {}}
        need = {}
        for cfg in CONFIGS:
            au = audit_of(root, clip, cfg)
            if au is None:
                continue
            mb = boxes_by_frame(au, "plates")
            cov, misses = [], []
            for fr_i, *box, sc in refs:
                c = covered_frac(box, mb.get(fr_i, []))
                cov.append(round(c, 3))
                if c < 0.5:
                    key = f"{clip}_p{fr_i}_{int(box[0])}_{int(box[1])}_{int(box[2])}_{int(box[3])}"
                    misses.append(key)
                    need.setdefault(fr_i, {})[key] = box
            res[clip]["configs"][cfg] = {"coverage": cov, "misses": misses,
                                         "n_plate_tracks": sum(d.get("channel") == "plates" for d in au["detections"])}
        frames = read_frames(inp, need.keys())
        for fr_i, items in need.items():
            for key, box in items.items():
                p = work / "crops" / f"{key}.jpg"
                if not p.exists() and fr_i in frames:
                    save_crop(frames[fr_i], box, p, pad=1.0, min_side=256)
        print(clip, "plate refs", len(refs), {c: len(v["misses"]) for c, v in res[clip]["configs"].items()})
    (work / "plates.json").write_text(json.dumps(_write_stage(a, work / "plates.json", res, True)))


# ------------------------------------------------------------------ VLM
def stage_vlm(a):
    work = Path(a.work)
    keys = set()
    for fn in ("faces.json", "plates.json"):
        p = work / fn
        if not p.exists():
            continue
        for clip, r in json.loads(p.read_text()).items():
            for cfg, c in r["configs"].items():
                keys.update(c.get("misses", []))
                keys.update(c.get("prec_sample", []))
                if fn == "faces.json":
                    keys.update(f"pair_{k}__{cfg}" for k in c.get("misses", []))
    out_p = work / "vlm.json"
    done = json.loads(out_p.read_text()) if out_p.exists() else {}
    todo = [k for k in sorted(keys) if k not in done and (work / "crops" / f"{k}.jpg").exists()]
    print(f"VLM: {len(todo)} crops to judge ({len(done)} cached)", flush=True)
    if todo:
        items = [{"id": k, "img": str(work / "crops" / f"{k}.jpg"),
                  "prompt": PAIR_PROMPT if k.startswith("pair_") else
                  PLATE_PROMPT if re.search(r"_p\d+_\d", k) else VLM_PROMPT}
                 for k in todo]
        fin, fout = work / "vlm_in.json", work / "vlm_out.json"
        fin.write_text(json.dumps({"model": VLM_MODEL, "items": items}))
        py = os.environ.get("CUSTODIAN_VLM_PYTHON", sys.executable)
        subprocess.run([py, str(Path(__file__).resolve()), "_vlm_worker", str(fin), str(fout)], check=True,
                       env={**os.environ, "HF_HUB_OFFLINE": "1"})
        done.update(json.loads(fout.read_text()))
        out_p.write_text(json.dumps(done, indent=0))


def _vlm_worker(fin, fout):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    req = json.load(open(fin))
    model = AutoModelForImageTextToText.from_pretrained(req["model"], dtype=torch.bfloat16, device_map="cuda")
    proc = AutoProcessor.from_pretrained(req["model"])
    res = {}
    for i, it in enumerate(req["items"]):
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(it["img"]).convert("RGB")},
                                             {"type": "text", "text": it["prompt"]}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=12, do_sample=False)
        res[it["id"]] = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        if i % 100 == 0:
            print(f"  vlm {i}/{len(req['items'])}", flush=True)
    json.dump(res, open(fout, "w"))


def vlm_letter(ans: str | None) -> tuple[str, str]:
    if not ans:
        return "?", ""
    m = re.match(r"\s*\(?([A-Ca-c])\b\)?[:.)\s-]*(.*)", ans)
    if not m:
        return "?", ans[:30]
    return m[1].upper(), m[2].strip().strip(".").lower()[:30]


# ------------------------------------------------------------------ LLM classification of strings
LLM_PROMPT = """A de-identification system masked the following text. Classify it.
Text: "{t}"
Answer with ONE line: a letter, a colon, and a 1-3 word type.
A) personal data of a private individual (person's name, phone number, e-mail, ID/account/member number, street address, date of birth, licence plate)
B) not personal data (organisation, brand, product, event or place name, a generic word or label, a number that is not an identifier, or unreadable/garbled text)
Example answers: "A: person name", "B: organisation name", "B: garbled text"."""


def llm_classify(text: str, model="qwen2.5:7b", url="http://localhost:11434") -> tuple[str, str]:
    import urllib.request
    body = json.dumps({"model": model, "prompt": LLM_PROMPT.format(t=text.replace('"', "'")),
                       "stream": False, "options": {"temperature": 0}}).encode()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(f"{url}/api/generate", body,
                                                                     {"Content-Type": "application/json"}),
                                              timeout=120).read())["response"]
    except Exception as e:  # noqa: BLE001
        return "?", f"llm error {type(e).__name__}"
    m = re.search(r"\b([AB])\s*[:)]\s*(.*)", r)
    if not m:
        return "?", r.strip()[:30]
    return m[1], m[2].strip().strip(".").lower()[:30]


def planted_values():
    sys.path.insert(0, str(HERE))
    from web_pii_checklist import BENIGN, ITEMS
    return ({c: [v for k, _, v in it if k == "screen"] for c, it in ITEMS.items()},
            {c: [v for k, _, v in it if k == "speech"] for c, it in ITEMS.items()}, BENIGN)


def planted_match(txt: str, value: str) -> bool:
    """Loose match of a (possibly garbled) OCR re-read against a planted value."""
    import difflib
    n = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
    a, b = n(txt), n(value)
    if len(a) < 3:
        return False
    if a in b or b in a or difflib.SequenceMatcher(None, a, b).ratio() >= 0.6:
        return True
    da, db = re.sub(r"\D", "", txt), re.sub(r"\D", "", value)
    return len(da) >= 4 and (da in db or difflib.SequenceMatcher(None, da, db).ratio() >= 0.7)


# ------------------------------------------------------------------ on-screen text
def stage_text(a):
    sys.path.insert(0, str(HERE))
    from evaluate import fuzzy_in, norm
    from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR
    ocr = RapidOCR(params={"Rec.lang_type": LangRec.EN, "Rec.ocr_version": OCRVersion.PPOCRV5,
                           "Rec.model_type": ModelType.MOBILE, "Det.ocr_version": OCRVersion.PPOCRV5,
                           "Det.lang_type": LangDet.CH, "Det.model_type": ModelType.MOBILE,
                           "Global.log_level": "error"})
    import cv2
    root, work = Path(a.root), Path(a.work)
    (work / "text_crops").mkdir(exist_ok=True)
    screen, _, benign = planted_values()
    raw, res, cache = {}, {}, {}
    # optional manual review of re-reads that came back empty/garbled: {"clip|text_sha16": [class, type]}
    rp = work / "text_review.json"
    review = {k: v for k, v in json.loads(rp.read_text()).items() if "|" in k} if rp.exists() else {}
    for clip in clips_of(root):
        inp = root / "inputs" / f"{clip}.mp4"
        res[clip] = {}
        per_cfg = {}
        need = set()
        for cfg in CONFIGS:
            au = audit_of(root, clip, cfg)
            if au is None:
                continue
            tr = []
            for d in au["detections"]:
                if d.get("channel") != "text":
                    continue
                acts = d.get("action") if isinstance(d.get("action"), list) else [str(d.get("action"))]
                kept = any("kept_visible" in x for x in acts)
                bx = [b for b in (d.get("boxes") or []) if len(b) > 5 and b[5] == "det"] or (d.get("boxes") or [])
                bs = [bx[int(i * (len(bx) - 1) / 2)] for i in range(3)] if bx else []
                tr.append({"id": d["id"], "kept": kept, "label": d.get("pii_label"), "sha": d.get("text_sha16"),
                           "boxes": bs})
                if not kept:
                    need.update(int(b[0]) for b in bs)
            per_cfg[cfg] = tr
        frames = read_frames(inp, need)
        for cfg, tr in per_cfg.items():
            rows = []
            for t in tr:
                if t["kept"]:
                    rows.append({"kept_visible": True, "label": t["label"]})
                    continue
                txt = ""
                for j, b in enumerate(t["boxes"]):
                    if int(b[0]) not in frames:
                        continue
                    fr = frames[int(b[0])]
                    x1, y1, x2, y2 = [int(v) for v in b[1:5]]
                    ph, pw = int(0.6 * (y2 - y1)) + 4, int(0.15 * (x2 - x1)) + 4
                    crop = fr[max(0, y1 - ph):y2 + ph, max(0, x1 - pw):x2 + pw]
                    if crop.size:
                        if crop.shape[0] < 64:
                            crop = cv2.resize(crop, None, fx=64 / crop.shape[0], fy=64 / crop.shape[0],
                                              interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(str(work / "text_crops" / f"{clip}_{cfg}_{t['id']}_{j}.jpg"), crop)
                        r = ocr(crop)
                        cand = " ".join(r.txts) if r.txts else ""
                        if len(norm(cand)) > len(norm(txt)):
                            txt = cand
                key = t["sha"] or txt
                if f"{clip}|{t['sha']}" in review:
                    cls = tuple(review[f"{clip}|{t['sha']}"])
                elif key in cache:
                    cls = cache[key]
                elif norm(txt) and any(planted_match(txt, v) for v in screen.get(clip, [])):
                    cls = ("planted", "planted " + (t["label"] or "").lower())
                elif any(fuzzy_in(b, [txt]) for b in benign.get(clip, [])):
                    cls = ("benign_planted", "planted benign line")
                elif not norm(txt):
                    cls = ("unreadable", "no text re-read")
                else:
                    l, ty = llm_classify(txt)
                    cls = ({"A": "personal", "B": "not_personal"}.get(l, "unclear"), ty)
                cache[key] = cls
                rows.append({"kept_visible": False, "label": t["label"], "class": cls[0], "type": cls[1]})
                raw.setdefault(clip, {}).setdefault(cfg, []).append([t["id"], t["label"], txt, *cls])
            res[clip][cfg] = rows
        print(clip, {c: sum(not r.get("kept_visible") for r in v) for c, v in res[clip].items()}, flush=True)
    res = _write_stage(a, work / "text.json", res, False)
    raw = _write_stage(a, work / "text_raw_PRIVATE.json", raw, False)
    (work / "text.json").write_text(json.dumps(res, indent=0))
    (work / "text_raw_PRIVATE.json").write_text(json.dumps(raw, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------ speech
def stage_speech(a):
    import difflib
    import tempfile
    import imageio_ffmpeg
    from faster_whisper import WhisperModel
    root, work = Path(a.root), Path(a.work)
    _, speech, _ = planted_values()
    wm = WhisperModel("small.en", device="cuda", compute_type="float16")

    def words(path):
        with tempfile.TemporaryDirectory() as td:
            w = os.path.join(td, "a.wav")
            r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(path),
                                "-ac", "1", "-ar", "16000", w])
            if r.returncode != 0 or not os.path.exists(w):
                return []
            segs, _ = wm.transcribe(w, word_timestamps=True, beam_size=5, temperature=0.0)
            return [(x.word.strip(), x.start, x.end) for s in segs for x in (s.words or [])]

    nw = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    res, raw = {}, {}
    for clip in clips_of(root):
        inp = root / "inputs" / f"{clip}.mp4"
        ow = [(w, s, e) for w, s, e in words(inp) if nw(w)]
        toks = {t for v in speech.get(clip, []) for t in v}
        res[clip] = {"n_words_original": len(ow), "configs": {}}
        raw[clip] = {"original": " ".join(w for w, *_ in ow)}
        for cfg in CONFIGS:
            au = audit_of(root, clip, cfg)
            if au is None:
                continue
            evs = [d for d in au["detections"] if d.get("channel") == "audio"]
            ev_rows = []
            spans = []
            for d in evs:
                t0, t1 = float(d["t_start"]), float(d["t_end"])
                spans.append((t0, t1))
                ws = [w for w, s, e in ow if e > t0 - 0.05 and s < t1 + 0.05]
                txt = " ".join(ws)
                if toks and any(t in nw(txt) for t in toks):
                    cls = ("planted", "planted " + (d.get("pii_label") or "").lower())
                elif not nw(txt):
                    cls = ("no_words", "no original words in span")
                else:
                    l, ty = llm_classify(txt)
                    cls = ({"A": "personal", "B": "not_personal"}.get(l, "unclear"), ty)
                ev_rows.append({"label": d.get("pii_label"), "class": cls[0], "type": cls[1],
                                "dur_s": round(t1 - t0, 2)})
                raw[clip].setdefault(cfg, []).append([d.get("pii_label"), txt, *cls])
            xw = [(w, s, e) for w, s, e in words(root / "outputs" / clip / f"{cfg}.mp4") if nw(w)]
            sm = difflib.SequenceMatcher(None, [nw(w) for w, *_ in ow], [nw(w) for w, *_ in xw], autojunk=False)
            inside = outside = inside_pii = 0
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == "equal":
                    continue
                for w, s, e in ow[i1:i2]:
                    if any(e > t0 - 0.3 and s < t1 + 0.3 for t0, t1 in spans):
                        inside += 1
                        inside_pii += any(t in nw(w) or nw(w) in t for t in toks if len(nw(w)) >= 3)
                    else:
                        outside += 1
            res[clip]["configs"][cfg] = {"events": ev_rows, "words_changed_in_spans": inside,
                                         "words_changed_in_spans_planted": inside_pii,
                                         "words_changed_outside_spans": outside, "n_words_output": len(xw)}
            print(clip, cfg, len(evs), "events", [r["class"] for r in ev_rows], "diff in/out", inside, outside,
                  flush=True)
    res = _write_stage(a, work / "speech.json", res, True)
    raw = _write_stage(a, work / "speech_raw_PRIVATE.json", raw, False)
    (work / "speech.json").write_text(json.dumps(res, indent=0))
    (work / "speech_raw_PRIVATE.json").write_text(json.dumps(raw, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------ assemble
def stage_assemble(a):
    root, work = Path(a.root), Path(a.work)
    F = json.loads((work / "faces.json").read_text())
    V = json.loads((work / "vlm.json").read_text())
    rv = work / "faces_review.json"   # verdicts reviewed by eye override the VLM
    review = {k: v for k, v in json.loads(rv.read_text()).items() if not k.startswith("_")} if rv.exists() else {}
    V.update(review)
    P = json.loads((work / "plates.json").read_text()) if (work / "plates.json").exists() else {}
    T = json.loads((work / "text.json").read_text())
    S = json.loads((work / "speech.json").read_text())
    CL = json.loads((root / "checklist.json").read_text())
    PR = json.loads((root / "face_privacy.json").read_text())
    _, _, benign = planted_values()
    before = {}
    bp = Path(a.before) if a.before else None
    if bp and bp.exists():   # same measurement on the outputs made BEFORE the person filter
        B = json.loads(bp.read_text())["aggregate"]
        keep = ("faces_precision", "faces_precision_ci", "faces_prec_ok", "faces_prec_n", "fp_types",
                "faces_precision_frame_weighted", "faces_real_recall", "faces_real_recall_ci",
                "faces_recall_confident", "faces_real_face_visible_in_output", "faces_n_ref",
                "faces_n_covered", "faces_miss_real_face", "faces_n_ref_confident", "faces_n_covered_confident")
        before = {c: {k: v.get(k) for k in keep} for c, v in B.items()}
    out = {"generated_by": "custodian_video/demo/web_accuracy.py", "method": METHOD,
           "before_face_fp_filter": before,
           "manual_overrides": {"vlm_verdicts": len(review)}, "clips": {}, "aggregate": {},
           "benchmark": benchmark()}
    agg = {c: {} for c in CONFIGS}

    def add(cfg, k, v):
        agg[cfg][k] = agg[cfg].get(k, 0) + v

    for clip, fr in F.items():
        refs = fr["refs"]
        conf = [r[5] >= 0.8 for r in refs]
        out["clips"][clip] = {}
        for cfg, r in fr["configs"].items():
            cov = [c >= 0.5 for c in r["coverage"]]
            miss_cls = {k: vlm_letter(V.get(k)) for k in r["misses"]}
            # map miss keys back to confident flag
            miss_conf = [conf[i] for i, c in enumerate(cov) if not c]
            n_real = sum(l == "A" for l, _ in miss_cls.values())
            n_back = sum(l == "B" for l, _ in miss_cls.values())
            n_non = sum(l == "C" for l, _ in miss_cls.values())
            n_unk = sum(l == "?" for l, _ in miss_cls.values())
            real_conf = sum(miss_cls[k][0] == "A" for k, cf in zip(r["misses"], miss_conf) if cf)
            # residual: real-face misses whose face the VLM still sees unaltered in the output
            visible = [k for k in r["misses"] if miss_cls[k][0] in "A?" and
                       vlm_letter(V.get(f"pair_{k}__{cfg}"))[0] in "A?"]
            visible_conf = sum(k in visible for k, cf in zip(r["misses"], miss_conf) if cf)
            ncov, ncov_conf = sum(cov), sum(c for c, cf in zip(cov, conf) if cf)
            prec = [vlm_letter(V.get(k)) for k in r["prec_sample"]]
            p_ok = sum(l in "AB" for l, _ in prec if l != "?")
            p_n = sum(l != "?" for l, _ in prec)
            wts = r.get("prec_weight") or [1.0] * len(prec)
            w_ok = sum(w for (l, _), w in zip(prec, wts) if l in "AB")
            w_n = sum(w for (l, _), w in zip(prec, wts) if l != "?")
            fp_types = {}
            for l, w in prec:
                if l == "C":
                    c_ = fp_category(w)
                    fp_types[c_] = fp_types.get(c_, 0) + 1
            miss_by_size = {}
            for (i, c), rf in zip(enumerate(cov), refs):
                if not c:
                    s = min(rf[3] - rf[1], rf[4] - rf[2])
                    b = "<48px" if s < 48 else ("48-96px" if s < 96 else ">=96px")
                    k = r["misses"][sum(1 for x in cov[:i] if not x)]
                    if miss_cls[k][0] in "A?":
                        miss_by_size[b] = miss_by_size.get(b, 0) + 1
            fc = {
                "n_ref": len(refs), "n_covered": ncov, "recall_all": rnd(ncov / len(refs)) if refs else None,
                "n_ref_confident": sum(conf), "n_covered_confident": ncov_conf,
                "recall_confident": rnd(ncov_conf / sum(conf)) if sum(conf) else None,
                "n_miss": len(r["misses"]), "miss_real_face": n_real, "miss_real_face_confident": real_conf,
                "miss_back_of_head": n_back, "miss_not_face": n_non, "miss_unjudged": n_unk,
                "real_face_recall": rnd(ncov / (ncov + n_real + n_unk)) if (ncov + n_real + n_unk) else None,
                "real_face_recall_ci": wilson(ncov, ncov + n_real + n_unk),
                "real_face_miss_by_size": miss_by_size,
                "real_face_visible_in_output": len(visible), "real_face_visible_in_output_confident": visible_conf,
                "real_face_recall_residual": rnd(1 - len(visible) / (ncov + n_real + n_unk))
                if (ncov + n_real + n_unk) else None,
                "prec_n": p_n, "prec_ok": p_ok, "prec_face": sum(l == "A" for l, _ in prec),
                "prec_back_of_head": sum(l == "B" for l, _ in prec),
                "precision": rnd(p_ok / p_n) if p_n else None, "precision_ci": wilson(p_ok, p_n),
                "precision_frame_weighted": rnd(w_ok / w_n) if w_n else None,
                "fp_types": fp_types, "n_face_tracks": r["n_face_tracks"],
            }
            au = audit_of(root, clip, cfg) or {}
            ff_ = au.get("face_fp_filter")
            if ff_:   # person-context filter: face tracks left unmasked (counts only)
                fc["fp_filter_unmasked_tracks"] = ff_.get("n_unmasked", 0)
                fc["fp_filter_tracks_considered"] = ff_.get("n_tracks", 0)
                add(cfg, "faces_fp_filter_unmasked_tracks", fc["fp_filter_unmasked_tracks"])
                add(cfg, "faces_fp_filter_tracks_considered", fc["fp_filter_tracks_considered"])
            for k in ("n_ref", "n_covered", "n_ref_confident", "n_covered_confident", "miss_real_face",
                      "miss_real_face_confident", "miss_back_of_head", "miss_not_face", "miss_unjudged",
                      "prec_n", "prec_ok", "real_face_visible_in_output",
                      "real_face_visible_in_output_confident"):
                add(cfg, "faces_" + k, fc[k])
            add(cfg, "faces_prec_w_ok", w_ok); add(cfg, "faces_prec_w_n", w_n)
            for t, n in fp_types.items():
                agg[cfg].setdefault("fp_types", {})
                agg[cfg]["fp_types"][t] = agg[cfg]["fp_types"].get(t, 0) + n
            for b, n in miss_by_size.items():
                agg[cfg].setdefault("real_face_miss_by_size", {})
                agg[cfg]["real_face_miss_by_size"][b] = agg[cfg]["real_face_miss_by_size"].get(b, 0) + n
            row = {"faces": fc}
            # identity (privacy_report.py)
            pr = PR.get(clip, {}).get(cfg)
            if pr and pr.get("n_ref"):
                row["identity"] = {k: pr.get(k) for k in (
                    "n_ref", "sface_reid_masked", "arc_reid_masked", "sface_reid_confident", "arc_reid_confident",
                    "impostor_sface", "impostor_arc", "n_impostor_pairs", "sface_max_masked")}
                nm = round(pr["n_ref"] * pr.get("coverage", 1))
                for k in ("sface_reid_masked", "arc_reid_masked"):
                    if pr.get(k) is not None:
                        add(cfg, "id_" + k + "_num", pr[k] * nm)
                        add(cfg, "id_" + k + "_den", nm)
                for k in ("impostor_sface", "impostor_arc"):
                    if pr.get(k) is not None:
                        add(cfg, "id_" + k + "_num", pr[k] * pr["n_impostor_pairs"])
                        add(cfg, "id_" + k + "_den", pr["n_impostor_pairs"])
            # plates
            pl = P.get(clip, {}).get("configs", {}).get(cfg)
            if pl is not None:
                pc = [c >= 0.5 for c in pl["coverage"]]
                pm = [vlm_letter(V.get(k), ) for k in pl["misses"]]
                real = sum(l == "A" for l, _ in pm) + sum(l == "?" for l, _ in pm)
                n = len(pc)
                row["plates"] = {"n_ref": n, "n_covered": sum(pc), "recall": rnd(sum(pc) / n) if n else None,
                                 "recall_ci": wilson(sum(pc), n), "miss_real_plate": real,
                                 "miss_not_plate": sum(l == "B" for l, _ in pm),
                                 "real_plate_recall": rnd(sum(pc) / (sum(pc) + real)) if n else None,
                                 "real_plate_recall_ci": wilson(sum(pc), sum(pc) + real),
                                 "n_plate_tracks": pl["n_plate_tracks"]}
                add(cfg, "plates_n_ref", n); add(cfg, "plates_n_covered", sum(pc)); add(cfg, "plates_miss_real", real)
            # text
            tr = T.get(clip, {}).get(cfg, [])
            masked = [t for t in tr if not t.get("kept_visible")]
            cls = {}
            for t in masked:
                cls[t["class"]] = cls.get(t["class"], 0) + 1
            nonpii_types = {}
            for t in masked:
                if t["class"] in ("not_personal", "benign_planted"):
                    nonpii_types[t["type"]] = nonpii_types.get(t["type"], 0) + 1
            ck = CL.get(clip, {}).get(cfg)
            scr = [i for i in ck["items"] if i["channel"] == "screen"] if ck else []
            spk = [i for i in ck["items"] if i["channel"] == "speech"] if ck else []
            ben = ck.get("benign_lines_readable") if ck else None
            row["text"] = {"masked_tracks": len(masked), "kept_visible_scene_gate": len(tr) - len(masked),
                           "masked_by_class": cls, "masked_not_pii_types": nonpii_types,
                           "planted_screen_caught": sum(i["caught"] for i in scr), "planted_screen_n": len(scr),
                           "benign_lines_readable": ben}
            add(cfg, "text_masked", len(masked))
            add(cfg, "text_masked_pii", cls.get("planted", 0) + cls.get("personal", 0))
            add(cfg, "text_masked_not_pii", cls.get("not_personal", 0) + cls.get("benign_planted", 0))
            add(cfg, "text_masked_unclear", cls.get("unclear", 0) + cls.get("unreadable", 0))
            add(cfg, "screen_caught", row["text"]["planted_screen_caught"]); add(cfg, "screen_n", len(scr))
            if ben:
                k_, n_ = (int(x) for x in ben.split("/"))
                add(cfg, "benign_readable", k_); add(cfg, "benign_n", n_)
            # speech
            sp = S.get(clip, {}).get("configs", {}).get(cfg)
            if sp is not None:
                ev = sp["events"]
                ecls = {}
                for e in ev:
                    ecls[e["class"]] = ecls.get(e["class"], 0) + 1
                false = ecls.get("not_personal", 0) + ecls.get("no_words", 0)
                row["speech"] = {"events": len(ev), "events_by_class": ecls, "false_events": false,
                                 "false_event_types": sorted({"ordinary words" if e["class"] == "not_personal"
                                                              else "no words in span" for e in ev
                                                              if e["class"] in ("not_personal", "no_words")}),
                                 "planted_speech_caught": sum(i["caught"] for i in spk), "planted_speech_n": len(spk),
                                 "words_changed_in_spans": sp["words_changed_in_spans"],
                                 "words_changed_in_spans_planted": sp["words_changed_in_spans_planted"],
                                 "words_changed_outside_spans": sp["words_changed_outside_spans"],
                                 "n_words_original": S[clip]["n_words_original"]}
                add(cfg, "speech_events", len(ev)); add(cfg, "speech_false_events", false)
                add(cfg, "speech_caught", row["speech"]["planted_speech_caught"]); add(cfg, "speech_n", len(spk))
                add(cfg, "speech_words_in_spans_nonplanted",
                    sp["words_changed_in_spans"] - sp["words_changed_in_spans_planted"])
                add(cfg, "speech_words_original", S[clip]["n_words_original"])
                add(cfg, "speech_words_outside", sp["words_changed_outside_spans"])
            # who produced the replacement values (guardian-replace runs; counts only).
            # A value = one planted item (clip, PII type; screen and speech together, since
            # the planted items of a clip have distinct types), or for anything else one
            # format-free value (value_key16).  Planted vs other comes from the text and
            # speech stages (same detection order as the audit).
            if (au.get("settings") or {}).get("pii_backend") == "guardian-replace":
                trows = iter(T.get(clip, {}).get(cfg, []))
                srows = iter(((S.get(clip, {}).get("configs", {}).get(cfg)) or {}).get("events", []))
                vals, chans = {}, {}
                for d in au.get("detections", []):
                    if d.get("channel") not in ("text", "audio"):
                        continue
                    row_ = next(trows if d["channel"] == "text" else srows, {}) or {}
                    acts = " ".join(d.get("action") or []) if isinstance(d.get("action"), list) else str(d.get("action"))
                    if "surrogate" not in acts:
                        continue
                    src_ = "guardian" if d.get("surrogate_source") == "guardian" else "local"
                    planted = row_.get("class") == "planted"
                    key = ("planted", d.get("pii_label")) if planted else ("other", d.get("value_key16"))
                    vals.setdefault(key, set()).add(src_)
                    ch = "screen" if d["channel"] == "text" else "speech"
                    chans.setdefault((ch, key), set()).add(src_)
                    if src_ == "local" and d.get("surrogate_note"):
                        agg[cfg].setdefault("replace_fallback_reasons", {})
                        fr = agg[cfg]["replace_fallback_reasons"]
                        fr[d["surrogate_note"]] = fr.get(d["surrogate_note"], 0) + 1
                rp = {}
                for ch in ("screen", "speech"):
                    for kind in ("planted", "other"):
                        ss = [v for (c_, k_), v in chans.items() if c_ == ch and k_[0] == kind]
                        rp[f"{ch}_{kind}_guardian"] = sum(v == {"guardian"} for v in ss)
                        rp[f"{ch}_{kind}_local"] = len(ss) - rp[f"{ch}_{kind}_guardian"]
                for kind in ("planted", "other"):
                    ss = [v for k_, v in vals.items() if k_[0] == kind]
                    rp[f"{kind}_values_guardian"] = sum(v == {"guardian"} for v in ss)
                    rp[f"{kind}_values_local"] = len(ss) - rp[f"{kind}_values_guardian"]
                row["replacements"] = rp
                for k, v in rp.items():
                    add(cfg, "replace_" + k, v)
            out["clips"][clip][cfg] = row
    for cfg, g in agg.items():
        if not g:
            continue
        nc, nr, nu = g.get("faces_n_covered", 0), g.get("faces_miss_real_face", 0), g.get("faces_miss_unjudged", 0)
        A = {
            "clips": len(F), **{k: v for k, v in g.items() if not k.startswith("id_")},
            "faces_recall_all": rnd(nc / g["faces_n_ref"]) if g.get("faces_n_ref") else None,
            "faces_recall_confident": rnd(g["faces_n_covered_confident"] / g["faces_n_ref_confident"])
            if g.get("faces_n_ref_confident") else None,
            "faces_real_recall": rnd(nc / (nc + nr + nu)) if nc + nr + nu else None,
            "faces_real_recall_ci": wilson(nc, nc + nr + nu),
            "faces_real_recall_residual": rnd(1 - g.get("faces_real_face_visible_in_output", 0) / (nc + nr + nu))
            if nc + nr + nu else None,
            "faces_real_recall_residual_ci": wilson(nc + nr + nu - g.get("faces_real_face_visible_in_output", 0),
                                                    nc + nr + nu),
            "faces_precision": rnd(g["faces_prec_ok"] / g["faces_prec_n"]) if g.get("faces_prec_n") else None,
            "faces_precision_ci": wilson(g.get("faces_prec_ok", 0), g.get("faces_prec_n", 0)),
            "faces_precision_frame_weighted": rnd(g["faces_prec_w_ok"] / g["faces_prec_w_n"])
            if g.get("faces_prec_w_n") else None,
            "plates_real_recall": rnd(g["plates_n_covered"] / (g["plates_n_covered"] + g["plates_miss_real"]))
            if g.get("plates_n_ref") else None,
            "text_precision": rnd(g["text_masked_pii"] / (g["text_masked_pii"] + g["text_masked_not_pii"]))
            if g.get("text_masked_pii", 0) + g.get("text_masked_not_pii", 0) else None,
            "text_precision_ci": wilson(g.get("text_masked_pii", 0),
                                        g.get("text_masked_pii", 0) + g.get("text_masked_not_pii", 0)),
        }
        for kind in ("planted", "other"):
            n_ = g.get(f"replace_{kind}_values_guardian", 0) + g.get(f"replace_{kind}_values_local", 0)
            if n_:
                A[f"replace_{kind}_values_guardian_share"] = rnd(g[f"replace_{kind}_values_guardian"] / n_)
        for k in ("sface_reid_masked", "arc_reid_masked", "impostor_sface", "impostor_arc"):
            d = g.get(f"id_{k}_den")
            A[k] = rnd(g[f"id_{k}_num"] / d) if d else None
        out["aggregate"][cfg] = A
    txt = json.dumps(out, indent=1, ensure_ascii=False)
    sys.path.insert(0, str(HERE))
    from build_web_assets import PII_LIKE
    hits = PII_LIKE.findall(txt)
    assert not hits, f"PII-like strings in accuracy.json: {set(hits)}"
    Path(a.out).write_text(txt)
    print(json.dumps(out["aggregate"], indent=1))


FP_CATS = [("animal", r"dog|animal|horse|bird(?! feeder)|creature|cat\b"),
           ("vehicle part", r"car|wheel|grille|headlight|bumper|dashboard|steering|plate|hood|vehicle"),
           ("hand / body / clothing", r"hand|finger|wrist|arm\b|cloth|shoulder|fabric|backpack|strap|helmet|torso|shirt"),
           ("plant", r"plant|tree|foliage|flower|vegetation|branch|leaf|petal"),
           ("building / sign / statue / picture", r"build|wall|window|door|clock|tower|statue|carving|text|logo|sign|cross|"
                                                  r"illustration|frame|light|poster|book|pole|stand|curtain|faucet|"
                                                  r"refrigerator|appliance|laptop|screen|feeder|water"),
           ("blurry / unclear", r"blur|unclear|shape|pattern|dark|texture|object|line|fragment|part|chair|meat|splash")]


def fp_category(w: str) -> str:
    for cat, rx in FP_CATS:
        if re.search(rx, w or ""):
            return cat
    return "other"


def rnd(x):
    return None if x is None else round(float(x), 4)


def benchmark():
    """Large-scale masking coverage / leak / re-ID from the 481-clip evaluation."""
    R = HERE.parent / "eval" / "results"
    out = {}
    for tag in ("v0.2", "v0.2-gate32"):
        p = R / tag / "summary.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows = {}
        for arm in ("redact_blur", "redact_box", "transform"):
            for sub in ("all faces", "faces >=64px"):
                v = d["privacy"].get(f"{arm}|all|{sub}")
                if not v:
                    continue
                rows[f"{arm}|{sub}"] = {k: (round(v[k], 2) if isinstance(v.get(k), float) else v.get(k))
                                        for k in ("n_clips", "n_faces", "covered", "leak", "reid_same",
                                                  "reid_same_sf", "impostor", "rank1_100", "detected")} | \
                    {k + "_ci": [round(x, 2) for x in v[k + "_ci"]] for k in
                     ("covered", "leak", "reid_same", "reid_same_sf", "impostor", "rank1_100", "detected")
                     if v.get(k + "_ci")}
            idn = d.get("identification", {}).get(f"{arm}|detector attacker")
            if idn:
                rows[f"{arm}|crossvid"] = {"rank1_clip": round(idn["rank1_clip"], 2),
                                           "rank1_clip_ci": [round(x, 2) for x in idn["rank1_clip_ci"]],
                                           "n_probes": idn["n_probes"]}
        out[tag] = {"n_clips": d.get("n_clips"), "thr_arc": round(d["thresholds"]["thr_arc_far1e-3"], 3),
                    "rows": rows}
    return out


METHOD = {
    "faces_recall": "Reference faces: YuNet (an independent detector) on the original clip, every 2nd frame, "
                    "faces >= 24 px; 'all' = score >= 0.6, 'confident' = score >= 0.8. A reference face counts as "
                    "masked when >= 50% of its box lies inside the output's masked face regions (audit log).",
    "faces_real_recall": "Every reference face left unmasked is cropped from the original and judged by a local "
                         "VLM (Qwen3-VL-8B): real face / head from behind / not a face. Real-face recall = masked / "
                         "(masked + misses judged a real face). Masked references are not VLM-checked, so this "
                         "is slightly optimistic if YuNet also fired on non-faces that happened to be masked. "
                         "Residual check: for every real-face miss the VLM also compares the original crop with the "
                         "output at the same place; a face counts as LEFT VISIBLE only if it is still recognisably "
                         "there, unaltered (the 50% box rule misses faces whose mask is smaller than YuNet's box).",
    "faces_precision": "Up to 50 masked face regions per clip and configuration, sampled across tracks, cropped "
                       "from the original and judged by Qwen3-VL-8B. Precision = share that show a human face or "
                       "head; 95% Wilson interval. Sampling is even across tracks, so short false-positive tracks "
                       "count as much as long true ones; the frame-weighted figure re-weights each sample by its "
                       "track's length (share of masked face-frames that cover a head).",
    "plates": "Reference plates: open-image-models YOLOv9-t plate detector (MIT; not used by the pipeline, which "
              "uses EgoBlur) on the original, every 2nd frame, confidence >= 0.35; masked when >= 50% of the "
              "plate box is inside a masked plate region. Misses are checked by the VLM (plate / not a plate).",
    "text": "Planted values: OCR (RapidOCR) of every 5th output frame; caught = not readable in any of them. "
            "Masked text regions: re-read by OCR from the original (3 frames per track) and matched to the "
            "planted values or classified by a local LLM (qwen2.5:7b) as personal data or not; re-reads that "
            "came back empty were classified by eye from the crops. Only counts and types are published.",
    "speech": "Planted values: Whisper small.en re-transcript of the output; caught = not in it. False bleeps: "
              "each bleep/replacement span in the audit is matched to the original's Whisper words and "
              "classified as planted PII, other personal data, or not personal data.",
    "replacements": "Blur + Custodian Guardian Layer only. Each replaced value is attributed to Guardian Layer when "
                    "every replacement of it (screen and speech) used Guardian Layer's value, otherwise to the local "
                    "fallback (audit log surrogate_source). Planted values: the 9 planted items (one per clip and PII "
                    "type, screen and speech counted once); other values: replaced text that is not a planted item "
                    "(the same regions are replaced in Blur + fake text/speech), counted once per distinct value.",
    "identity": "SFace (the pipeline's own recogniser) and ArcFace-R50 (evaluation only) compare the original "
                "face with the output pixels at the same place; re-ID = similarity above threshold (SFace 0.363, "
                "ArcFace 0.217 = FAR 1e-3). Impostor floor: the same test between two different people.",
}


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "_vlm_worker":
        return _vlm_worker(sys.argv[2], sys.argv[3])
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["faces", "plates", "vlm", "text", "speech", "assemble"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--out", default=str(HERE.parents[1] / "web" / "video" / "accuracy.json"))
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--min-face", type=int, default=24)
    ap.add_argument("--n-prec", type=int, default=50)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="faces/plates/text/speech: measure only these configurations and merge them into the "
                         "existing stage files (the other configurations' numbers are kept as measured)")
    ap.add_argument("--before", default=None,
                    help="accuracy.json measured before the face fp filter (assemble: before/after block)")
    a = ap.parse_args()
    a.work = a.work or str(Path(a.root) / "accuracy_work")
    if a.configs and a.stage != "assemble":
        global CONFIGS
        CONFIGS = [c for c in CONFIGS if c in a.configs]
    Path(a.work).mkdir(parents=True, exist_ok=True)
    {"faces": stage_faces, "plates": stage_plates, "vlm": stage_vlm, "text": stage_text,
     "speech": stage_speech, "assemble": stage_assemble}[a.stage](a)


if __name__ == "__main__":
    main()
