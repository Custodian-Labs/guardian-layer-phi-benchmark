"""Coarse appearance attributes of a face crop, used ONLY to condition the
surrogate-face generator (surrogate_face.py, realism "v2") so that the new
identity keeps the person's apparent age band, gender presentation, skin tone,
hair, facial hair and glasses -- the attributes whose change makes a surrogate
look "pasted on" -- while the identity itself changes.

Model: Qwen3-VL-8B-Instruct (Apache-2.0), local, in a worker subprocess under
$CUSTODIAN_VLM_PYTHON (same arrangement as scene_context.py: it needs
transformers>=4.57, the custodian-video env pins an older one).  Nothing leaves
the machine.  The attributes are kept in memory only (not written to the audit
log); any failure returns {} and the generator falls back to a neutral prompt.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

PROMPT = """Look at the person's face in this video frame crop. Describe only coarse visible appearance, as JSON with exactly these keys:
{"age": one of ["child","teen","20s","30s","40s","50s","60s","70s+"],
 "gender": one of ["woman","man","person"],
 "skin": one of ["very light","light","medium","olive","brown","dark"],
 "hair": short phrase for hair colour and length, e.g. "shoulder-length light brown hair", or "bald",
 "facial_hair": one of ["none","stubble","moustache","beard"],
 "glasses": one of ["none","glasses","sunglasses"],
 "makeup": one of ["none","light","heavy"],
 "build": one of ["slim","average","round"],
 "appearance": one of ["East Asian","Southeast Asian","South Asian","Black","White","Middle Eastern","Latino","unclear"]}
Answer with the JSON only."""

AGES = {"child", "teen", "20s", "30s", "40s", "50s", "60s", "70s+"}


def _clean(d: dict) -> dict:
    out = {}
    if not isinstance(d, dict):
        return out
    for k in ("age", "gender", "skin", "hair", "facial_hair", "glasses", "makeup", "build", "appearance"):
        v = d.get(k)
        if isinstance(v, str) and 0 < len(v) < 60:
            out[k] = v.strip().lower()
    if out.get("age") not in AGES:
        out.pop("age", None)
    return out


VERIFY_PROMPT = """This is a portrait photo. Describe it as JSON with exactly these keys:
{"age": one of ["child","teen","20s","30s","40s","50s","60s","70s+"],
 "gender": one of ["woman","man","person"],
 "skin": one of ["very light","light","medium","olive","brown","dark"],
 "glasses": one of ["none","glasses","sunglasses"],
 "real_photo": 1-10, how much it looks like an unedited photo of a real person (10 = indistinguishable from a real photo; low for plastic / airbrushed / CGI / painted skin, distorted features, odd eyes or teeth),
 "attractive": 1-10, how pleasant and good-looking the face is to an average viewer,
 "artifacts": one of ["none","minor","major"]}
Answer with the JSON only."""

AGE_ORDER = ["child", "teen", "20s", "30s", "40s", "50s", "60s", "70s+"]
SKIN_ORDER = ["very light", "light", "medium", "olive", "brown", "dark"]
SKIN_RANK = {"very light": 0, "light": 1, "medium": 2, "olive": 2, "brown": 3, "dark": 4}


def _clean_verify(d: dict) -> dict:
    out = _clean(d)
    if isinstance(d, dict):
        for k in ("real_photo", "attractive"):
            try:
                out[k] = float(d.get(k))
            except (TypeError, ValueError):
                pass
        if str(d.get("artifacts", "")).lower() in ("none", "minor", "major"):
            out["artifacts"] = str(d["artifacts"]).lower()
    return out


def attribute_mismatch(orig: dict, cand: dict) -> list[str]:
    """Reasons a synthetic candidate does NOT match the person's apparent gender
    presentation, age band (+-1 band) and skin tone (+-1 step); [] = match.
    Attributes missing on the original side are not checked; missing on the
    candidate side count as a mismatch (unverified)."""
    bad = []
    g = orig.get("gender")
    if g in ("woman", "man") and cand.get("gender") != g:
        bad.append("gender")
    a = orig.get("age")
    if a in AGE_ORDER and (cand.get("age") not in AGE_ORDER or
                           abs(AGE_ORDER.index(cand["age"]) - AGE_ORDER.index(a)) > 1):
        bad.append("age")
    s = orig.get("skin")
    if s in SKIN_RANK and (cand.get("skin") not in SKIN_RANK or abs(SKIN_RANK[cand["skin"]] - SKIN_RANK[s]) > 1):
        bad.append("skin")
    if cand.get("artifacts") == "major":
        bad.append("artifacts")
    return bad


def quality_score(cand: dict) -> float:
    """Qwen3-VL realism + appeal ratings (0-20); minor artefacts cost 2."""
    return (cand.get("real_photo", 0.0) + cand.get("attractive", 0.0)
            - (2.0 if cand.get("artifacts") == "minor" else 0.0))


def prompt_sdxl(a: dict, features: str = "", appearance: bool = True) -> tuple[str, str]:
    """(positive, negative) photographic portrait prompt for an SDXL photoreal
    model (RealVisXL) from the coarse attributes: same age band, gender
    presentation, skin tone, hair, facial hair, glasses; frontal, mouth closed.
    Kept under CLIP's 77 tokens so the seed-chosen `features` are not cut off.
    RealVisXL renders unretouched faces a few years older than prompted, so the
    age text is shifted ~3-5 years down (checked with Qwen3-VL on the demo clips)."""
    age = a.get("age")
    g = a.get("gender", "person")
    g = g if g in ("woman", "man") else "person"
    noun = {"child": "child", "teen": "teenager"}.get(age, g)
    pro = {"woman": "her", "man": "his"}.get(g, "their")
    age_txt = {"20s": f"in {pro} mid twenties", "30s": "about thirty years old",
               "40s": f"in {pro} late thirties", "50s": f"in {pro} late forties",
               "60s": f"in {pro} late fifties", "70s+": f"in {pro} late sixties"}.get(age, "")
    app = a.get("appearance", "")
    app = app.title() if (appearance and app in APPEARANCE
                          and os.environ.get("CUSTODIAN_FACE_ATTR_APPEARANCE", "1") != "0") else ""
    looks = {"woman": "beautiful", "man": "handsome"}.get(g, "good-looking")
    parts = [f"RAW photo, portrait of a {looks} {app} {noun} {age_txt}".replace("  ", " ")]
    if a.get("skin") in SKIN:
        parts.append(SKIN[a["skin"]])
    if a.get("hair"):
        parts.append(a["hair"] + " tied back")
    if a.get("facial_hair") in ("stubble", "moustache", "beard"):
        parts.append({"stubble": "light stubble", "moustache": "moustache", "beard": "neat beard"}[a["facial_hair"]])
    # no glasses in the portrait even if the person wears them: the swap re-renders the
    # original's glasses region from the frame, and glasses in the source portrait pull the
    # identity embedding towards the original's (more guard trips on the clip-1 glasses wearer)
    if features:
        parts.append(features)
    parts += ["facing camera, gentle friendly expression, mouth closed, clear forehead",
              "soft studio light, grey background, natural skin texture, sharp focus"]
    # kept short: CLIP truncates the negative prompt at 77 tokens as well
    neg = ("bindi, forehead mark, glasses, cartoon, 3d render, plastic skin, airbrushed, heavy make-up, "
           "deformed, asymmetric eyes, open mouth, teeth, frown, profile, hands, hat, bangs, text, "
           "black and white, blurry")
    if g == "woman":
        neg += ", man, beard"
    elif g == "man":
        neg += ", woman"
    if age in ("child", "teen", "20s", "30s"):
        neg += ", old, wrinkles, grey hair"
    elif age == "40s":
        neg += ", elderly, deep wrinkles"
    elif age in ("50s", "60s", "70s+"):
        neg += ", teenager"
    return ", ".join(p for p in parts if p), neg


PAIR_PROMPT = """Two portrait photos of different people: A (left) and B (right). Which one looks more like a natural, attractive, high-quality photo of a real person (natural skin, pleasant face, friendly look, no artefacts, not plastic or CGI)? Answer with one letter: A or B."""


class FaceAttributes:
    def __init__(self, model: str = "Qwen/Qwen3-VL-8B-Instruct", python: str | None = None,
                 timeout: float = 1200.0):
        self.model = model
        self.python = python or os.environ.get("CUSTODIAN_VLM_PYTHON") or sys.executable
        self.timeout = timeout
        self.name = f"{model} (Apache-2.0) face attributes, local"
        self.seconds = 0.0
        self.used_model = None

    def __call__(self, crops: dict) -> dict:
        """crops: {key: BGR image} -> {key: attribute dict} ({} on any failure)."""
        return self.ask(crops, PROMPT, _clean)

    def verify(self, crops: dict) -> dict:
        """Synthetic candidate portraits -> {key: attributes + quality ratings} ({} on failure)."""
        return self.ask(crops, VERIFY_PROMPT, _clean_verify)

    def compare(self, pairs: dict) -> dict:
        """{key: side-by-side image (A left, B right)} -> {key: "A" | "B" | None}."""
        res = self.ask(pairs, PAIR_PROMPT, None)
        out = {}
        for k, v in res.items():
            v = (v or "").strip().upper()
            out[k] = v[0] if v[:1] in ("A", "B") else None
        return out

    def ask(self, crops: dict, prompt: str, clean) -> dict:
        import cv2
        if not crops:
            return {}
        t0 = time.time()
        res = {}
        with tempfile.TemporaryDirectory(prefix="cvg_attr_") as d:
            items = []
            for k, im in crops.items():
                p = os.path.join(d, f"{k}.jpg")
                cv2.imwrite(p, im, [cv2.IMWRITE_JPEG_QUALITY, 92])
                items.append({"id": str(k), "img": p})
            fin, fout = os.path.join(d, "in.json"), os.path.join(d, "out.json")
            ans = {}
            # shared GPUs: fall back to the smaller Qwen3-VL-4B (Apache-2.0) on failure / OOM
            for model in dict.fromkeys([self.model, "Qwen/Qwen3-VL-4B-Instruct"]):
                json.dump({"model": model, "prompt": prompt, "items": items}, open(fin, "w"))
                try:
                    r = subprocess.run([self.python, os.path.abspath(__file__), "--worker", fin, fout],
                                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                                       timeout=self.timeout, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                    if r.returncode != 0:
                        raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "worker failed")
                    ans = json.load(open(fout, encoding="utf-8"))
                    self.used_model = model
                    break
                except Exception as e:
                    print(f"[face-attributes] VLM worker failed with {model} ({type(e).__name__}: "
                          f"{str(e)[:160]})", flush=True)
            if not ans:
                print("[face-attributes] no attributes; using a neutral prompt", flush=True)
        for k in crops:
            a = ans.get(str(k)) or ""
            if clean is None:      # free-text answer
                res[k] = a
                continue
            try:
                s = a[a.index("{"):a.rindex("}") + 1]
                res[k] = clean(json.loads(s))
            except Exception:
                res[k] = {}
        self.seconds += time.time() - t0
        return res


SKIN = {"very light": "very fair pale skin", "light": "fair light skin", "medium": "light-medium skin",
        "olive": "olive skin", "brown": "brown skin", "dark": "dark skin"}


APPEARANCE = {"east asian", "southeast asian", "south asian", "black", "white", "middle eastern", "latino"}


def prompt_from_attributes(a: dict) -> tuple[str, str]:
    """(positive, negative) prompt for the SD generator: the same coarse
    attributes as the original (age band, gender presentation, skin tone, hair,
    facial hair, glasses), natural unretouched skin, no beautification."""
    age = a.get("age")
    g = a.get("gender", "person")
    g = g if g in ("woman", "man") else "person"
    noun = {"child": "child", "teen": "teenager"}.get(age, g)
    pro = {"woman": "her", "man": "his"}.get(g, "their")
    # SD-1.5 renders unretouched, make-up-free faces about a decade older than
    # prompted (checked on the demo clips), so the prompt is shifted ~5-10 years down
    age_txt = {"20s": f"in {pro} twenties", "30s": f"in {pro} late twenties",
               "40s": f"in {pro} mid thirties", "50s": f"in {pro} mid forties",
               "60s": f"in {pro} mid fifties", "70s+": f"in {pro} late sixties"}.get(age, "")
    app = a.get("appearance", "")
    app = app.title() if app in APPEARANCE and os.environ.get("CUSTODIAN_FACE_ATTR_APPEARANCE", "1") != "0" else ""
    young = age in ("20s", "30s")
    parts = [f"photo of a {'young ' if young else ''}{app} {noun} {age_txt}".replace("  ", " ").strip()]
    if a.get("skin") in SKIN:
        parts.append(SKIN[a["skin"]])
    if a.get("hair"):
        parts.append(a["hair"])
    if a.get("facial_hair") in ("stubble", "moustache", "beard"):
        parts.append(a["facial_hair"])
    if a.get("glasses") in ("glasses", "sunglasses"):
        parts.append(f"wearing {a['glasses']}")
    if a.get("build") == "round":
        parts.append("full round face")
    if age in ("50s", "60s", "70s+"):
        parts.append("natural age lines")
    parts += ["relaxed pleasant expression, mouth closed", "youthful skin" if young else "",
              "sharp modern colour photo, natural skin"]
    neg = ("makeup, lipstick, eyeliner, beauty filter, airbrushed, retouched, porcelain skin, glamour, "
           "fashion model, doll, plastic, cgi, 3d render, cartoon, painting, deformed, disfigured, "
           "extra eyes, text, watermark, smiling, grin, teeth, open mouth, frown, sepia, vintage, "
           "black and white, monochrome, old photo")
    if age in ("child", "teen", "20s", "30s"):
        neg += ", middle-aged, old, elderly, wrinkles, forehead wrinkles, grey hair"
    elif age == "40s":
        neg += ", elderly, aged, deep wrinkles, grey hair"
    elif age in ("50s", "60s", "70s+"):
        neg += ", teenager"
    return ", ".join(p for p in parts if p), neg


def _worker(fin: str, fout: str) -> None:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    req = json.load(open(fin, encoding="utf-8"))
    model = AutoModelForImageTextToText.from_pretrained(req["model"], dtype=torch.bfloat16,
                                                        device_map="cuda")
    proc = AutoProcessor.from_pretrained(req["model"])
    res = {}
    for it in req["items"]:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": Image.open(it["img"]).convert("RGB")},
            {"type": "text", "text": req["prompt"]}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=120, do_sample=False)
        res[it["id"]] = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    json.dump(res, open(fout, "w", encoding="utf-8"))


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--worker":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _worker(sys.argv[2], sys.argv[3])
