"""Scene-context gate for on-screen text PII (local VLM).

The OCR + PII detector cannot tell a name on a patient form from the name of a
shop on a storefront.  Before an on-screen text track is masked, a local VLM
looks at the full frame (text region outlined in red) plus a close-up and says
WHERE the text is.  Only an explicit "public signage" answer (shop sign,
storefront, billboard/advert, street/traffic sign, building lettering, livery
on a vehicle) or "not real text" (clock face, window pattern misread by OCR)
lets the text stay visible; anything else, including a VLM
failure, keeps the mask (privacy favours recall).  One decision per text
track.

Model: Qwen3-VL-8B-Instruct (Apache-2.0) via transformers, from the local HF
cache.  It needs transformers>=4.57, which the custodian-video env does not
have (diffusers/SD pin), so the model runs in a worker subprocess under the
interpreter given by --vlm-python / $CUSTODIAN_VLM_PYTHON (default: this
interpreter).  The worker is this same file run as a script; it only imports
torch, transformers and PIL.  Nothing leaves the machine.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

PROMPT = """Image 1 is a full video frame. Image 2 is a close-up of the same place.
The text inside the RED box reads: "{line}".
Where is this text physically written or displayed? Answer with ONE letter only.
A) a paper document, form, letter, card, receipt or label in the scene
B) an electronic display (computer, phone, tablet, TV or monitor screen), or a caption/graphic overlaid on the video
C) an ID badge, name tag, uniform name patch or lanyard
D) a whiteboard, blackboard, notice board or handwritten note
E) a vehicle licence plate
F) public signage: shop sign, storefront or shop-window lettering/sticker, restaurant sign, billboard, poster or advert, street or traffic sign, building lettering, or advertising/branding printed on a vehicle
G) not real text: a clock face, window, roof, decoration, texture or other pattern misread as text
H) something else, or cannot tell"""

CATEGORIES = {"A": "document", "B": "screen_or_overlay", "C": "badge", "D": "whiteboard_or_note",
              "E": "licence_plate", "F": "public_signage", "G": "not_text(ocr_noise)", "H": "other_or_unclear"}
UNMASK = {"F", "G"}


class SceneContextGate:
    def __init__(self, model: str = "Qwen/Qwen3-VL-8B-Instruct", python: str | None = None,
                 timeout: float = 1800.0):
        self.model = model
        self.python = python or os.environ.get("CUSTODIAN_VLM_PYTHON") or sys.executable
        self.timeout = timeout
        self.name = f"{model} (Apache-2.0) scene-context gate, local"
        self.stats = {"queries": 0, "unmasked_signage_or_noise": 0, "kept_masked": 0, "errors": 0, "seconds": 0.0}

    @staticmethod
    def _render(frame, quad, line, d, key):
        import cv2
        import numpy as np
        H, W = frame.shape[:2]
        q = np.asarray(quad, np.float32)
        x1, y1 = q.min(0)
        x2, y2 = q.max(0)
        full = frame.copy()
        th = max(2, int(round(max(H, W) / 500)))
        cv2.polylines(full, [q.astype(np.int32)], True, (0, 0, 255), th)
        s = min(1.0, 1280 / max(H, W))
        full_s = cv2.resize(full, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        bw, bh = x2 - x1, y2 - y1
        cw, ch = max(4 * bw, 320), max(8 * bh, 240)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        X1, Y1 = int(max(0, cx - cw / 2)), int(max(0, cy - ch / 2))
        X2, Y2 = int(min(W, cx + cw / 2)), int(min(H, cy + ch / 2))
        crop = full[Y1:Y2, X1:X2]
        z = min(3.0, 640 / max(1, max(crop.shape[:2])))
        crop = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA)
        pf, pc = os.path.join(d, f"{key}_full.jpg"), os.path.join(d, f"{key}_crop.jpg")
        cv2.imwrite(pf, full_s, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(pc, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return {"id": key, "full": pf, "crop": pc, "line": line}

    def classify(self, items: list[tuple[str, object, object, str]]) -> dict[str, dict]:
        """items: (key, frame_bgr, quad (4x2), ocr_line) -> {key: {category, mask, answer}}"""
        out = {}
        if not items:
            return out
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="cvg_vlm_") as d:
            reqs = [self._render(fr, q, line, d, k) for k, fr, q, line in items]
            fin, fout = os.path.join(d, "in.json"), os.path.join(d, "out.json")
            with open(fin, "w", encoding="utf-8") as f:
                json.dump({"model": self.model, "prompt": PROMPT, "items": reqs}, f)
            try:
                r = subprocess.run([self.python, os.path.abspath(__file__), "--worker", fin, fout],
                                   capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=self.timeout,
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                if r.returncode != 0:
                    raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "worker failed")
                with open(fout, encoding="utf-8") as f:
                    ans = json.load(f)
            except Exception as e:
                print(f"[scene-gate] VLM worker failed ({type(e).__name__}: {str(e)[:200]}); "
                      f"keeping every text mask")
                self.stats["errors"] += 1
                ans = {}
        for k, *_ in items:
            a = (ans.get(k) or "").strip()
            letter = next((c for c in a.upper() if c in CATEGORIES), "H") if a else None
            mask = letter not in UNMASK
            out[k] = {"category": CATEGORIES.get(letter, "vlm_error") if letter else "vlm_error",
                      "mask": mask, "answer": a[:20]}
            self.stats["queries"] += 1
            self.stats["kept_masked" if mask else "unmasked_signage_or_noise"] += 1
        self.stats["seconds"] = round(self.stats["seconds"] + time.time() - t0, 2)
        return out


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
            {"type": "image", "image": Image.open(it["full"]).convert("RGB")},
            {"type": "image", "image": Image.open(it["crop"]).convert("RGB")},
            {"type": "text", "text": req["prompt"].format(line=it["line"].replace('"', "'"))}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=6, do_sample=False)
        res[it["id"]] = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    json.dump(res, open(fout, "w", encoding="utf-8"))


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--worker":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _worker(sys.argv[2], sys.argv[3])
