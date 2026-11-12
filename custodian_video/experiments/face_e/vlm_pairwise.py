"""No-reference naturalness judgement: local Qwen3-VL-8B-Instruct (Apache-2.0)
pairwise "which face looks like a natural, unedited video frame?".

For --n random frames (fixed seed) where the largest face-track box is >= 48 px,
the same face close-up is cut from output A and output B; the pair is shown in a
random order (fixed seed) and the model answers 1 or 2.  Reports the win rate of
B over A, ties/invalid answers and the position bias (how often "1" was chosen).

  python vlm_pairwise.py --a v1=a.mp4 --b v2=b.mp4 --audit b.audit.json --n 20 --json-out x.json
  (runs the model in a subprocess under $CUSTODIAN_VLM_PYTHON)
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROMPT = ("Image 1 and Image 2 are close-ups of a person's face cut from two video frames. "
          "One or both may have been digitally edited (face replaced, pasted or generated). "
          "Which image looks more like a natural, unedited video frame of a real person? "
          "Look at the face, its edges, skin texture, lighting and how it fits the head and hair. "
          "Answer with one digit only: 1 or 2.")


def crops(a, b, audit, n, seed, dst, k=1.9, size=448):
    import cv2
    import numpy as np
    boxes = {}
    for d in json.loads(Path(audit).read_text())["detections"]:
        if d["channel"] == "faces":
            for bx in d["boxes"]:
                boxes.setdefault(bx[0], []).append(bx[1:5])
    cand = sorted(f for f, bs in boxes.items() if max(min(x[2] - x[0], x[3] - x[1]) for x in bs) >= 48)
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(cand, size=min(n, len(cand)), replace=False).tolist())
    items = []
    ca, cb = cv2.VideoCapture(a), cv2.VideoCapture(b)
    for f in pick:
        ims = []
        for c in (ca, cb):
            c.set(cv2.CAP_PROP_POS_FRAMES, f)
            ims.append(c.read()[1])
        bx = max(boxes[f], key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        H, W = ims[0].shape[:2]
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        s = max(bx[2] - bx[0], bx[3] - bx[1]) * k / 2
        x1, y1, x2, y2 = int(max(0, cx - s)), int(max(0, cy - s)), int(min(W, cx + s)), int(min(H, cy + s))
        swap = bool(rng.integers(2))
        paths = []
        for j, im in enumerate(ims):
            p = os.path.join(dst, f"f{f}_{'ab'[j]}.jpg")
            cv2.imwrite(p, cv2.resize(im[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_CUBIC),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(p)
        first, second = (paths[1], paths[0]) if swap else (paths[0], paths[1])
        items.append({"id": f"f{f}", "frame": f, "img1": first, "img2": second, "b_is": 1 if swap else 2})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--audit", required=True); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--json-out")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--keep", help="copy the crops here")
    a = ap.parse_args()
    na, pa = a.a.split("=", 1); nb, pb = a.b.split("=", 1)
    py = os.environ.get("CUSTODIAN_VLM_PYTHON", sys.executable)
    with tempfile.TemporaryDirectory(prefix="vlmpair_") as d:
        items = crops(pa, pb, a.audit, a.n, a.seed, d)
        fin, fout = os.path.join(d, "in.json"), os.path.join(d, "out.json")
        json.dump({"model": a.model, "prompt": PROMPT, "items": items}, open(fin, "w"))
        r = subprocess.run([py, os.path.abspath(__file__), "--worker", fin, fout], capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr[-2000:])
        ans = json.load(open(fout))
        if a.keep:
            import shutil
            Path(a.keep).mkdir(parents=True, exist_ok=True)
            for it in items:
                for k in ("img1", "img2"):
                    shutil.copy(it[k], Path(a.keep) / f"{it['id']}_{k}_{'B' if it['b_is'] == int(k[-1]) else 'A'}.jpg")
    wins = loss = bad = first = 0
    per = []
    for it in items:
        s = (ans.get(it["id"]) or "").strip()
        dgt = next((c for c in s if c in "12"), None)
        if dgt is None:
            bad += 1; per.append((it["frame"], None)); continue
        first += dgt == "1"
        win = int(dgt) == it["b_is"]
        wins += win; loss += not win
        per.append((it["frame"], nb if win else na))
    n = wins + loss
    res = {"A": na, "B": nb, "n": len(items), "valid": n, f"{nb}_preferred": wins, f"{na}_preferred": loss,
           f"{nb}_win_rate": round(wins / max(n, 1), 3), "invalid": bad,
           "position_bias_chose_image1": round(first / max(n, 1), 3), "per_frame": per}
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame"}))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=1))


def _worker(fin, fout):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    req = json.load(open(fin))
    model = AutoModelForImageTextToText.from_pretrained(req["model"], dtype=torch.bfloat16, device_map="cuda")
    proc = AutoProcessor.from_pretrained(req["model"])
    res = {}
    for it in req["items"]:
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "Image 1:"}, {"type": "image", "image": Image.open(it["img1"]).convert("RGB")},
            {"type": "text", "text": "Image 2:"}, {"type": "image", "image": Image.open(it["img2"]).convert("RGB")},
            {"type": "text", "text": req["prompt"]}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=4, do_sample=False)
        res[it["id"]] = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    json.dump(res, open(fout, "w"))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        _worker(sys.argv[2], sys.argv[3])
    else:
        main()
