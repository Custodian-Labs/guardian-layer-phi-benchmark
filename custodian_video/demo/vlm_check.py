"""Local VLM-as-judge visual QA of keyframes (Qwen2.5-VL-7B-Instruct, Apache-2.0,
from the local HF cache; nothing leaves the machine). Needs transformers>=4.49.

  python demo/vlm_check.py img1.png [img2.png ...] --prompt "..."
"""
import argparse
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

DEFAULT = ("Describe what you see. For every human face: is it clearly visible and "
           "identifiable, blurred/pixelated, blacked out, or does it look like a realistic face? "
           "Note visual artifacts. Transcribe any readable text exactly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--prompt", default=DEFAULT)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max-side", type=int, default=1280)
    a = ap.parse_args()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda")
    proc = AutoProcessor.from_pretrained(a.model)
    for p in a.images:
        im = Image.open(p).convert("RGB")
        s = min(1.0, a.max_side / max(im.size))
        im = im.resize((int(im.width * s), int(im.height * s)))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": a.prompt}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
        out = model.generate(**inp, max_new_tokens=400, do_sample=False)
        ans = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        print(f"=== {p}\n{ans}\n", flush=True)


if __name__ == "__main__":
    sys.exit(main())
