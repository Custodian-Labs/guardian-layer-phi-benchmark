"""Face / hair / accessory parsing for surrogate paste-back masks (experimental,
used by surrogate_face_v3.py only).

Model: MediaPipe "selfie multiclass 256x256" image segmenter (Google, Apache-2.0
model card; classes background, hair, body-skin, face-skin, clothes, others =
accessories).  On a LivePortrait-style head crop it separates face skin from
hair at the hairline, and its "others" class picks up glasses frames and
earrings.  It runs in a persistent worker subprocess under $CUSTODIAN_MP_PYTHON
(a Python with `mediapipe`; the custodian-video env does not have it).
Weights: $CUSTODIAN_FACE_PARSE_MODEL (default
experiments/face_f/models/selfie_multiclass_256x256.tflite), from
https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
"""
from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np

_DEF_MODEL = Path(__file__).resolve().parents[1] / "experiments/face_f/models/selfie_multiclass_256x256.tflite"


class FaceParser:
    """parse(list of BGR images) -> float32 (n, 256, 256, 3): face-skin, hair, others."""

    def __init__(self, python: str | None = None, model: str | None = None):
        self.python = python or os.environ.get("CUSTODIAN_MP_PYTHON")
        self.model = model or os.environ.get("CUSTODIAN_FACE_PARSE_MODEL", str(_DEF_MODEL))
        if not self.python or not Path(self.python).exists():
            raise FileNotFoundError("CUSTODIAN_MP_PYTHON (a python with mediapipe) is not set")
        if not Path(self.model).exists():
            raise FileNotFoundError(self.model)
        self.p = subprocess.Popen([self.python, str(Path(__file__).with_name("face_parse_worker.py")),
                                   self.model], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
        self.n = 0

    def parse(self, ims, batch: int = 64) -> np.ndarray:
        res = []
        for s in range(0, len(ims), batch):
            chunk = [cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA) if im.shape[:2] != (256, 256)
                     else im for im in ims[s:s + batch]]
            a = np.ascontiguousarray(np.stack(chunk).astype(np.uint8))
            self.p.stdin.write(struct.pack("<I", len(chunk)) + a.tobytes())
            self.p.stdin.flush()
            need = a.nbytes
            buf = bytearray()
            while len(buf) < need:
                b = self.p.stdout.read(need - len(buf))
                if not b:
                    raise RuntimeError("face parser worker died")
                buf += b
            res.append(np.frombuffer(bytes(buf), np.uint8).reshape(a.shape).astype(np.float32) / 255)
            self.n += len(chunk)
        return np.concatenate(res) if res else np.zeros((0, 256, 256, 3), np.float32)

    def close(self):
        if self.p and self.p.poll() is None:
            try:
                self.p.stdin.write(struct.pack("<I", 0)); self.p.stdin.flush()
                self.p.wait(timeout=10)
            except Exception:
                self.p.kill()
        self.p = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
