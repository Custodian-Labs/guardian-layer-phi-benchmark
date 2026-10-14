"""Pixel operations: redact styles (blur / pixelate / box) and on-screen-text
surrogate rendering (inpaint + re-typeset a same-type fake value)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _roi(frame, box, pad=0):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    return x1, y1, x2, y2


def redact_box(frame: np.ndarray, box, style: str = "blur", shape: str = "ellipse") -> None:
    """In-place redaction of an axis-aligned box (ellipse for faces)."""
    x1, y1, x2, y2 = _roi(frame, box)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = frame[y1:y2, x1:x2]
    bw, bh = x2 - x1, y2 - y1
    if style == "blur":
        k = max(3, min(bw, bh) | 1)
        # coarse mosaic (<= 4 cells across) + heavy Gaussian: plain light blur of
        # faces is partly invertible and small blurred faces stay re-identifiable
        cells = max(1, min(4, max(bw, bh) // 16))
        small = cv2.resize(roi, (max(1, cells * bw // max(bw, bh)), max(1, cells * bh // max(bw, bh))),
                           interpolation=cv2.INTER_AREA)
        rep = cv2.GaussianBlur(cv2.resize(small, (bw, bh), interpolation=cv2.INTER_LINEAR), (k, k), 0)
    elif style == "pixelate":
        # >= 5 px blocks and at most ~8 blocks across: tiny faces must not stay recognisable
        n = max(2, min(8, max(bw, bh) // 5))
        small = cv2.resize(roi, (max(1, bw * n // max(bw, bh)), max(1, bh * n // max(bw, bh))),
                           interpolation=cv2.INTER_AREA)
        rep = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
    else:  # solid box
        rep = np.zeros_like(roi)
    if shape == "ellipse":
        m = np.zeros((bh, bw), np.uint8)
        cv2.ellipse(m, (bw // 2, bh // 2), (bw // 2, bh // 2), 0, 0, 360, 255, -1)
        roi[m > 0] = rep[m > 0]
    else:
        roi[:] = rep


def redact_quad(frame: np.ndarray, quad: np.ndarray, style: str = "box") -> None:
    x1, y1 = quad.min(0)
    x2, y2 = quad.max(0)
    X1, Y1, X2, Y2 = _roi(frame, (x1, y1, x2, y2), pad=2)
    if X2 - X1 < 2 or Y2 - Y1 < 2:
        return
    m = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(m, [quad.astype(np.int32)], 255)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    tmp = frame.copy()
    redact_box(tmp, (X1, Y1, X2, Y2), style=style, shape="rect")
    frame[m > 0] = tmp[m > 0]


@lru_cache(maxsize=4)
def _font_path() -> str | None:
    for p in ["/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/liberation/LiberationSans-Regular.ttf"]:
        if Path(p).exists():
            return p
    return None


def render_surrogate_text(frame: np.ndarray, quad: np.ndarray, text: str) -> None:
    """Replace the text inside `quad` with `text`: estimate background/ink
    colours, paint background, typeset surrogate, warp into the quad."""
    quad = quad.astype(np.float32)
    w = int(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
    h = int(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
    if w < 4 or h < 4:
        return
    # rectify the original patch to estimate colours
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(quad, dst)
    patch = cv2.warpPerspective(frame, M, (w, h)).reshape(-1, 3).astype(np.float32)
    border = np.concatenate([patch.reshape(h, w, 3)[[0, -1]].reshape(-1, 3),
                             patch.reshape(h, w, 3)[:, [0, -1]].reshape(-1, 3)])
    bg = np.median(border, 0)
    d = np.linalg.norm(patch - bg, axis=1)
    ink = np.median(patch[d >= np.quantile(d, 0.85)], 0) if len(d) else 255 - bg
    # typeset (supersample x3 for anti-aliasing)
    S = 3
    img = Image.new("RGB", (w * S, h * S), tuple(int(c) for c in bg[::-1]))
    dr = ImageDraw.Draw(img)
    fp = _font_path()
    size = int(h * S * 0.95)
    while True:
        font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default(size=size)
        l, t, r, b = dr.textbbox((0, 0), text, font=font)
        if (r - l) <= w * S * 0.98 or size < 8:
            break
        size = int(size * 0.92)
    dr.text(((w * S - (r - l)) / 2 - l, (h * S - (b - t)) / 2 - t), text, font=font,
            fill=tuple(int(c) for c in ink[::-1]))
    rendered = cv2.resize(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), (w, h),
                          interpolation=cv2.INTER_AREA)
    Minv = cv2.getPerspectiveTransform(dst, quad)
    H, W = frame.shape[:2]
    warped = cv2.warpPerspective(rendered, Minv, (W, H))
    m = cv2.warpPerspective(np.full((h, w), 255, np.uint8), Minv, (W, H))
    frame[m > 127] = warped[m > 127]
