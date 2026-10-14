"""Offline tracking + temporal densification.

Privacy masking favours RECALL: a face that the detector misses for 3 frames
must still be masked. Because we process files (not live streams) we can run
two passes: detect everything, link detections into tracks, then

  * interpolate boxes/landmarks across short detector gaps,
  * pad each track a few frames before its first / after its last detection,
  * smooth coordinates (centered moving average) to kill jitter/flicker,
  * dilate boxes by a safety margin.

Greedy IoU association (with a centre-distance fallback for small, fast
objects) is enough here; ByteTrack/OC-SORT (MIT) can be swapped in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Det:
    frame: int
    box: np.ndarray            # x1, y1, x2, y2 (float)
    score: float
    label: str
    lm: np.ndarray | None = None   # (5, 2) landmarks for faces
    meta: dict = field(default_factory=dict)


@dataclass
class Track:
    tid: int
    label: str
    dets: dict = field(default_factory=dict)     # frame -> Det
    dense: dict = field(default_factory=dict)    # frame -> (box, lm, src)

    @property
    def first(self):
        return min(self.dets)

    @property
    def last(self):
        return max(self.dets)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _center_close(a, b, gap=1, k=0.6, min_px=20.0) -> bool:
    """Centre-distance fallback for small / fast objects (IoU ~ 0); the
    tolerance grows with the number of frames since the track was last seen."""
    ca, cb = (a[:2] + a[2:]) / 2, (b[:2] + b[2:]) / 2
    diag = np.hypot(*(a[2:] - a[:2]))
    size_ok = 0.5 < diag / max(1e-3, np.hypot(*(b[2:] - b[:2]))) < 2.0
    return size_ok and np.hypot(*(ca - cb)) < max(k * diag, min_px) * (1 + 0.5 * (gap - 1))


def link(dets_by_frame: dict[int, list[Det]], iou_thr: float = 0.2, max_gap: int = 10,
         key=None, start_id: int = 0) -> list[Track]:
    """Greedy frame-by-frame association. `key(det)` (optional) must match for
    two detections to be linked (e.g. same OCR text for on-screen text)."""
    tracks: list[Track] = []
    active: list[Track] = []
    nid = start_id
    for f in sorted(dets_by_frame):
        dets = dets_by_frame[f]
        active = [t for t in active if f - t.last <= max_gap]
        pairs = []
        for ti, t in enumerate(active):
            lb = t.dets[t.last].box
            for di, d in enumerate(dets):
                if d.label != t.label:
                    continue
                if key is not None and key(d) != key(t.dets[t.last]):
                    continue
                s = iou(lb, d.box)
                if s < iou_thr and not _center_close(lb, d.box, f - t.last):
                    continue
                pairs.append((s + 1e-3, ti, di))
        used_t, used_d = set(), set()
        for s, ti, di in sorted(pairs, reverse=True):
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            active[ti].dets[f] = dets[di]
        for di, d in enumerate(dets):
            if di not in used_d:
                t = Track(nid, d.label, {f: d}); nid += 1
                tracks.append(t); active.append(t)
    return tracks


def _smooth(arr: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(arr) < 3:
        return arr
    win = min(win, len(arr) if len(arr) % 2 else len(arr) - 1)
    pad = win // 2
    p = np.pad(arr, ((pad, pad),) + ((0, 0),) * (arr.ndim - 1), mode="edge")
    ker = np.ones(win) / win
    out = np.empty_like(arr)
    flat = p.reshape(len(p), -1)
    o = out.reshape(len(arr), -1)
    for j in range(flat.shape[1]):
        o[:, j] = np.convolve(flat[:, j], ker, mode="valid")
    return out


def densify(t: Track, n_frames: int, max_interp_gap: int = 15, pad_before: int = 3,
            pad_after: int = 3, smooth_win: int = 5, dilate: float = 1.2,
            frame_wh: tuple[int, int] | None = None, small_px: float = 0.0,
            small_extra: float = 0.0) -> Track:
    """Interpolate / pad / smooth / dilate a track to every frame.

    `small_px` / `small_extra` (faces): boxes whose short side is below
    `small_px` get up to `small_extra` more dilation (linear, full at <= 0.4 x
    small_px): small-face detections are loose and a small face's hair / head
    outline still identifies it, so the mask must reach further."""
    fs = sorted(t.dets)
    segs, cur = [], [fs[0]]
    for f in fs[1:]:
        if f - cur[-1] <= max_interp_gap:
            cur.append(f)
        else:
            segs.append(cur); cur = [f]
    segs.append(cur)
    dense = {}
    for seg in segs:
        frames = list(range(max(0, seg[0] - pad_before), min(n_frames, seg[-1] + pad_after + 1)))
        obs_f = np.array(seg)
        boxes = np.stack([t.dets[f].box for f in seg])
        has_lm = all(t.dets[f].lm is not None for f in seg)
        lms = np.stack([t.dets[f].lm.reshape(-1) for f in seg]) if has_lm else None
        fa = np.array(frames)
        bi = np.stack([np.interp(fa, obs_f, boxes[:, j]) for j in range(4)], 1)
        li = np.stack([np.interp(fa, obs_f, lms[:, j]) for j in range(lms.shape[1])], 1) \
            if has_lm else None
        braw = bi.copy()
        bi = _smooth(bi, smooth_win)
        if li is not None:
            li = _smooth(li, smooth_win)

        def grow(b):
            c = (b[:, :2] + b[:, 2:]) / 2
            hw = (b[:, 2:] - b[:, :2]) / 2 * dilate
            if small_px > 0 and small_extra > 0:
                side = (b[:, 2:] - b[:, :2]).min(1)
                hw *= (1 + small_extra * np.clip((small_px - side) / (0.6 * small_px), 0, 1))[:, None]
            return np.concatenate([c - hw, c + hw], 1)
        bi = grow(bi)
        if small_px > 0:
            # fail closed (faces): the mask box is the envelope of the smoothed box and
            # the unsmoothed one, so temporal smoothing never makes the mask lag a
            # fast-moving face
            br = grow(braw)
            bi = np.concatenate([np.minimum(bi[:, :2], br[:, :2]), np.maximum(bi[:, 2:], br[:, 2:])], 1)
        if frame_wh:
            bi[:, [0, 2]] = bi[:, [0, 2]].clip(0, frame_wh[0] - 1)
            bi[:, [1, 3]] = bi[:, [1, 3]].clip(0, frame_wh[1] - 1)
        obs = set(seg)
        for k, f in enumerate(frames):
            src = "det" if f in obs else ("interp" if seg[0] <= f <= seg[-1] else "pad")
            dense[f] = (bi[k], None if li is None else li[k].reshape(-1, 2), src)
    t.dense = dense
    return t
