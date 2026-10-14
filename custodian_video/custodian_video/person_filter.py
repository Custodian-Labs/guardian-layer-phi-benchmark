"""Person-context filter for face false positives (`--face-fp-filter person`).

EgoBlur at its recall-first threshold also fires on dogs, plants, statues,
façade ornaments, signs, hands and van grilles.  This filter lets a face TRACK
go unmasked only when every independent signal says it is not a person's head.
It fails closed: any doubt, missing data or error keeps the mask.

A face track may be left unmasked only if ALL of these hold:
  (a) none of its detected boxes (after the usual mask dilation) touches the
      head region of any person box found within +-`every` frames, at any
      detected frame of the track.  Head region = the top `head_frac` of the
      person box, widened by `margin_x` of its width on each side and raised
      by `margin_up` of its height (generous on purpose);
  (b) the YuNet recall net never contributed a detection to it;
  (c) its maximum EgoBlur score is below `max_score`;
  (d) none of its RAW detector boxes (before mask dilation) lies within
      `edge_frac` of the frame border (person boxes are truncated there, so a
      missing person box is no evidence).  Low-score tracks (max score <
      `low_score`, raw box >= `precise_min_side`) may touch the TOP border: a person cut off at the top still
      has a body in the frame, so the person detector (and its head region,
      which starts at the top of the box) still sees them;
  (e) it has at least `min_det` detected frames, so one missed person
      detection can never decide alone -- or, for a single-frame track, a
      confident animal box (COCO bird ... giraffe) contains its centre;
  (f) second look: on the exact detected frames the full-frame person
      detector AND a zoomed run (crop of `zoom_ctx` x the box, upscaled) find
      no person whose head region touches the box.
The person detector must have run on every frame the decision relies on.

Precise head region (low-score tracks only, max score < `low_score`, raw box
short side >= `precise_min_side`; the same tracks may touch the top border): the
generous head region of a person box (a) covers far more than the head (the
whole upper background beside a seated person).  For these tracks a touch
counts only if the box centre lies in the head region or at least
`min_overlap` of the box area does (grazing the widened margin is not enough;
the zoomed second look still applies).  A counted touch is EXCUSED only if
that person is explained by exactly one CONFIDENT face of another track on the
same frame (EgoBlur score >= `conf_face`, short side >= `conf_min_side` px,
centre inside that person's head region), the candidate's centre lies outside
that face's head estimate (face box widened by `head_side` of its width each
side and raised by `head_up` of its height), and the candidate is at least
`min_rel_size` of that face's size (a much smaller face next to a big one could
be someone further back).  The confident face itself stays masked by its own
track.  Zero, two or more confident faces in the person's head region, or a
person without one (seen from behind, face missed) -> the touch counts and the
track stays masked.

Person detector: torchvision Faster R-CNN ResNet-50-FPN v2 with the COCO_V1
weights (person and animal classes only).  Code: BSD-3-Clause (torchvision).
Weights: trained on COCO 2017 (annotations CC BY 4.0; images are Flickr photos
under their own licences); torchvision leaves checking the weights' terms to
the user.  No Ultralytics / AGPL code.  The VLM is never consulted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COCO_PERSON = 1
COCO_ANIMALS = set(range(16, 26))   # bird cat dog horse sheep cow elephant bear zebra giraffe


@dataclass
class FilterParams:
    max_score: float = 0.5         # (c) tracks scoring at/above this always stay masked
    head_frac: float = 0.4         # (a) top 40 % of a person box is "head region"
    margin_x: float = 0.25         # (a) widened by 25 % of the person width each side
    margin_up: float = 0.15        # (a) raised by 15 % of the person height
    edge_frac: float = 0.03        # (d) boxes within 3 % of the frame border stay masked
    person_threshold: float = 0.3  # person boxes used at this score (low = more keeps)
    every: int = 2                 # person detector cadence in pass 1 (frames)
    min_det: int = 2               # (e) detected frames needed to unmask
    animal_threshold: float = 0.5  # (e) animal box that may explain a single-frame track
    zoom_ctx: float = 6.0          # (f) zoomed crop = 6 x the box (at least 256 px) ...
    zoom_side: int = 800           # (f) ... resized to 800 px on its long side
    max_look_frames: int = 24      # (f) detected frames re-checked per candidate (evenly spaced)
    low_score: float = 0.4         # tracks below this: precise head region + top border allowed
    conf_face: float = 0.9         # a "confident" face (explains its person's head)
    conf_min_side: float = 96.0    # ... with a short side of at least this many px
    head_side: float = 0.25        # head estimate = confident face box widened by 25 % each side
    head_up: float = 0.35          # ... and raised by 35 % of its height
    min_rel_size: float = 0.5      # candidate long side >= 50 % of the confident face's
    min_overlap: float = 0.4       # low-score: a head-region touch counts if >= 40 % of the box (or its centre) is inside
    precise_min_side: float = 64.0 # the precise-head-region rules apply only to raw boxes >= this (px):
                                   # small faces = distant people, whose person boxes are unreliable


class PersonDetector:
    name = ("torchvision Faster R-CNN R50-FPN v2, COCO_V1 weights, person (+animal) classes "
            "(code BSD-3-Clause; COCO-trained weights)")

    def __init__(self, device: str = "cuda", threshold: float = 0.3):
        import torch
        from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                                  fasterrcnn_resnet50_fpn_v2)
        self.torch = torch
        self.dev = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
        self.m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
                                            box_score_thresh=min(0.05, threshold)).to(self.dev).eval()
        self.threshold = threshold

    def detect(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (persons, animals): (N, 5) float32 arrays x1, y1, x2, y2, score."""
        torch = self.torch
        t = torch.from_numpy(np.ascontiguousarray(frame_bgr[:, :, ::-1].transpose(2, 0, 1))).to(self.dev)
        with torch.no_grad():
            o = self.m([t.float() / 255.0])[0]
        b = torch.cat([o["boxes"], o["scores"][:, None]], 1).float().cpu().numpy().astype(np.float32)
        lab = o["labels"].cpu().numpy()
        persons = b[(lab == COCO_PERSON) & (b[:, 4] >= self.threshold)]
        animals = b[np.isin(lab, list(COCO_ANIMALS))]
        return persons, animals

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        return self.detect(frame_bgr)[0]

    def zoom_persons(self, frame_bgr: np.ndarray, box, p: "FilterParams") -> np.ndarray:
        """Persons found in an upscaled crop around `box`, in frame coordinates."""
        import cv2
        H, W = frame_bgr.shape[:2]
        s = max(p.zoom_ctx * max(box[2] - box[0], box[3] - box[1]), 256)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        X1, Y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        X2, Y2 = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
        c = frame_bgr[Y1:Y2, X1:X2]
        if c.size == 0:
            return np.zeros((0, 5), np.float32)
        z = p.zoom_side / max(c.shape[:2])
        c = cv2.resize(c, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA)
        q = self.detect(c)[0].copy()
        if len(q):
            q[:, :4] = q[:, :4] / z + np.array([X1, Y1, X1, Y1], np.float32)
        return q


def head_regions(persons: np.ndarray, p: FilterParams) -> np.ndarray:
    if persons is None or len(persons) == 0:
        return np.zeros((0, 4), np.float32)
    x1, y1, x2, y2 = persons[:, 0], persons[:, 1], persons[:, 2], persons[:, 3]
    w, h = x2 - x1, y2 - y1
    return np.stack([x1 - p.margin_x * w, y1 - p.margin_up * h,
                     x2 + p.margin_x * w, y1 + p.head_frac * h], 1)


def _touches(box, regs) -> bool:
    if len(regs) == 0:
        return False
    return bool(np.any((np.minimum(box[2], regs[:, 2]) > np.maximum(box[0], regs[:, 0])) &
                       (np.minimum(box[3], regs[:, 3]) > np.maximum(box[1], regs[:, 1]))))


def _explained(box, region, confs, p: FilterParams) -> bool:
    """True if the touch of `box` with one person's head `region` is explained by
    exactly one confident face of that person, away from `box` (see module doc)."""
    if confs is None or len(confs) == 0:
        return False
    c = np.asarray(confs, np.float32).reshape(-1, 4)
    cx, cy = (c[:, 0] + c[:, 2]) / 2, (c[:, 1] + c[:, 3]) / 2
    side = np.minimum(c[:, 2] - c[:, 0], c[:, 3] - c[:, 1])
    inr = (region[0] <= cx) & (cx <= region[2]) & (region[1] <= cy) & (cy <= region[3])
    cs = c[inr & (side >= p.conf_min_side)]
    if len(cs) != 1 or int(inr.sum()) != 1:
        return False
    x1, y1, x2, y2 = cs[0]
    w, h = x2 - x1, y2 - y1
    hx1, hy1, hx2, hy2 = x1 - p.head_side * w, y1 - p.head_up * h, x2 + p.head_side * w, y2
    bx, by = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if hx1 <= bx <= hx2 and hy1 <= by <= hy2:
        return False
    return max(box[2] - box[0], box[3] - box[1]) >= p.min_rel_size * max(w, h)


def _head_touch(box, cbox, persons, confs, low: bool, p: FilterParams) -> bool:
    """Does `box` touch an (unexplained) person head region?  `cbox` = the box whose
    centre / size is tested for the explanation (raw detector box when known)."""
    return _head_touch_why(box, cbox, persons, confs, low, p)[0]


def _head_touch_why(box, cbox, persons, confs, low: bool, p: FilterParams) -> tuple[bool, str]:
    """-> (touches an unexplained head region, how the touches were excused: "" | "graze" | "explained")."""
    regs = head_regions(persons, p)
    low = low and min(cbox[2] - cbox[0], cbox[3] - cbox[1]) >= p.precise_min_side
    how = set()
    for r in regs:
        if _touches(box, r[None]):
            if low and not _substantial(cbox, r, p):
                how.add("graze")   # only grazes the margin of the generous head region
                continue
            if low and _explained(cbox, r, confs, p):
                how.add("explained")
                continue
            return True, ""
    return False, "+".join(sorted(how))


def _substantial(box, r, p: FilterParams) -> bool:
    """Low-score tracks: the touch counts if the box centre lies in the head region
    or at least `min_overlap` of the box area does."""
    bx, by = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if r[0] <= bx <= r[2] and r[1] <= by <= r[3]:
        return True
    iw = max(0.0, min(box[2], r[2]) - max(box[0], r[0]))
    ih = max(0.0, min(box[3], r[3]) - max(box[1], r[1]))
    area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    return iw * ih / area >= p.min_overlap


def prescreen(det_boxes: dict, persons: dict, frame_wh, max_score: float, recall_net: bool,
              p: FilterParams, raw_boxes: dict | None = None, conf_faces: dict | None = None) -> tuple[bool, str]:
    """Rules (a)-(d) on pass-1 data. -> (candidate_for_unmasking, reason).
    `det_boxes`: frame -> mask box at DETECTED frames; `raw_boxes`: frame -> raw
    detector box (edge rule, explanation test; defaults to `det_boxes`);
    `persons`: frame -> (N, 5) person boxes for every frame the detector ran on;
    `conf_faces`: frame -> (K, 4) confident face boxes of OTHER tracks."""
    if not det_boxes:
        return False, "no_detections"
    if recall_net:
        return False, "yunet_recall_net"
    if max_score >= p.max_score:
        return False, "score>=%.2f" % p.max_score
    raw = raw_boxes or det_boxes
    low = max_score < p.low_score
    W, H = frame_wh
    ex, ey = p.edge_frac * W, p.edge_frac * H
    for f in det_boxes:
        b = raw.get(f, det_boxes[f])
        top_ok = low and min(b[2] - b[0], b[3] - b[1]) >= p.precise_min_side
        if b[0] <= ex or b[2] >= W - 1 - ex or b[3] >= H - 1 - ey or (b[1] <= ey and not top_ok):
            return False, "near_frame_edge"
    for f, b in det_boxes.items():
        near = [g for g in range(f - p.every, f + p.every + 1) if g in persons]
        if not near:
            return False, "no_person_pass"   # detector did not run here: fail closed
        cf = (conf_faces or {}).get(f)
        for g in near:
            if _head_touch(b, raw.get(f, b), persons[g], cf, low, p):
                return False, "person_head_region"
    return True, "candidate"


def look_frames(det_boxes: dict, p: FilterParams) -> list[int]:
    fs = sorted(det_boxes)
    if len(fs) <= p.max_look_frames:
        return fs
    return [fs[int(round(i * (len(fs) - 1) / (p.max_look_frames - 1)))] for i in range(p.max_look_frames)]


def second_look(det_boxes: dict, frames: dict, detector: PersonDetector, p: FilterParams,
                max_score: float = 0.0, raw_boxes: dict | None = None,
                conf_faces: dict | None = None) -> tuple[bool, str]:
    """Rules (e)-(f) on the exact detected frames (`frames`: frame -> BGR image).
    -> (unmask, reason)."""
    fs = look_frames(det_boxes, p)
    if any(f not in frames for f in fs):
        return False, "frame_unavailable"
    single = len(det_boxes) < p.min_det
    low = max_score < p.low_score and conf_faces is not None
    raw = raw_boxes or det_boxes
    explained = False
    exc = set()
    for f in fs:
        b = det_boxes[f]
        cb = raw.get(f, b)
        cf = (conf_faces or {}).get(f)
        persons, animals = detector.detect(frames[f])
        t1, h1 = _head_touch_why(b, cb, persons, cf, low, p)
        if t1:
            return False, "person_head_region(exact_frame)"
        t2, h2 = _head_touch_why(b, cb, detector.zoom_persons(frames[f], b, p), cf, low, p)
        if t2:
            return False, "person_head_region(zoomed)"
        for h in (h1 + "+" + h2).split("+"):
            if h:
                exc.add(h)
        if single:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            a = animals[animals[:, 4] >= p.animal_threshold] if len(animals) else animals
            explained = explained or bool(len(a) and np.any((a[:, 0] <= cx) & (cx <= a[:, 2]) &
                                                            (a[:, 1] <= cy) & (cy <= a[:, 3])))
    if single and not explained:
        return False, "single_detection_unexplained"
    return True, ("no_person_head;score<%.2f;no_recall_net;off_edge;%s%s" %
                  (p.max_score, "single_frame_animal" if single else ">=%d_frames" % p.min_det,
                   (";head_region_" + "+".join(sorted(exc))) if exc else ""))
