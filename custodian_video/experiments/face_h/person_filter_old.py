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
  (d) none of its boxes lies within `edge_frac` of the frame border (person
      boxes are truncated there, so a missing person box is no evidence);
  (e) it has at least `min_det` detected frames, so one missed person
      detection can never decide alone -- or, for a single-frame track, a
      confident animal box (COCO bird ... giraffe) contains its centre;
  (f) second look: on the exact detected frames the full-frame person
      detector AND a zoomed run (crop of `zoom_ctx` x the box, upscaled) find
      no person whose head region touches the box.
The person detector must have run on every frame the decision relies on.

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


def prescreen(det_boxes: dict, persons: dict, frame_wh, max_score: float, recall_net: bool,
              p: FilterParams) -> tuple[bool, str]:
    """Rules (a)-(d) on pass-1 data. -> (candidate_for_unmasking, reason).
    `det_boxes`: frame -> mask box at DETECTED frames; `persons`: frame -> (N, 5)
    person boxes for every frame the detector ran on."""
    if not det_boxes:
        return False, "no_detections"
    if recall_net:
        return False, "yunet_recall_net"
    if max_score >= p.max_score:
        return False, "score>=%.2f" % p.max_score
    W, H = frame_wh
    ex, ey = p.edge_frac * W, p.edge_frac * H
    for b in det_boxes.values():
        if b[0] <= ex or b[1] <= ey or b[2] >= W - 1 - ex or b[3] >= H - 1 - ey:
            return False, "near_frame_edge"
    for f, b in det_boxes.items():
        near = [g for g in range(f - p.every, f + p.every + 1) if g in persons]
        if not near:
            return False, "no_person_pass"   # detector did not run here: fail closed
        for g in near:
            if _touches(b, head_regions(persons[g], p)):
                return False, "person_head_region"
    return True, "candidate"


def look_frames(det_boxes: dict, p: FilterParams) -> list[int]:
    fs = sorted(det_boxes)
    if len(fs) <= p.max_look_frames:
        return fs
    return [fs[int(round(i * (len(fs) - 1) / (p.max_look_frames - 1)))] for i in range(p.max_look_frames)]


def second_look(det_boxes: dict, frames: dict, detector: PersonDetector, p: FilterParams) -> tuple[bool, str]:
    """Rules (e)-(f) on the exact detected frames (`frames`: frame -> BGR image).
    -> (unmask, reason)."""
    fs = look_frames(det_boxes, p)
    if any(f not in frames for f in fs):
        return False, "frame_unavailable"
    single = len(det_boxes) < p.min_det
    explained = False
    for f in fs:
        b = det_boxes[f]
        persons, animals = detector.detect(frames[f])
        if _touches(b, head_regions(persons, p)):
            return False, "person_head_region(exact_frame)"
        if _touches(b, head_regions(detector.zoom_persons(frames[f], b, p), p)):
            return False, "person_head_region(zoomed)"
        if single:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            a = animals[animals[:, 4] >= p.animal_threshold] if len(animals) else animals
            explained = explained or bool(len(a) and np.any((a[:, 0] <= cx) & (cx <= a[:, 2]) &
                                                            (a[:, 1] <= cy) & (cy <= a[:, 3])))
    if single and not explained:
        return False, "single_detection_unexplained"
    return True, ("no_person_head;score<%.2f;no_recall_net;off_edge;%s" %
                  (p.max_score, "single_frame_animal" if single else ">=%d_frames" % p.min_det))
