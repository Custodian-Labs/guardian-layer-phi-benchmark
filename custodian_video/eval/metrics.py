"""Stage 3: per-clip privacy and utility measurements for every arm.

  python metrics.py --tag v0.1 [--arms original,redact_blur,redact_box,transform]

Writes results/<tag>/per_clip/<arm>/<clip>.json and face embeddings to DATA/embeddings/<tag>/<arm>/<clip>.npz.
Original-video features are version-independent and cached in DATA/cache/orig_<CACHE_VERSION>/,
so a new pipeline tag only re-measures the processed arms.

Privacy (per reference face = YuNet face, score>=0.7, min side>=24 px, on the ORIGINAL frame)
  sim_same      ArcFace-R50 cosine between the original face and the output pixels at the SAME
                landmarks (attacker knows where the face is) -- thresholded in analyze.py
  sim_same_sf   same with SFace
  detected      YuNet (score>=0.6) still finds a face overlapping the original (IoU>=0.3)
  det_sim       ArcFace cosine of that re-detected face vs the original (leak if >= threshold)
  covered       the pipeline's audit log has a face mask box containing the face centre
vox2 probes additionally store per-frame embeddings for cross-video identification against a
118-identity gallery of *other* videos of the same people (analyze.py).

Utility
  k400  : VideoMAE-B (K400 fine-tuned, 16 frames) top-1/top-5 vs label; X-CLIP B/32 top-1/top-5;
          Keypoint R-CNN pose agreement (OKS vs the original video's keypoints, persons matched by IoU)
  mot17 : Faster R-CNN v2 (COCO) person AP50 vs MOT17 GT (ignore regions handled), recall@0.5,
          per-frame count MAE
  all   : face-still-detected rate (detected), i.e. downstream face analytics keep working
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common import (CACHE, CLIPS, EMB, RESULTS, FaceModels, arm_video, box_iou, load_manifest,
                    read_frames, uniform_idxs, xywh2xyxy)

METRIC_VERSION = "m2"       # bump when per-clip measurements change
CACHE_VERSION = "m1"        # bump when original-video features change
torch.set_num_threads(8)
cv2.setNumThreads(8)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
COCO_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
                        .87, .87, .89, .89]) / 10.0


# ----------------------------------------------------------------------------- models
class Models:
    def __init__(self):
        import torchvision
        from transformers import (AutoProcessor, VideoMAEForVideoClassification,
                                  VideoMAEImageProcessor, XCLIPModel)
        self.face = FaceModels(gpu=DEV == "cuda")
        self.vm_proc = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.vm = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics").to(DEV).eval().half()
        self.labels = [self.vm.config.id2label[i] for i in range(len(self.vm.config.id2label))]
        self.xc_proc = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        self.xc = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32").to(DEV).eval()
        self.xc_text = self.xc_proc(text=list(self.labels),
                                    return_tensors="pt", padding=True)
        tv = torchvision.models.detection
        self.kp = tv.keypointrcnn_resnet50_fpn(weights=tv.KeypointRCNN_ResNet50_FPN_Weights.COCO_V1).to(DEV).eval()
        self.det = tv.fasterrcnn_resnet50_fpn_v2(weights=tv.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).to(DEV).eval()

    @torch.no_grad()
    def action(self, rgb16, rgb8):
        x = self.vm_proc(list(rgb16), return_tensors="pt")["pixel_values"].to(DEV).half()
        p = self.vm(pixel_values=x).logits.float().softmax(-1)[0].cpu().numpy()
        xv = self.xc_proc(videos=list(rgb8), return_tensors="pt")["pixel_values"].to(DEV)
        out = self.xc(input_ids=self.xc_text["input_ids"].to(DEV),
                      attention_mask=self.xc_text["attention_mask"].to(DEV), pixel_values=xv)
        q = out.logits_per_video.float().softmax(-1)[0].cpu().numpy()
        return p, q

    @torch.no_grad()
    def _run(self, model, frames, bs=4):
        res = []
        for i in range(0, len(frames), bs):
            ims = [torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(DEV)
                   for f in frames[i:i + bs]]
            for o in model(ims):
                res.append({k: v.cpu().numpy() for k, v in o.items()})
        return res

    def pose(self, frames, thr=0.7):
        out = []
        for o in self._run(self.kp, frames):
            keep = (o["labels"] == 1) & (o["scores"] >= thr)
            out.append(dict(boxes=o["boxes"][keep], kps=o["keypoints"][keep][..., :2]))
        return out

    def persons(self, frames):
        out = []
        for o in self._run(self.det, frames):
            keep = (o["labels"] == 1) & (o["scores"] >= 0.05)
            out.append(dict(boxes=o["boxes"][keep], scores=o["scores"][keep]))
        return out


# ----------------------------------------------------------------------------- helpers
def frame_plan(clip):
    n = clip["n"]
    if clip["dataset"] == "mot17":
        face = list(range(0, n, 10))
        det = list(range(0, n, 5))
        return dict(face=face, det=det, act16=[], act8=[], pose=[])
    face = uniform_idxs(n, 16)
    return dict(face=face, det=[], act16=face if clip["dataset"] == "k400" else [],
                act8=uniform_idxs(n, 8) if clip["dataset"] == "k400" else [],
                pose=face if clip["dataset"] == "k400" else [])


def all_idxs(plan):
    return sorted(set().union(*[set(v) for v in plan.values()]))


def rgb(fr):
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def oks(gt_kps, dt_kps, box):
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    d2 = ((gt_kps - dt_kps) ** 2).sum(-1)
    return np.exp(-d2 / (2 * area * (2 * COCO_SIGMAS) ** 2))


def match(boxes_a, boxes_b, thr=0.5):
    """greedy IoU matching; returns list of (i, j) pairs"""
    pairs, used = [], set()
    cand = sorted(((box_iou(a, b), i, j) for i, a in enumerate(boxes_a) for j, b in enumerate(boxes_b)),
                  reverse=True)
    ua = set()
    for iou, i, j in cand:
        if iou < thr:
            break
        if i in ua or j in used:
            continue
        ua.add(i); used.add(j); pairs.append((i, j))
    return pairs


def ap50(per_frame_dets, gt, frames):
    """VOC all-point AP at IoU 0.5 with MOT17 ignore handling."""
    recs, n_pos = [], 0
    for fi, d in zip(frames, per_frame_dets):
        g = gt.get(str(fi), [])
        valid = [xywh2xyxy(r[:4]) for r in g if r[5] == 1 and r[4] == 1 and r[6] >= 0.25]
        ign = [xywh2xyxy(r[:4]) for r in g if not (r[5] == 1 and r[4] == 1 and r[6] >= 0.25)]
        n_pos += len(valid)
        taken = set()
        for k in np.argsort(-d["scores"]):
            b = d["boxes"][k]
            ious = [box_iou(b, v) for v in valid]
            j = int(np.argmax(ious)) if ious else -1
            if j >= 0 and ious[j] >= 0.5 and j not in taken:
                taken.add(j); recs.append((d["scores"][k], 1))
            elif any(box_iou(b, x) >= 0.5 for x in ign) or (j >= 0 and ious[j] >= 0.5):
                continue      # ignore region or duplicate on an ignored/occluded person
            else:
                recs.append((d["scores"][k], 0))
    if n_pos == 0:
        return None
    recs.sort(key=lambda x: -x[0])
    tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([1 - r[1] for r in recs])
    rec = tp / n_pos; prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def count_and_recall(per_frame_dets, gt, frames, thr=0.5):
    errs, tp, npos = [], 0, 0
    for fi, d in zip(frames, per_frame_dets):
        g = gt.get(str(fi), [])
        valid = [xywh2xyxy(r[:4]) for r in g if r[5] == 1 and r[4] == 1 and r[6] >= 0.25]
        boxes = d["boxes"][d["scores"] >= thr]
        errs.append(abs(len(boxes) - len(valid)))
        npos += len(valid)
        tp += len(match(valid, list(boxes), 0.5))
    return float(np.mean(errs)), (tp / npos if npos else None)


def audit_face_boxes(audit_path):
    """{frame: [xyxy, ...]} of face masks + per-track actions, defensive to schema changes."""
    try:
        a = json.loads(Path(audit_path).read_text())
    except Exception:
        return None, {}
    by_frame, actions = {}, []
    for d in a.get("detections", []):
        if d.get("channel") != "faces":
            continue
        actions.append(d.get("action", []))
        for b in d.get("boxes", []):
            by_frame.setdefault(int(b[0]), []).append([float(v) for v in b[1:5]])
    n_tr = len(actions)
    sur = sum(any(x.startswith("surrogate") and "fallback" not in x for x in acts) for acts in actions)
    fb = sum(any("fallback" in x for x in acts) for acts in actions)
    st = dict(face_tracks=n_tr, tracks_surrogate=sur, tracks_blur_fallback=fb,
              timing=a.get("timing", {}), tool=a.get("tool"))
    return by_frame, st


# ----------------------------------------------------------------------------- original features
def original_features(M: Models, clip) -> dict:
    cdir = CACHE / f"orig_{CACHE_VERSION}"
    cp = cdir / f"{clip['id']}.pkl"
    if cp.exists():
        return pickle.loads(cp.read_bytes())
    plan = frame_plan(clip)
    frs = read_frames(CLIPS / clip["dataset"] / f"{clip['id']}.mp4", all_idxs(plan))
    refs = {}
    for f in plan["face"]:
        if f not in frs:
            continue
        rows = [r for r in M.face.detect(frs[f], 0.7) if min(r[2], r[3]) >= 24]
        rows = np.array(rows, np.float32).reshape(-1, 15)
        refs[f] = dict(rows=rows, arc=M.face.arc_embed(frs[f], rows), sf=M.face.sface_embed(frs[f], rows))
    feat = dict(plan=plan, refs=refs)
    if plan["act16"]:
        feat["vm"], feat["xc"] = M.action([rgb(frs[i]) for i in plan["act16"]], [rgb(frs[i]) for i in plan["act8"]])
        feat["pose"] = M.pose([frs[i] for i in plan["pose"]])
    if plan["det"]:
        feat["persons"] = M.persons([frs[i] for i in plan["det"] if i in frs])
    cdir.mkdir(parents=True, exist_ok=True)
    cp.write_bytes(pickle.dumps(feat))
    return feat


def gallery_embeddings(M: Models, clips) -> Path:
    """Per-identity ArcFace/SFace embeddings from original gallery videos (largest face/frame)."""
    gp = CACHE / f"gallery_{CACHE_VERSION}.npz"
    if gp.exists():
        return gp
    ids, arc, sf, frame_id = [], [], [], []
    for c in [c for c in clips if c["dataset"] == "vox2_gallery"]:
        frs = read_frames(CLIPS / c["dataset"] / f"{c['id']}.mp4", uniform_idxs(c["n"], 16))
        for f, fr in frs.items():
            rows = M.face.detect(fr, 0.7)
            if len(rows) == 0:
                continue
            r = rows[np.argmax(rows[:, 2] * rows[:, 3])][None]
            ids.append(c["label"]); frame_id.append(f)
            arc.append(M.face.arc_embed(fr, r)[0]); sf.append(M.face.sface_embed(fr, r)[0])
    gp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(gp, ids=np.array(ids), arc=np.array(arc, np.float32), sf=np.array(sf, np.float32),
             frame=np.array(frame_id))
    return gp


# ----------------------------------------------------------------------------- per arm
def measure(M: Models, clip, arm, tag, out_dir: Path):
    of = original_features(M, clip)
    plan = of["plan"]
    vpath = arm_video(clip, arm, tag)
    res = dict(clip=clip["id"], dataset=clip["dataset"], arm=arm, label=clip.get("label"))
    if not vpath.exists():
        res["missing"] = True
        return res, None
    frs = read_frames(vpath, all_idxs(plan)) if arm != "original" else \
        read_frames(CLIPS / clip["dataset"] / f"{clip['id']}.mp4", all_idxs(plan))
    boxes_by_frame, st = (None, {}) if arm == "original" else audit_face_boxes(vpath.with_suffix(".audit.json"))
    res["audit"] = st

    # ---- faces
    faces, probe_same, probe_det, emb_o, emb_m = [], [], [], [], []
    for f in plan["face"]:
        ref = of["refs"].get(f)
        if ref is None or f not in frs or len(ref["rows"]) == 0:
            continue
        fr = frs[f]
        rows = ref["rows"]
        arc_m = M.face.arc_embed(fr, rows)
        sf_m = M.face.sface_embed(fr, rows)
        dets = M.face.detect(fr, 0.6)
        emb_o.append(ref["arc"]); emb_m.append(arc_m)
        for k, r in enumerate(rows):
            rb = xywh2xyxy(r[:4])
            best, bi = 0.0, -1
            for j, d in enumerate(dets):
                iou = box_iou(rb, xywh2xyxy(d[:4]))
                if iou > best:
                    best, bi = iou, j
            detected = best >= 0.3
            det_sim = float(M.face.arc_embed(fr, dets[bi:bi + 1])[0] @ ref["arc"][k]) if detected else None
            cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
            covered = None if boxes_by_frame is None else any(
                b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes_by_frame.get(f, []))
            faces.append(dict(f=f, size=float(min(r[2], r[3])), sim_same=float(arc_m[k] @ ref["arc"][k]),
                              sim_same_sf=float(sf_m[k] @ ref["sf"][k]), detected=bool(detected),
                              det_sim=det_sim, covered=covered))
        if clip["dataset"] == "vox2":
            k = int(np.argmax(rows[:, 2] * rows[:, 3]))
            probe_same.append(arc_m[k])
            rb = xywh2xyxy(rows[k][:4])
            cand = [d for d in dets if box_iou(rb, xywh2xyxy(d[:4])) >= 0.3]
            if cand:
                probe_det.append(M.face.arc_embed(fr, np.array(cand[:1]))[0])
    res["faces"] = faces

    # ---- action + pose (k400)
    if plan["act16"]:
        p, q = M.action([rgb(frs[i]) for i in plan["act16"]], [rgb(frs[i]) for i in plan["act8"]])
        lab = M.labels.index(clip["label"])
        for name, pr, pr0 in [("videomae", p, of["vm"]), ("xclip", q, of["xc"])]:
            top5 = np.argsort(-pr)[:5]
            res[name] = dict(top1=int(top5[0] == lab), top5=int(lab in top5), pred=M.labels[int(top5[0])],
                             p_label=float(pr[lab]), agree_orig=int(top5[0] == int(np.argmax(pr0))))
        # reference persons: score>=0.7 on the original; candidates in the output: score>=0.3, so
        # that score jitter around the reference threshold does not count as a lost person
        pose = M.pose([frs[i] for i in plan["pose"]], thr=0.3)
        oks_all, oks_head, n_ref, n_match = [], [], 0, 0
        for o, a in zip(of["pose"], pose):
            pairs = dict(match(list(o["boxes"]), list(a["boxes"]), 0.5))
            for i in range(len(o["boxes"])):
                n_ref += 1
                if i in pairs:
                    n_match += 1
                    s = oks(o["kps"][i], a["kps"][pairs[i]], o["boxes"][i])
                    oks_all.append(float(s.mean())); oks_head.append(float(s[:5].mean()))
                else:
                    oks_all.append(0.0); oks_head.append(0.0)
        res["pose"] = dict(n_persons=n_ref, person_recall=(n_match / n_ref if n_ref else None),
                           oks=(float(np.mean(oks_all)) if oks_all else None),
                           oks_head=(float(np.mean(oks_head)) if oks_head else None))

    # ---- persons (mot17)
    if plan["det"]:
        gt = json.loads((CLIPS / clip["gt"]).read_text())
        fl = [i for i in plan["det"] if i in frs]
        dets = M.persons([frs[i] for i in fl]) if arm != "original" else of["persons"]
        mae, rec = count_and_recall(dets, gt, fl)
        res["mot"] = dict(ap50=ap50(dets, gt, fl), recall=rec, count_mae=mae)

    npz = dict(arc_o=np.concatenate(emb_o).astype(np.float16) if emb_o else np.zeros((0, 512), np.float16),
               arc_m=np.concatenate(emb_m).astype(np.float16) if emb_m else np.zeros((0, 512), np.float16))
    if clip["dataset"] == "vox2":
        npz.update(probe_same=np.array(probe_same, np.float32).reshape(-1, 512),
                   probe_det=np.array(probe_det, np.float32).reshape(-1, 512))
    return res, npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", default="original,reencode,redact_blur,redact_box,transform")
    ap.add_argument("--datasets", default="k400,mot17,vox2")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    clips = load_manifest()
    t0 = time.time()
    M = Models()
    gallery_embeddings(M, clips)
    ds = set(a.datasets.split(","))
    todo = [c for c in clips if c["dataset"] in ds]
    for arm in a.arms.split(","):
        od = RESULTS / a.tag / "per_clip" / arm
        od.mkdir(parents=True, exist_ok=True)
        n_new = 0
        for i, c in enumerate(todo):
            jp = od / f"{c['id']}.json"
            if jp.exists() and not a.force:
                prev = json.loads(jp.read_text())
                if not prev.get("missing") and prev.get("metric_version") == METRIC_VERSION:
                    continue
            res, npz = measure(M, c, arm, a.tag, od)
            res["metric_version"] = METRIC_VERSION
            jp.write_text(json.dumps(res))
            if npz is not None:
                (EMB / a.tag / arm).mkdir(parents=True, exist_ok=True)
                np.savez(EMB / a.tag / arm / f"{c['id']}.npz", **npz)
            n_new += 1
            if n_new % 20 == 0:
                print(f"  {arm}: {i + 1}/{len(todo)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        print(f"== {arm}: {n_new} clips measured", flush=True)
    print(f"metrics done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
