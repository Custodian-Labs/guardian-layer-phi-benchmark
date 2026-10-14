"""Per-frame detectors (all local).

faces  : default **EgoBlur** (Meta, Apache-2.0 code AND weights, "research and
         commercial"), Faster R-CNN TorchScript -> boxes.  EgoBlur gives no
         landmarks, so 5-point landmarks (needed by the surrogate-face backend)
         come from YuNet run on the face crop (`--face-landmarks yunet`), or from
         a fixed template placed in the box (`template`, licence-clean but
         cruder).  `--face-detector centerface` keeps the v0.1 CenterFace path
         (MIT code+weights inside `deface`, but WIDER-FACE-derived -> research).
plates : default EgoBlur licence-plate model (Apache-2.0); `--plate-detector
         yolov9` keeps open-image-models YOLOv9 (MIT, training-data provenance
         not stated).

EgoBlur weights: Gen1 TorchScript (`ego_blur_face.jit`, `ego_blur_lp.jit`) from
the official Meta repo https://huggingface.co/projectaria/EgoBlur (revision
9c0b319, Apache-2.0).  Gen2 weights (`ego_blur_face_gen2.jit`, ...) are only
handed out after an e-mail form on projectaria.com; they are the same
Faster R-CNN TorchScript interface and can be dropped into the model directory
(EGOBLUR_DIR, default ~/.cache/custodian_video/egoblur_gen1) — the loader
prefers *_gen2.jit when present.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .tracking import Det

MODELS = Path.home() / ".cache" / "custodian_video"
EGOBLUR_DIR = Path(os.environ.get("EGOBLUR_DIR", MODELS / "egoblur_gen1"))

# ArcFace 112x112 5-point template (same order as CenterFace / YuNet:
# image-left eye, image-right eye, nose, mouth left, mouth right)
_T112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                  [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def ort_providers(device: str) -> list[str]:
    import onnxruntime as ort
    if device == "cpu":
        return ["CPUExecutionProvider"]
    try:
        ort.preload_dlls()  # CUDA/cuDNN from the nvidia-* pip wheels
    except Exception:
        pass
    av = ort.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in av]


def template_landmarks(box: np.ndarray) -> np.ndarray:
    """5 landmarks placed by a fixed template inside a (frontal) face box."""
    x1, y1, x2, y2 = box
    t = (_T112 - [8, 8]) / 96.0          # ArcFace crop ~ face box with 8px margin
    return np.stack([x1 + t[:, 0] * (x2 - x1), y1 + t[:, 1] * (y2 - y1)], 1).astype(np.float32)


class CenterFaceDetector:
    name = "centerface (deface pkg, MIT; WIDER-FACE-derived -> research only)"
    license = "MIT code+weights; trained/evaluated on WIDER FACE (non-commercial dataset)"

    def __init__(self, device: str = "cuda", threshold: float = 0.35, max_side: int = 1920):
        import onnxruntime as ort
        from deface.centerface import CenterFace
        prov = ort_providers(device)
        ort.set_default_logger_severity(3)
        self.cf = CenterFace(backend="onnxrt", override_execution_provider=prov[0])
        self.threshold, self.max_side = threshold, max_side
        self.provider = prov[0]

    def __call__(self, frame_bgr: np.ndarray, fidx: int) -> list[Det]:
        h, w = frame_bgr.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        self.cf.in_shape = (int(w * s), int(h * s))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        dets, lms = self.cf(rgb, threshold=self.threshold)
        out = []
        for d, lm in zip(dets, lms):
            out.append(Det(fidx, d[:4].astype(np.float32), float(d[4]), "face",
                           lm.reshape(5, 2).astype(np.float32)))
        return out


FaceDetector = CenterFaceDetector   # v0.1 name (kept for scripts that import it)


class _EgoBlurModel:
    """Thin wrapper around an EgoBlur TorchScript Faster R-CNN."""

    def __init__(self, kind: str, device: str):
        import torch
        import torchvision  # noqa: F401  (registers torchvision::nms used inside the TorchScript)
        cand = [EGOBLUR_DIR / f"ego_blur_{kind}_gen2.jit", EGOBLUR_DIR / f"ego_blur_{kind}.jit"]
        path = next((p for p in cand if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"EgoBlur {kind} weights not found in {EGOBLUR_DIR} (see README: download "
                f"ego_blur_{kind}.zip from huggingface.co/projectaria/EgoBlur and unzip)")
        self.path = path
        self.gen = "gen2" if "gen2" in path.name else "gen1"
        self.torch = torch
        self.dev = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
        self.kind = kind
        if self.gen == "gen2":
            self._load_gen2(path)
        else:
            self.m = torch.jit.load(str(path), map_location="cpu").to(self.dev).eval()

    # Gen2 is a detectron2 GeneralizedRCNN TorchScript whose outputs are scripted
    # `Instances`; it only runs through Meta's own predictor (pip package `egoblur`,
    # Apache-2.0, installed --no-deps; module `gen2.script`). We keep its preprocessing
    # (BGR uint8, shortest edge 1200 capped at 1200) and post-processing (box rescale,
    # NMS) and apply our threshold to the returned scores. Default thresholds are the
    # official "camera-rgb" values from gen2/script/constants.py.
    GEN2_DEFAULT_THRESHOLD = {"face": 0.67416, "lp": 0.74475}

    def _load_gen2(self, path):
        from contextlib import ExitStack
        from gen2.script.constants import RESIZE_MAX_GEN2, RESIZE_MIN_GEN2
        from gen2.script.detectron2.export.torchscript_patch import patch_instances
        from gen2.script.predictor import PATCH_INSTANCES_FIELDS, ClassID, EgoblurDetector
        # patch_instances registers the scripted Instances class; keep it open for the
        # detector's lifetime (re-entering per frame would re-script the class each time)
        self._ctx = ExitStack()
        self._ctx.enter_context(patch_instances(fields=PATCH_INSTANCES_FIELDS))
        # gen2.script aliases itself as a top-level `detectron2` module without a __spec__;
        # importlib.util.find_spec("detectron2") then raises ValueError, which breaks
        # transformers/diffusers availability checks (e.g. the SD img2img import used by
        # transform mode). Give the alias a spec so those checks see a normal module.
        import importlib.util
        import sys
        for name, mod in list(sys.modules.items()):
            if (name == "detectron2" or name.startswith("detectron2.")) and getattr(mod, "__spec__", None) is None:
                mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        self.det = EgoblurDetector(
            model_path=str(path), device=self.dev,
            detection_class=ClassID.FACE if self.kind == "face" else ClassID.LICENSE_PLATE,
            score_threshold=0.0, nms_iou_threshold=0.3,
            resize_aug={"min_size_test": RESIZE_MIN_GEN2, "max_size_test": RESIZE_MAX_GEN2})

    def _gen2(self, t):
        torch = self.torch
        img, orig_hw, in_hw = self.det.pre_process(t[None])
        preds = self.det.inference(img)
        d = self.det.get_detections(output_tensor=preds, timestamp_s=0.0, stream_id="",
                                    rotation_angle=0.0, model_input_hw_list=in_hw,
                                    target_img_hw_list=orig_hw)[0]
        if d is None:
            return torch.zeros((0, 4)), torch.zeros(0)
        b, s = (d.face_bboxes, d.face_scores) if self.kind == "face" else (d.lp_bboxes, d.lp_scores)
        return torch.as_tensor(np.asarray(b, np.float32)).reshape(-1, 4), \
            torch.as_tensor(np.asarray(s, np.float32)).reshape(-1)

    def __call__(self, frame_bgr: np.ndarray, threshold: float, nms: float = 0.3):
        import torchvision
        torch = self.torch
        t = torch.from_numpy(np.ascontiguousarray(frame_bgr.transpose(2, 0, 1))).to(self.dev)
        with torch.no_grad():
            if self.gen == "gen2":
                b, s = self._gen2(t)
            else:
                b, _, s, _ = self.m(t)
        if len(b) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        k = torchvision.ops.nms(b, s, nms)
        b, s = b[k].float().cpu().numpy(), s[k].float().cpu().numpy()
        keep = s >= threshold
        return b[keep], s[keep]


class EgoBlurFaceDetector:
    license = "Apache-2.0 (code and weights; Meta EgoBlur, research and commercial use)"

    def __init__(self, device: str = "cuda", threshold: float | None = None, landmarks: str = "yunet",
                 recall_yunet: float = 0.0):
        self.model = _EgoBlurModel("face", device)
        if threshold is None:  # Gen1 0.1 (recall first); Gen2 its official camera-rgb default
            threshold = 0.1 if self.model.gen == "gen1" else _EgoBlurModel.GEN2_DEFAULT_THRESHOLD["face"]
        self.threshold, self.lm_mode = threshold, landmarks
        self.provider = self.model.dev
        self.name = f"EgoBlur {self.model.gen} face ({self.model.path.name}, Apache-2.0)" + \
                    (f" + {landmarks} landmarks" if landmarks != "none" else "")
        self.yunet = None
        if landmarks == "yunet":
            p = MODELS / "face_detection_yunet_2023mar.onnx"
            if p.exists():
                self.yunet = cv2.FaceDetectorYN.create(str(p), "", (320, 320), 0.3, 0.3, 50)
            else:
                print(f"[faces] {p} missing; landmarks fall back to the box template")
        self.n_lm_yunet = self.n_lm_template = 0
        # recall safety net (fail closed): full-frame YuNet faces >= `recall_yunet`
        # that no EgoBlur box covers are added as extra face detections.  Their
        # score is capped below the surrogate gate (0.6), so a track made only of
        # such detections is always BLURRED, never given an unverified surrogate.
        self.recall_thr, self.n_recall = recall_yunet, 0
        self._yn_full = None
        if recall_yunet > 0:
            p = MODELS / "face_detection_yunet_2023mar.onnx"
            if p.exists():
                self._yn_full = cv2.FaceDetectorYN.create(str(p), "", (320, 320), recall_yunet, 0.3, 5000)
                self.name += f" + YuNet recall net (>= {recall_yunet})"

    def _landmarks(self, frame, box) -> np.ndarray:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if self.yunet is not None and min(bw, bh) >= 24:
            H, W = frame.shape[:2]
            X1, Y1 = int(max(0, x1 - 0.3 * bw)), int(max(0, y1 - 0.3 * bh))
            X2, Y2 = int(min(W, x2 + 0.3 * bw)), int(min(H, y2 + 0.3 * bh))
            crop = frame[Y1:Y2, X1:X2]
            s = 160.0 / max(1, max(crop.shape[:2]))   # normalise crop scale for YuNet
            if s < 1:
                crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            else:
                s = 1.0
            self.yunet.setInputSize((crop.shape[1], crop.shape[0]))
            _, f = self.yunet.detect(crop)
            if f is not None and len(f):
                best = max(f, key=lambda r: r[14])
                lm = best[4:14].reshape(5, 2) / s + [X1, Y1]
                c = lm.mean(0)
                if x1 - 0.2 * bw <= c[0] <= x2 + 0.2 * bw and y1 - 0.2 * bh <= c[1] <= y2 + 0.2 * bh:
                    self.n_lm_yunet += 1
                    return lm.astype(np.float32)
        self.n_lm_template += 1
        return template_landmarks(box)

    def __call__(self, frame_bgr: np.ndarray, fidx: int) -> list[Det]:
        boxes, scores = self.model(frame_bgr, self.threshold)
        out = []
        for b, s in zip(boxes, scores):
            lm = None if self.lm_mode == "none" else (
                template_landmarks(b) if self.lm_mode == "template" else self._landmarks(frame_bgr, b))
            out.append(Det(fidx, b.astype(np.float32), float(s), "face", lm))
        if self._yn_full is not None:
            H, W = frame_bgr.shape[:2]
            self._yn_full.setInputSize((W, H))
            _, fs = self._yn_full.detect(frame_bgr)
            for r in (fs if fs is not None else []):
                x, y, w, h = [float(v) for v in r[:4]]
                cx, cy = x + w / 2, y + h / 2
                if any(d.box[0] <= cx <= d.box[2] and d.box[1] <= cy <= d.box[3] for d in out):
                    continue
                b = np.array([max(0, x), max(0, y), min(W - 1, x + w), min(H - 1, y + h)], np.float32)
                out.append(Det(fidx, b, min(0.59, float(r[14])), "face",
                               None if self.lm_mode == "none" else r[4:14].reshape(5, 2).astype(np.float32),
                               {"src": "yunet_recall"}))
                self.n_recall += 1
        return out


class EgoBlurPlateDetector:
    license = "Apache-2.0 (code and weights; Meta EgoBlur)"

    def __init__(self, device: str = "cuda", threshold: float | None = None):
        # Gen1 0.8: on the demo clips Gen1 LP scores dark grilles, bags and a text
        # panel at 0.3-0.8, real plates at >= 0.9 (keeps 40/41 YOLOv9 plates).
        # Gen2: its official camera-rgb default.
        self.model = _EgoBlurModel("lp", device)
        if threshold is None:
            threshold = 0.8 if self.model.gen == "gen1" else _EgoBlurModel.GEN2_DEFAULT_THRESHOLD["lp"]
        self.threshold = threshold
        self.name = f"EgoBlur {self.model.gen} licence plate ({self.model.path.name}, Apache-2.0)"

    def __call__(self, frame_bgr: np.ndarray, fidx: int) -> list[Det]:
        boxes, scores = self.model(frame_bgr, self.threshold)
        return [Det(fidx, b.astype(np.float32), float(s), "license_plate") for b, s in zip(boxes, scores)]


class YoloPlateDetector:
    name = "open-image-models yolo-v9-t-640-license-plate (MIT)"
    license = "MIT (training-data provenance not stated)"

    def __init__(self, device: str = "cuda", threshold: float = 0.35,
                 model: str = "yolo-v9-t-640-license-plate-end2end"):
        import warnings
        from open_image_models import create_detector
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.det = create_detector(model, conf_thresh=threshold, providers=ort_providers(device))

    def __call__(self, frame_bgr: np.ndarray, fidx: int) -> list[Det]:
        out = []
        for r in self.det.predict(frame_bgr):
            b = r.bounding_box
            out.append(Det(fidx, np.array([b.x1, b.y1, b.x2, b.y2], np.float32),
                           float(r.confidence), "license_plate"))
        return out


PlateDetector = YoloPlateDetector   # v0.1 name


def make_face_detector(kind: str, device: str, threshold: float | None, landmarks: str = "yunet",
                       recall_yunet: float = 0.0):
    if kind == "centerface":
        return CenterFaceDetector(device, 0.35 if threshold is None else threshold)
    return EgoBlurFaceDetector(device, threshold, landmarks, recall_yunet)


def make_plate_detector(kind: str, device: str, threshold: float | None):
    if kind == "yolov9":
        return YoloPlateDetector(device, 0.35 if threshold is None else threshold)
    return EgoBlurPlateDetector(device, threshold)
