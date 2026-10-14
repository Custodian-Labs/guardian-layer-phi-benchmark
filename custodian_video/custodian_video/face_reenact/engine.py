"""Minimal, InsightFace-free LivePortrait runtime used for expression-preserving
surrogate faces.

Licences (checked 2026-10):
  * LivePortrait code (vendored in ./liveportrait, upstream KwaiVGI/LivePortrait):
    MIT.  Weights (HF KwaiVGI/LivePortrait, `liveportrait/` subfolder: appearance
    extractor F, motion extractor M, warping W, SPADE generator G, stitching /
    retargeting S, landmark.onnx): MIT.
  * The upstream repo ALSO ships InsightFace buffalo_l detector + 106-pt landmark
    weights, which are non-commercial.  The upstream LICENSE says "remove and
    replace InsightFace's detection models to fully comply with the MIT licence".
    We never download or load them: faces come from the pipeline's own detector
    (CenterFace, 5 points) and dense landmarks from LivePortrait's own MIT
    `landmark.onnx` (203 points).
  * Caveat (data provenance, not licence): LivePortrait was trained on public
    talking-head datasets (VoxCeleb, MEAD, RAVDESS, AAHQ) plus private data.

Coordinate conventions: a "crop" is LivePortrait's square, roll-aligned head crop
(scale 2.3 x landmark extent, shifted 12.5 % downwards), represented by
(cx, cy, size, angle_rad).  M_o2c maps original-frame pixels to a D x D crop.
"""
from __future__ import annotations

import os
from math import acos
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = Path(os.environ.get(
    "CUSTODIAN_LP_WEIGHTS", Path.home() / ".cache" / "custodian_video" / "liveportrait"))

MODEL_CFG = {
    "appearance_feature_extractor": dict(image_channel=3, block_expansion=64, num_down_blocks=2,
                                         max_features=512, reshape_channel=32, reshape_depth=16,
                                         num_resblocks=6),
    "motion_extractor": dict(num_kp=21, backbone="convnextv2_tiny"),
    "warping_module": dict(num_kp=21, block_expansion=64, max_features=512, num_down_blocks=2,
                           reshape_channel=32, estimate_occlusion_map=True,
                           dense_motion_params=dict(block_expansion=32, max_features=1024,
                                                    num_blocks=5, reshape_depth=16, compress=4)),
    "spade_generator": dict(upscale=2, block_expansion=64, max_features=512, num_down_blocks=2),
}
STITCH_CFG = dict(input_size=126, hidden_sizes=[128, 128, 64], output_size=65)

# 203-pt layout (LivePortrait landmark.onnx): 0-23 left eye, 24-47 right eye,
# 48 / 66 mouth corners, 90 / 102 inner-lip mid points.
EYE_L, EYE_R = [0, 6, 12, 18], [24, 30, 36, 42]
MOUTH_L, MOUTH_R, LIP_TOP, LIP_BOT = 48, 66, 90, 102


def weights_available(wdir: Path | str = DEFAULT_WEIGHTS) -> bool:
    w = Path(wdir)
    need = ["base_models/appearance_feature_extractor.pth", "base_models/motion_extractor.pth",
            "base_models/warping_module.pth", "base_models/spade_generator.pth",
            "retargeting_models/stitching_retargeting_module.pth", "landmark.onnx"]
    return all((w / n).exists() for n in need)


def download_weights(wdir: Path | str = DEFAULT_WEIGHTS) -> None:
    """One-off download of the MIT LivePortrait human weights (NOT the
    InsightFace / animal / X-Pose files that live in the same HF repo)."""
    from huggingface_hub import snapshot_download
    tmp = Path(wdir).parent / "_lp_dl"
    snapshot_download("KwaiVGI/LivePortrait", local_dir=str(tmp),
                      allow_patterns=["liveportrait/base_models/*", "liveportrait/landmark.onnx",
                                      "liveportrait/retargeting_models/*"])
    Path(wdir).parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp / "liveportrait", wdir)


# --------------------------------------------------------------------------- geometry
def mouth_open_ratio(l203: np.ndarray) -> float:
    """Inner-lip gap / mouth width (LivePortrait's lip-close ratio)."""
    return float(np.linalg.norm(l203[LIP_TOP] - l203[LIP_BOT]) /
                 (np.linalg.norm(l203[MOUTH_L] - l203[MOUTH_R]) + 1e-6))


def eye_open_ratio(l203: np.ndarray) -> float:
    a = np.linalg.norm(l203[6] - l203[18]) / (np.linalg.norm(l203[0] - l203[12]) + 1e-6)
    b = np.linalg.norm(l203[30] - l203[42]) / (np.linalg.norm(l203[24] - l203[36]) + 1e-6)
    return float((a + b) / 2)


def rect_from_pts(pts: np.ndarray, scale: float, vy_ratio: float, kind: str = "203"):
    """LivePortrait's parse_rect_from_landmark (square, roll-aligned)."""
    if kind == "203":
        le, re_ = pts[EYE_L].mean(0), pts[EYE_R].mean(0)
        lip = (pts[MOUTH_L] + pts[MOUTH_R]) / 2
    else:  # 5-pt CenterFace/ArcFace order: le, re, nose, ml, mr
        le, re_, lip = pts[0], pts[1], (pts[3] + pts[4]) / 2
    p0, p1 = (le + re_) / 2, lip
    uy = p1 - p0
    n = np.linalg.norm(uy)
    uy = np.array([0, 1], np.float32) if n < 1e-3 else uy / n
    ux = np.array([uy[1], -uy[0]], np.float32)
    ang = acos(float(np.clip(ux[0], -1, 1)))
    if ux[1] < 0:
        ang = -ang
    R = np.stack([ux, uy])
    c0 = pts.mean(0)
    r = (pts - c0) @ R.T
    lt, rb = r.min(0), r.max(0)
    c1 = (lt + rb) / 2
    size = float(max(rb - lt)) * scale
    c = c0 + ux * c1[0] + uy * c1[1] + uy * (vy_ratio * size)
    return np.array([c[0], c[1], size, ang], np.float64)


def m_o2c(rect, dsize: int) -> np.ndarray:
    cx, cy, size, ang = rect
    s = dsize / size
    ct, st = np.cos(ang), np.sin(ang)
    t = dsize / 2
    return np.array([[s * ct, s * st, t - s * (ct * cx + st * cy)],
                     [-s * st, s * ct, t - s * (-st * cx + ct * cy)]], np.float64)


def inv2x3(M):
    return cv2.invertAffineTransform(M.astype(np.float64))


def tpts(p, M):
    return p @ M[:, :2].T + M[:, 2]


def smooth_series(x: np.ndarray, sigma: float, circular: bool = False) -> np.ndarray:
    """Zero-phase Gaussian smoothing along axis 0 (offline, so no lag)."""
    if sigma <= 0 or len(x) < 3:
        return x
    if circular:
        x = np.unwrap(x, axis=0)
    r = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    p = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    flat = p.reshape(len(p), -1)
    out = np.stack([np.convolve(flat[:, j], k, mode="valid") for j in range(flat.shape[1])], 1)
    return out.reshape(x.shape)


def face_mask_from_lmk(l203_crop: np.ndarray, dsize: int, forehead: float = 0.28,
                       grow: float = 1.06, feather: float = 0.035) -> np.ndarray:
    """Soft face mask (crop space) from 203 landmarks: convex hull, extended up
    to the hairline (the 203 points stop at the brows) and slightly dilated."""
    pts = l203_crop.astype(np.float32)
    c = pts.mean(0)
    eye_y = (pts[EYE_L].mean(0)[1] + pts[EYE_R].mean(0)[1]) / 2
    h = pts[:, 1].max() - pts[:, 1].min()
    ext = pts.copy()
    up = ext[:, 1] < eye_y
    ext[up, 1] -= forehead * h * np.clip((eye_y - ext[up, 1]) / (h * 0.3 + 1e-6), 0.3, 1.0)
    ext = c + (np.concatenate([pts, ext]) - c) * grow
    m = np.zeros((dsize, dsize), np.float32)
    cv2.fillConvexPoly(m, cv2.convexHull(ext.astype(np.int32)), 1.0)
    k = max(3, int(feather * dsize))
    return cv2.GaussianBlur(m, (0, 0), k)


# --------------------------------------------------------------------------- engine
class LivePortraitEngine:
    def __init__(self, device: str = "cuda", weights: Path | str = DEFAULT_WEIGHTS,
                 half: bool = True):
        import torch
        import onnxruntime as ort
        from .liveportrait.modules.appearance_feature_extractor import AppearanceFeatureExtractor
        from .liveportrait.modules.motion_extractor import MotionExtractor
        from .liveportrait.modules.spade_generator import SPADEDecoder
        from .liveportrait.modules.stitching_retargeting_network import StitchingRetargetingNetwork
        from .liveportrait.modules.warping_network import WarpingNetwork

        w = Path(weights)
        if not weights_available(w):
            raise FileNotFoundError(f"LivePortrait weights not found in {w} "
                                    f"(run custodian_video.face_reenact.engine.download_weights())")
        self.torch = torch
        self.device = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
        self.half = half and self.device == "cuda"

        def ld(cls, name, f):
            m = cls(**MODEL_CFG[name])
            m.load_state_dict(torch.load(w / "base_models" / f, map_location="cpu", weights_only=False))
            return m.to(self.device).eval()

        self.F = ld(AppearanceFeatureExtractor, "appearance_feature_extractor",
                    "appearance_feature_extractor.pth")
        self.M = ld(MotionExtractor, "motion_extractor", "motion_extractor.pth")
        self.W = ld(WarpingNetwork, "warping_module", "warping_module.pth")
        self.G = ld(SPADEDecoder, "spade_generator", "spade_generator.pth")
        ck = torch.load(w / "retargeting_models" / "stitching_retargeting_module.pth",
                        map_location="cpu", weights_only=False)
        self.S = StitchingRetargetingNetwork(**STITCH_CFG)
        self.S.load_state_dict({k.replace("module.", ""): v for k, v in ck["retarget_shoulder"].items()})
        self.S = self.S.to(self.device).eval()

        prov = ["CPUExecutionProvider"]
        if self.device == "cuda":
            try:
                ort.preload_dlls()
            except Exception:
                pass
            if "CUDAExecutionProvider" in ort.get_available_providers():
                prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.lmk_sess = ort.InferenceSession(str(w / "landmark.onnx"), providers=prov)

    # ---------------- landmarks
    def _lmk_once(self, frame_bgr, rect):
        M = m_o2c(rect, 224)
        crop = cv2.warpAffine(frame_bgr, M, (224, 224), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.
        out = self.lmk_sess.run(None, {"input": x})
        p = out[2][0].reshape(-1, 2) * 224
        return tpts(p, inv2x3(M))

    def landmarks203(self, frame_bgr: np.ndarray, lm5: np.ndarray | None = None,
                     prev203: np.ndarray | None = None) -> np.ndarray:
        """203 landmarks in frame coords, seeded by 5 points (or a previous 203)."""
        if prev203 is not None:
            rect = rect_from_pts(prev203, 1.5, -0.1)
        else:
            rect = rect_from_pts(np.asarray(lm5, np.float32), 3.2, -0.12, kind="5")
        p = self._lmk_once(frame_bgr, rect)
        return self._lmk_once(frame_bgr, rect_from_pts(p, 1.5, -0.1))  # refine

    # ---------------- motion / appearance
    def _to_t(self, imgs_bgr):
        torch = self.torch
        x = np.stack([cv2.cvtColor(cv2.resize(i, (256, 256), interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2RGB) for i in imgs_bgr]).astype(np.float32) / 255.
        return torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)

    def _ctx(self):
        torch = self.torch
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.half) \
            if self.device == "cuda" else torch.autocast(device_type="cpu", enabled=False)

    def motion(self, crops_bgr: list) -> dict:
        """Batched implicit-keypoint motion: pitch/yaw/roll (deg), exp (Bx21x3), t, scale, kp."""
        from .liveportrait.camera import headpose_pred_to_degree
        torch = self.torch
        out = {k: [] for k in ("pitch", "yaw", "roll", "exp", "t", "scale", "kp")}
        for i in range(0, len(crops_bgr), 32):
            with torch.no_grad(), self._ctx():
                k = self.M(self._to_t(crops_bgr[i:i + 32]))
            bs = k["kp"].shape[0]
            for n in ("pitch", "yaw", "roll"):
                out[n].append(headpose_pred_to_degree(k[n].float()).reshape(bs).cpu().numpy())
            out["exp"].append(k["exp"].float().reshape(bs, -1, 3).cpu().numpy())
            out["kp"].append(k["kp"].float().reshape(bs, -1, 3).cpu().numpy())
            out["t"].append(k["t"].float().cpu().numpy())
            out["scale"].append(k["scale"].float().reshape(bs).cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}

    def rot(self, pitch, yaw, roll):
        from .liveportrait.camera import get_rotation_matrix
        torch = self.torch
        f = lambda a: torch.as_tensor(np.atleast_1d(a), dtype=torch.float32, device=self.device)
        return get_rotation_matrix(f(pitch), f(yaw), f(roll))

    def source(self, crop_bgr) -> dict:
        torch = self.torch
        x = self._to_t([crop_bgr])
        with torch.no_grad(), self._ctx():
            f_s = self.F(x).float()
        m = self.motion([crop_bgr])
        info = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in m.items()}
        R_s = self.rot(m["pitch"], m["yaw"], m["roll"])
        x_s = self._kp(info["kp"], R_s, info["exp"], info["scale"], info["t"])
        return dict(f_s=f_s, info=info, R_s=R_s, x_s=x_s, raw=m)

    def _kp(self, kp, R, exp, scale, t):
        x = scale[:, None, None] * (kp @ R + exp)
        x[:, :, :2] = x[:, :, :2] + t[:, None, :2]
        return x

    # expression components LivePortrait copies from the driver in its absolute
    # (non-relative) mode: lips, eyes, brows; the rest stays the source's.
    EXP_FULL = [1, 2, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    def render(self, src: dict, R, exp, t=None, scale=None, absolute: bool = True) -> np.ndarray:
        """absolute=True : R = driving rotation, exp = driving expression (21x3),
                           t / scale = driving translation / scale (geometry follows
                           the original face inside its crop; identity = source's
                           canonical keypoints + appearance volume).
        absolute=False: R = relative rotation (driving_i @ driving_ref^T), exp =
                           expression delta (exp_i - exp_ref), source t / scale.
        Returns a 512x512 BGR crop."""
        torch = self.torch
        info = src["info"]
        dev = self.device
        Rm = torch.as_tensor(R, dtype=torch.float32, device=dev)[None]
        e = torch.as_tensor(exp, dtype=torch.float32, device=dev)[None]
        if absolute:
            R_new = Rm
            ex = info["exp"].clone()
            ex[:, self.EXP_FULL] = e[:, self.EXP_FULL]
            ex[:, 3:5, 1] = e[:, 3:5, 1]
            ex[:, 5, 2] = e[:, 5, 2]
            ex[:, 8, 2] = e[:, 8, 2]
            ex[:, 9, 1:] = e[:, 9, 1:]
        else:
            R_new = Rm @ src["R_s"]
            ex = info["exp"] + e
        tt = info["t"].clone() if t is None else torch.as_tensor(np.reshape(t, (1, 3)), dtype=torch.float32, device=dev)
        tt[:, 2] = 0
        sc = info["scale"] if scale is None else torch.as_tensor([float(scale)], dtype=torch.float32, device=dev)
        x_d = self._kp(info["kp"], R_new, ex, sc, tt)
        with torch.no_grad():
            feat = torch.cat([src["x_s"].reshape(1, -1), x_d.reshape(1, -1)], 1)
            delta = self.S(feat)
            n = x_d.shape[1]
            x_d = x_d + delta[:, :3 * n].reshape(1, n, 3)
            x_d[..., :2] = x_d[..., :2] + delta[:, 3 * n:3 * n + 2].reshape(1, 1, 2)
            with self._ctx():
                o = self.W(src["f_s"], kp_source=src["x_s"], kp_driving=x_d)
                img = self.G(feature=o["out"])
        img = img.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
