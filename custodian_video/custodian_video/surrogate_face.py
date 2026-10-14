"""`transform` mode for faces: replace each face TRACK with ONE consistent
synthetic surrogate identity (the video analogue of replacing a name with a
realistic fake name, consistently across the document).

Backend `sd15-keyframe`, two motion modes (`motion=`):

  "reenact" (default when the LivePortrait weights are present)
    1. per track pick the best keyframe (largest, most confident detection),
    2. crop it LivePortrait-style (roll-aligned head crop from 203 landmarks of
       LivePortrait's own MIT `landmark.onnx`, seeded by the CenterFace points),
    3. Stable Diffusion 1.5 img2img with a per-track seed -> a new identity that
       keeps coarse pose / lighting / skin tone / hair but not the face,
    4. every frame of the track: LivePortrait motion extractor on the ORIGINAL
       face crop -> head rotation + 21x3 implicit-keypoint expression; these are
       transferred RELATIVE to the keyframe onto the synthetic source (identity
       = synthetic canonical keypoints + appearance volume; motion = original),
       then warp + SPADE decode (MIT code and weights, no InsightFace),
    5. temporal smoothing (zero-phase, offline: crops, pose, expression, masks),
       paste back with a landmark-hull face mask, LAB colour match (EMA).
    Fallbacks per frame: static warp (below) when the face is small, the head is
    in strong profile or the dense landmarks are implausible; blur (return False)
    when the face is too small even for that.

  "static" (old behaviour, `motion="static"` or CUSTODIAN_FACE_MOTION=static)
    one SD-1.5 face per track similarity-warped onto the 5 landmarks every
    frame; mouth/eyes frozen.

Select with the env var CUSTODIAN_FACE_MOTION=reenact|static (pipeline.py does
not pass `motion` yet) or the constructor argument.

Realism (`realism=`, `--surrogate-realism` or CUSTODIAN_FACE_REALISM=v2|v1,
default v2; v1 = the behaviour before 2026-10, unchanged up to GPU run-to-run noise):

  v2 identity generation
    * coarse attributes of the keyframe face (apparent age band, gender
      presentation, skin tone, apparent ancestry, hair, facial hair, glasses)
      from the local Qwen3-VL (face_attributes.py, Apache-2.0), kept in memory
      only, turned into a prompt + an anti-beautification negative prompt (no
      make-up / airbrushing / glamour); CUSTODIAN_FACE_ATTRIBUTES=0 disables the
      VLM, CUSTODIAN_FACE_ATTR_APPEARANCE=0 drops the ancestry term;
    * SD-1.5 img2img (CreativeML OpenRAIL-M) + ControlNet v1.1 depth
      (lllyasviel/control_v11f1p_sd15_depth, OpenRAIL) on a blurred
      Depth-Anything-V2-Small depth map (Apache-2.0): head shape / width / pose
      are kept, nose / eye / lip relief is free to change; sd-vae-ft-mse decoder
      (MIT); grey-world white balance of the SD input (cast re-applied after) so
      a warm colour cast is not read as a different skin tone; seed-chosen
      facial-feature words make different tracks different people;
    * generation-time identity guard: up to `gen_tries` deterministic candidates
      (fixed per-track seed + i, mildly escalating denoise / decreasing control)
      are each re-enacted and pasted on 16 sampled frames of the track; the
      first whose 75th-percentile SFace cosine to the original is <= `gen_thr`
      is kept, else the lowest-scoring one.
  v2 paste-back (every frame is rendered in prepare(), so alignment and masks
  are smoothed zero-phase over time):
    * the LivePortrait output is landmarked (MIT landmark.onnx) and similarity-
      aligned to the original's jaw / cheek OUTLINE (fixes LivePortrait's size /
      offset mismatch that showed hair / background as a dark smear on the cheek
      and a doubled hairline); eyes / nose / mouth keep the surrogate's layout;
    * replaced region = the face: hull of the original's 203 landmarks plus a
      forehead band, INTERSECTED with the rendered face's own hull (the
      surrogate's hair / background is never pasted), feathered; in the
      forehead band only where the surrogate shows skin;
    * per-frame colour match (LAB, light EMA) + low-frequency illumination match
      inside the mask (normalised convolution);
    * texture match: the surrogate is blurred to the original's sharpness and
      luma grain is added up to the original's (robust, MAD) noise level;
    * optional Poisson blending (`blend="poisson"`, cv2.seamlessClone; not better
      on the demo clips);
    * per-frame identity guard as before, with an extra first retry that covers
      more of the face; the static fallback is also guarded (blur if it trips)
      and uses an inner-face ellipse for faces >= 96 px.
"""
from __future__ import annotations

import hashlib
import os
import time

import cv2
import numpy as np

# ArcFace 112x112 5-point template, scaled into a 512 crop (face ~ central 45%)
_T112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                  [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
_S = 230 / 112
TEMPLATE = _T112 * _S + (512 - 112 * _S) / 2
TEMPLATE[:, 1] += 10

LP_D = 512           # LivePortrait output / crop size
LP_SCALE, LP_VY = 2.3, -0.125


def _mask512(inner: bool = False) -> np.ndarray:
    m = np.zeros((512, 512), np.float32)
    c = TEMPLATE.mean(0)
    if inner:   # v2 static fallback: inner face only (brows to chin), feathered inwards
        cv2.ellipse(m, (int(c[0]), int(c[1]) - 6), (98, 132), 0, 0, 360, 1.0, -1)
        m = cv2.erode(m, np.ones((15, 15), np.uint8))
        return cv2.GaussianBlur(m, (0, 0), 9)
    cv2.ellipse(m, (int(c[0]), int(c[1]) - 18), (128, 168), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), 14)


# LivePortrait 203-pt layout: 0-47 eyes, 48-107 mouth, 108-144 jaw contour,
# 145-180 brows, 181-202 nose
EYE_L, EYE_R = [0, 6, 12, 18], [24, 30, 36, 42]
INNER = np.r_[0:108, 145:203]
CONTOUR = np.r_[108:145]          # jaw / cheek outline
BROWS = np.r_[145:181]


def _hull_mask(p, dsize, forehead):
    p = np.asarray(p, np.float32)
    top = p[BROWS].copy()
    h = p[:, 1].max() - top[:, 1].mean()
    top[:, 1] -= forehead * h
    m = np.zeros((dsize, dsize), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(np.concatenate([p, top])).astype(np.int32)), 255)
    return m


def inner_face_mask(p_orig, p_rend, dsize: int, forehead: float = 0.25, erode: float = 0.03,
                    feather: float = 0.035) -> np.ndarray:
    """Soft inner-face mask (crop space, 0..1): hull of the ORIGINAL's 203
    landmarks plus a forehead band, intersected with the rendered face's hull
    (so warped hair / background of the surrogate is never pasted), eroded and
    feathered INWARDS (alpha reaches 0 at the hull, 1 well inside)."""
    m = _hull_mask(p_orig, dsize, forehead)
    if p_rend is not None:
        m = cv2.bitwise_and(m, _hull_mask(p_rend, dsize, forehead))
    size = np.sqrt(max(1.0, float((m > 0).sum())))
    k = int(round((erode + 1.5 * feather) * size))
    if k > 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    elif k < 0:   # negative erode = grow (still clipped to the rendered face's hull below)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-2 * k + 1, -2 * k + 1)))
        if p_rend is not None:
            m = cv2.bitwise_and(m, cv2.dilate(_hull_mask(p_rend, dsize, forehead),
                                              np.ones((3, 3), np.uint8)))
    return cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), max(0.5, feather * size))


_FEATURES = [
    ["narrow nose", "broad nose", "straight nose", "small upturned nose", "aquiline nose", "wide flat nose"],
    ["thin lips", "full lips", "wide mouth", "small mouth", "thin upper lip"],
    ["deep-set eyes", "hooded eyes", "round eyes", "almond-shaped eyes", "wide-set eyes", "close-set eyes"],
    ["square jaw", "pointed chin", "round chin", "cleft chin", "soft jawline", "strong chin"],
    ["thick eyebrows", "thin eyebrows", "arched eyebrows", "straight eyebrows"],
    ["high cheekbones", "full cheeks", "hollow cheeks", "freckles", "a few moles", "dimples"],
]


def feature_descriptors(seed: int) -> str:
    """Seed-chosen, attribute-neutral facial-feature words: different tracks /
    candidates get different face structure while age, gender presentation,
    skin tone and hair stay as observed."""
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    return ", ".join(str(rng.choice(c)) for c in _FEATURES)


def _fit_outline(pr, po):
    """Similarity (2x3) mapping the rendered face's jaw/cheek outline onto the
    original's (least squares on the 37 contour points)."""
    A, _ = cv2.estimateAffinePartial2D(np.asarray(pr, np.float32)[CONTOUR],
                                       np.asarray(po, np.float32)[CONTOUR], method=cv2.LMEDS)
    if A is None:
        return np.float64([[1, 0, 0], [0, 1, 0]])
    s = np.hypot(A[0, 0], A[1, 0])
    if not 0.7 < s < 1.4:
        return np.float64([[1, 0, 0], [0, 1, 0]])
    return A.astype(np.float64)


def _hf_std(g, sel):
    """Robust high-frequency level (1.4826 x MAD of grey - Gaussian(1.0)) inside
    `sel`: dominated by flat-skin texture / sensor + codec noise, not by the
    sparse strong edges of eyes, brows and mouth."""
    if not sel.any():
        return 0.0
    r = (g - cv2.GaussianBlur(g, (0, 0), 1.0))[sel]
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


class FaceSurrogateBackend:
    name = "abstract"
    license = ""

    def prepare(self, video_path: str, tracks) -> None: ...

    def apply(self, frame: np.ndarray, track, box, lm) -> bool:
        """Paint surrogate in-place. Return False if not possible (caller redacts)."""
        raise NotImplementedError


class DeepPrivacy2Backend(FaceSurrogateBackend):
    """Hook: DeepPrivacy2 (github.com/hukkelas/deep_privacy2). Code Apache-2.0 but
    checkpoints follow FDF/FDH (CC BY-NC-SA) and sg3_torch_utils is NVIDIA
    non-commercial -> research/eval only. Install separately and implement
    `apply` by calling its face generator on the aligned crop."""
    name = "deepprivacy2 (hook, not bundled)"

    def __init__(self, *a, **k):
        raise NotImplementedError("DeepPrivacy2 backend is a documented hook; see README.")


def _lab_transfer(src, ref, sel, state=None, ema=0.35, src_small=None):
    """Match src's LAB mean/std (inside `sel`) halfway towards ref's. `state`
    (dict) keeps an EMA of the statistics to avoid colour flicker. If
    `src_small` is given, statistics come from it (same size as ref/sel) and the
    transform is applied to the full-size `src`."""
    a = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_s = a if src_small is None else cv2.cvtColor(src_small, cv2.COLOR_BGR2LAB).astype(np.float32)
    st = np.stack([a_s[sel].mean(0), a_s[sel].std(0) + 1e-3, b[sel].mean(0), b[sel].std(0) + 1e-3])
    if state is not None:
        if "st" in state:
            st = (1 - ema) * state["st"] + ema * st
        state["st"] = st
    ma, sa, mb, sb = st
    a = (a - ma) / sa * (0.5 * sa + 0.5 * sb) + mb
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


class SDKeyframeSurrogate(FaceSurrogateBackend):
    name = "sd15-keyframe (Stable Diffusion 1.5 img2img, CreativeML OpenRAIL-M)"

    def __init__(self, device: str = "cuda", strength: float = 0.75, steps: int = 30,
                 model_id: str = "runwayml/stable-diffusion-v1-5", min_face: int = 64,
                 min_score: float = 0.6, seed_salt: str = "custodian",
                 motion: str | None = None, reenact_min_face: int | None = None,
                 max_abs_yaw: float = 50.0, exp_sigma: float = 0.7, pose_sigma: float = 1.5,
                 crop_sigma: float = 2.0, lip_gain: float = 1.0, drive: str = "hybrid",
                 mask_grow: float = 1.10, cut_thr: float = 0.5,
                 guard_thr: float | None = 0.30, verbose: bool = True,
                 realism: str | None = None, blend: str = "alpha", gen_thr: float = 0.22,
                 gen_tries: int = 10, v2_strength: float = 0.85, control_scale: float = 0.8,
                 attributes: bool = True, grain: bool = True, depth_blur: float = 0.02,
                 align: float = 1.0, mask_erode: float = -0.08, mask_forehead: float = 0.55,
                 v2_min_face: int | None = None):
        self.device, self.strength, self.steps = device, strength, steps
        self.model_id, self.min_face, self.salt = model_id, min_face, seed_salt
        self.min_score = min_score
        self.pipe = None
        self.surr: dict[int, np.ndarray] = {}
        self.mask = _mask512()
        self.prep_seconds = 0.0
        self.motion = (motion or os.environ.get("CUSTODIAN_FACE_MOTION", "reenact")).lower()
        assert self.motion in ("reenact", "static"), self.motion
        self.reenact_min_face = reenact_min_face or min_face
        self.max_abs_yaw, self.lip_gain = max_abs_yaw, lip_gain
        self.exp_sigma, self.pose_sigma, self.crop_sigma = exp_sigma, pose_sigma, crop_sigma
        self.verbose = verbose
        self.drive = drive            # hybrid | absolute | relative (to the keyframe)
        self.mask_grow, self.cut_thr = mask_grow, cut_thr
        self.guard_thr, self._rec = guard_thr, None   # per-frame identity guard (None = off)
        self.lp = None                      # LivePortraitEngine
        self.src: dict[int, dict] = {}      # tid -> LivePortrait source (synthetic identity)
        self.plan: dict[int, dict] = {}     # tid -> {frame: per-frame drive}
        self._fidx: dict[tuple, int] = {}   # (tid, id(box)) -> frame
        self._col: dict[int, dict] = {}     # tid -> colour EMA state
        self.stats = {"reenact": 0, "static": 0, "none": 0}
        self.render_seconds = 0.0
        self.realism = (realism or os.environ.get("CUSTODIAN_FACE_REALISM", "v2")).lower()
        assert self.realism in ("v1", "v2"), self.realism
        self.blend = os.environ.get("CUSTODIAN_FACE_BLEND", blend)
        self.gen_thr, self.gen_tries = gen_thr, gen_tries
        self.v2_strength, self.control_scale = v2_strength, control_scale
        self.use_attributes = attributes and os.environ.get("CUSTODIAN_FACE_ATTRIBUTES", "1") != "0"
        self.grain, self.depth_blur = grain, depth_blur
        self.outline_align, self.mask_erode, self.mask_forehead = align, mask_erode, mask_forehead
        # v2 surrogates only on faces >= this many px (detector box short side); smaller -> blur
        self.v2_min_face = int(v2_min_face or os.environ.get("CUSTODIAN_FACE_V2_MIN_FACE", 64))
        self._guard_failed = False
        self.band_gate = "render"            # forehead band: replace where the surrogate shows skin
        self.mask_feather, self.illum_sigma = 0.035, 0.12
        self._illum = 1.0                     # weight of the low-frequency illumination match
        self.cpipe = self.depth = None
        self.gen_info: dict[int, dict] = {}   # tid -> {cos, tries} (no attributes: privacy)
        self._tex: dict[int, dict] = {}       # tid -> texture-match EMA state
        if self.motion == "reenact":
            self.name = ("sd15-keyframe+liveportrait-reenact (SD-1.5 identity, CreativeML OpenRAIL-M; "
                         "LivePortrait motion, MIT code+weights, no InsightFace)")
        if self.realism == "v2":
            self.mask, self._mask_full = _mask512(inner=True), _mask512()
            self.name = self.name.replace("sd15-keyframe", "sd15-keyframe-v2") + \
                " + v2 realism (Qwen3-VL attributes Apache-2.0; ControlNet-depth OpenRAIL; " \
                "Depth-Anything-V2-Small Apache-2.0; sd-vae-ft-mse MIT)"

    def _log(self, *a):
        if self.verbose:
            print("[surrogate]", *a, flush=True)

    # ------------------------------------------------------------------ SD
    def _load(self):
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline

        def build(dev):
            dt = torch.float16 if dev != "cpu" else torch.float32
            p = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id, torch_dtype=dt, safety_checker=None, requires_safety_checker=False)
            return p.to("cuda" if dev != "cpu" else "cpu")
        try:
            self.pipe = build(self.device)
        except torch.OutOfMemoryError:  # shared GPU is full -> slow but works
            print("[surrogate] CUDA OOM loading SD; falling back to CPU (slow)")
            self.pipe = None
            torch.cuda.empty_cache()
            self.device = "cpu"
            self.pipe = build("cpu")
        self.pipe.set_progress_bar_config(disable=True)

    @staticmethod
    def align(frame, lm):
        M, _ = cv2.estimateAffinePartial2D(lm.astype(np.float32), TEMPLATE, method=cv2.LMEDS)
        return cv2.warpAffine(frame, M, (512, 512), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT), M

    def generate(self, crop_bgr: np.ndarray, seed: int) -> np.ndarray:
        import torch
        from PIL import Image
        if self.pipe is None:
            self._load()
        g = torch.Generator(device="cuda" if self.device != "cpu" else "cpu").manual_seed(seed)
        img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        out = self.pipe(prompt="a natural candid photo of a different person's face, "
                               "realistic skin texture, sharp focus, same lighting",
                        negative_prompt="cartoon, painting, deformed, blurry, extra eyes, text, "
                                        "watermark, disfigured",
                        image=img, strength=self.strength, num_inference_steps=self.steps,
                        guidance_scale=7.0, generator=g).images[0]
        return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------ SD v2
    def _load_v2(self):
        import torch
        from diffusers import (AutoencoderKL, ControlNetModel,
                               StableDiffusionControlNetImg2ImgPipeline)
        from transformers import pipeline as hf_pipeline
        dev = "cuda" if self.device != "cpu" and torch.cuda.is_available() else "cpu"
        dt = torch.float16 if dev == "cuda" else torch.float32
        self.depth = hf_pipeline("depth-estimation", model=os.environ.get(
            "CUSTODIAN_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf"),
            device=0 if dev == "cuda" else -1)
        cn = ControlNetModel.from_pretrained(os.environ.get(
            "CUSTODIAN_CONTROLNET", "lllyasviel/control_v11f1p_sd15_depth"),
            torch_dtype=dt, **({"variant": "fp16"} if dt == torch.float16 else {}))
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dt)
        self.cpipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.model_id, controlnet=cn, vae=vae, torch_dtype=dt, safety_checker=None,
            requires_safety_checker=False).to(dev)
        self.cpipe.set_progress_bar_config(disable=True)
        self._gen_dev = dev

    def generate_v2(self, crop_bgr: np.ndarray, seed: int, attrs: dict, score_fn=None,
                    max_tries: int | None = None):
        """Attribute-conditioned, depth-controlled SD identity for one track.
        Candidates (fixed per-track seed + i) are scored by `score_fn(image)`
        (identity similarity to the original, lower = better; None = accept);
        the first scoring <= gen_thr is kept, else the lowest-scoring one.
        Returns (image, score or None, tries)."""
        import torch
        from PIL import Image
        from .face_attributes import prompt_from_attributes
        if self.cpipe is None:
            self._load_v2()
        rgb = lambda im: Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        d = np.array(self.depth(rgb(crop_bgr))["depth"].resize(crop_bgr.shape[1::-1]))
        d = cv2.normalize(d.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        # keep only the coarse head shape (outline, width, pose); fine relief of
        # nose / eyes / lips is identity and is left free to change
        d = cv2.GaussianBlur(d, (0, 0), self.depth_blur * d.shape[0]).astype(np.uint8)
        ctrl = Image.fromarray(np.stack([d] * 3, -1))
        # grey-world white balance of the SD input; the cast is re-applied afterwards
        f = crop_bgr.reshape(-1, 3).astype(np.float32).mean(0) + 1e-3
        gain = np.clip(f.mean() / f, 0.6, 1.6)
        wb = np.clip(crop_bgr * gain, 0, 255).astype(np.uint8)
        pos, neg = prompt_from_attributes(attrs or {})
        if os.environ.get("CUSTODIAN_DEBUG_GEN"):
            self._log(f"prompt: {pos} | neg: {neg}")
        best = None
        for i in range(max(1, max_tries or self.gen_tries)):
            g = torch.Generator(device=self._gen_dev).manual_seed(seed + 7919 * i)
            pos_i = pos + ", " + feature_descriptors(seed + 7919 * i)
            # escalate mildly: later candidates denoise a bit more, constrain geometry less
            stg = min(0.95, self.v2_strength + 0.02 * i)
            cs = max(0.5, self.control_scale - 0.04 * i)
            out = self.cpipe(prompt=pos_i, negative_prompt=neg, image=rgb(wb), control_image=ctrl,
                             strength=stg, controlnet_conditioning_scale=cs,
                             control_guidance_end=max(0.7, 1.0 - 0.04 * i),
                             num_inference_steps=self.steps, guidance_scale=6.0, generator=g).images[0]
            im = np.clip(cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR) / gain, 0, 255).astype(np.uint8)
            c = score_fn(im) if score_fn is not None else None
            if os.environ.get("CUSTODIAN_DEBUG_GEN"):
                self._log(f"gen try {i}: score {c}")
                cv2.imwrite(os.path.join(os.environ["CUSTODIAN_DEBUG_GEN"], f"gen_{seed}_{i}.jpg"),
                            np.hstack([crop_bgr, im]))
            if best is None or (c is not None and c < best[1]):
                best = (im, c, i + 1)
            if c is None or c <= self.gen_thr:
                break
        return best

    def _seed(self, video_path, tid):
        return int(hashlib.sha256(f"{self.salt}|{video_path}|{tid}".encode()).hexdigest()[:8], 16)

    def _v2_gate(self, box, lm) -> bool:
        """v2 fail-closed gate: a surrogate is only painted on a face whose
        (padded) box is >= 1.25 x v2_min_face, that has landmarks, and when the
        identity guard is available; everything else is blurred.  Below ~64 px a
        surrogate was measurably MORE re-identifiable than blur (face_e, clip6)."""
        if lm is None or min(box[2] - box[0], box[3] - box[1]) < 1.25 * self.v2_min_face:
            self.stats["v2_gate_blur"] = self.stats.get("v2_gate_blur", 0) + 1
            return False
        if not self._guard_ready() and os.environ.get("CUSTODIAN_FACE_ALLOW_UNGUARDED") != "1":
            self.stats["no_guard_blur"] = self.stats.get("no_guard_blur", 0) + 1
            return False
        return True

    def _keyframes(self, tracks):
        want = {}
        for t in tracks:
            cands = [d for d in t.dets.values() if d.lm is not None]
            if not cands:
                continue
            best = max(cands, key=lambda d: d.score * min(d.box[2] - d.box[0], d.box[3] - d.box[1]))
            mf = max(self.min_face, self.v2_min_face) if self.realism == "v2" else self.min_face
            if min(best.box[2] - best.box[0], best.box[3] - best.box[1]) < mf \
                    or best.score < self.min_score:
                continue  # small / low-confidence (maybe not a face): blur instead
            want.setdefault(best.frame, []).append((t, best))
        return want

    # ------------------------------------------------------------------ prepare
    def prepare(self, video_path: str, tracks) -> None:
        t0 = time.time()
        tracks = list(tracks)
        for t in tracks:
            for f, v in t.dense.items():
                self._fidx[(t.tid, id(v[0]))] = f
        if self.motion == "reenact":
            try:
                self._prepare_reenact(video_path, tracks)
            except Exception as e:  # never lose the face channel because of the re-enactor
                import traceback
                traceback.print_exc()
                self._log(f"re-enactment unavailable ({type(e).__name__}: {e}); "
                          f"falling back to static surrogates")
                self.motion, self.plan, self.src, self.lp = "static", {}, {}, None
                self.name = SDKeyframeSurrogate.name
        if self.motion == "static":
            self._prepare_static(video_path, tracks)
        self.prep_seconds = time.time() - t0

    def _prepare_static(self, video_path, tracks):
        want = self._keyframes([t for t in tracks if t.tid not in self.surr])
        if not want:
            return
        cap = cv2.VideoCapture(video_path)
        for f in sorted(want):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            if not ok:
                continue
            for t, d in want[f]:
                crop, _ = self.align(fr, d.lm)
                self.surr[t.tid] = self.generate(crop, self._seed(video_path, t.tid))
        cap.release()

    def _usable(self, box):
        return min(box[2] - box[0], box[3] - box[1]) / 1.3 >= self.reenact_min_face * 0.75

    def _prepare_reenact(self, video_path, tracks):
        from .face_reenact.engine import (LivePortraitEngine, m_o2c, rect_from_pts, smooth_series,
                                          inv2x3, tpts, mouth_open_ratio, weights_available)
        if not weights_available():
            raise FileNotFoundError("LivePortrait weights missing "
                                    "(custodian_video.face_reenact.engine.download_weights())")
        want = self._keyframes(tracks)
        kf = {t.tid: (f, d) for f, L in want.items() for t, d in L}
        if not kf:
            return
        try:
            self.lp = LivePortraitEngine(self.device)
        except Exception as e:
            import torch
            if isinstance(e, torch.OutOfMemoryError):
                torch.cuda.empty_cache()
                self._log("CUDA OOM loading LivePortrait; using CPU (slow)")
                self.lp = LivePortraitEngine("cpu")
            else:
                raise
        E = self.lp
        tr = {t.tid: t for t in tracks if t.tid in kf}
        # frames to drive: every dense frame with landmarks and a big-enough box
        need: dict[int, list] = {}
        for tid, t in tr.items():
            for f, (box, lm, src) in t.dense.items():
                if lm is not None and self._usable(box):
                    need.setdefault(f, []).append(tid)
            need.setdefault(kf[tid][0], [])
            if tid not in need[kf[tid][0]]:
                need[kf[tid][0]].append(tid)

        # pass A: dense 203-pt landmarks + hard shot cuts (HSV histogram correlation)
        l203: dict[int, dict] = {tid: {} for tid in tr}
        cap = cv2.VideoCapture(video_path)
        f = 0
        last = max(max(need), max(max(t.dense) for t in tr.values()))
        cuts, prev_h = set(), None
        while f <= last:
            ok, fr = cap.read()
            if not ok:
                break
            hsv = cv2.cvtColor(cv2.resize(fr, (160, 90)), cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(h, h)
            if prev_h is not None and cv2.compareHist(prev_h, h, cv2.HISTCMP_CORREL) < self.cut_thr:
                cuts.add(f)          # frame f starts a new shot
            prev_h = h
            for tid in need.get(f, []):
                lm = tr[tid].dense[f][1] if f in tr[tid].dense else kf[tid][1].lm
                l203[tid][f] = E.landmarks203(fr, lm)
            f += 1
        cap.release()

        # smooth crops; plausibility check of dense landmarks vs CenterFace points
        rects: dict[int, dict] = {}
        for tid, D in l203.items():
            fs = sorted(D)
            if not fs:
                continue
            ok = []
            for g in fs:
                lm = tr[tid].dense[g][1] if g in tr[tid].dense else kf[tid][1].lm
                p = D[g]
                eyes = np.stack([p[[0, 6, 12, 18]].mean(0), p[[24, 30, 36, 42]].mean(0)])
                iod = np.linalg.norm(lm[0] - lm[1]) + 1e-3
                ok.append(np.abs(eyes - lm[:2]).max() / iod < 0.35)
            R = np.stack([rect_from_pts(D[g], LP_SCALE, LP_VY) for g in fs])
            # smooth per contiguous run of frames
            runs, cur = [], [0]
            for i in range(1, len(fs)):
                if fs[i] - fs[i - 1] == 1:
                    cur.append(i)
                else:
                    runs.append(cur); cur = [i]
            runs.append(cur)
            Rs, Ls = R.copy(), {}
            P = np.stack([D[g] for g in fs])
            for run in runs:
                Rs[run, :3] = smooth_series(R[run, :3], self.crop_sigma)
                Rs[run, 3] = smooth_series(R[run, 3], self.crop_sigma, circular=True)
                P[run] = smooth_series(P[run], 1.0)
            rects[tid] = {g: (Rs[i], P[i], ok[i]) for i, g in enumerate(fs)}

        # pass B: driving crops -> motion; keyframe -> SD identity source
        crops: dict[int, list] = {tid: [] for tid in rects}
        frames_of: dict[int, list] = {tid: [] for tid in rects}
        kcrop: dict[int, tuple] = {}
        cap = cv2.VideoCapture(video_path)
        f = 0
        while f <= last:
            ok, fr = cap.read()
            if not ok:
                break
            for tid in need.get(f, []):
                if tid not in rects or f not in rects[tid]:
                    continue
                rect = rects[tid][f][0]
                c = cv2.warpAffine(fr, m_o2c(rect, LP_D), (LP_D, LP_D), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
                crops[tid].append(cv2.resize(c, (256, 256), interpolation=cv2.INTER_AREA))
                frames_of[tid].append(f)
                if f == kf[tid][0]:
                    kcrop[tid] = (c, rect, rects[tid][f][1], kf[tid][1].lm)
            f += 1
        cap.release()
        # SD identity on the LivePortrait crop of each track's keyframe
        attrs = {}
        if self.realism == "v2" and self.use_attributes and kcrop:
            try:
                from .face_attributes import FaceAttributes
                fa = FaceAttributes()
                attrs = fa({tid: v[0] for tid, v in kcrop.items()})
                self._log(f"face attributes for {sum(bool(a) for a in attrs.values())}/{len(kcrop)} "
                          f"tracks in {fa.seconds:.1f}s (kept in memory only)")
            except Exception as e:
                self._log(f"face attributes unavailable ({type(e).__name__}: {e}); neutral prompt")
        # motion of the ORIGINAL (needed before generation: v2 scores candidates
        # on re-enacted frames)
        mot = {}
        for tid in kcrop:
            fs = frames_of[tid]
            mo = E.motion(crops[tid])
            ang = np.stack([mo["pitch"], mo["yaw"], mo["roll"]], 1)
            exp = mo["exp"].copy()
            ts = np.concatenate([mo["t"], mo["scale"][:, None]], 1)
            # smooth per contiguous run
            brk = [0] + [i for i in range(1, len(fs)) if fs[i] - fs[i - 1] > 1] + [len(fs)]
            for a, b in zip(brk[:-1], brk[1:]):
                ang[a:b] = smooth_series(ang[a:b], self.pose_sigma)
                exp[a:b] = smooth_series(exp[a:b], self.exp_sigma)
                ts[a:b] = smooth_series(ts[a:b], self.pose_sigma)
            mot[tid] = (fs, ang, exp, ts)
        for tid, (c, rect, p203, lm5) in kcrop.items():
            seed = self._seed(video_path, tid)
            if self.realism == "v2":
                score = self._candidate_scorer(tid, rect, p203, lm5, mot[tid], crops[tid], rects[tid],
                                               tr[tid], kf[tid])
                b = kf[tid][1].box
                small = min(b[2] - b[0], b[3] - b[1]) < 96   # small faces: fewer candidates
                sd, sc, tries = self.generate_v2(c, seed, attrs.get(tid, {}), score,
                                                 max_tries=min(3, self.gen_tries) if small else None)
                self.gen_info[tid] = {"p75_guard_cos_sampled": None if sc is None else round(sc, 3),
                                      "tries": tries}
            else:
                sd = self.generate(c, seed)
            self._make_source(tid, sd, rect, p203, lm5)
        if self.realism == "v2" and self.gen_info:
            self._log("generation guard (p75 SFace cos over sampled re-enacted frames, best try): " +
                      ", ".join(f"{t}:{v['p75_guard_cos_sampled']}/{v['tries']}" for t, v in self.gen_info.items()))
        if self.pipe is not None or self.cpipe is not None:
            self.pipe = self.cpipe = self.depth = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        for tid in list(rects):
            if tid not in self.src:
                continue
            fs, ang, exp, ts = mot[tid]
            k = fs.index(kf[tid][0])
            Rall = E.rot(ang[:, 0], ang[:, 1], ang[:, 2]).cpu().numpy()
            R_ref = Rall[k]
            plan = {}
            exp_med, sc_med = np.median(exp, 0), float(np.median(ts[:, 3]))
            src = self.src[tid]["raw"]
            det_fs = np.array(sorted(tr[tid].dets))
            for i, g in enumerate(fs):
                rect, p203, ok = rects[tid][g]
                # padded frame on the other side of a shot cut from every detection:
                # there is no face here any more -> let the caller blur, don't paint one
                near = int(det_fs[np.abs(det_fs - g).argmin()])
                lo, hi = min(near, g), max(near, g)
                if any(lo < c <= hi for c in cuts):
                    plan[g] = dict(skip=True, ok=False)
                    continue
                if self.drive == "hybrid":
                    drv = self._hybrid_drv(src, Rall[i], exp[i], exp_med, ts[i], sc_med)
                elif self.drive == "absolute":
                    e = exp[i].copy()
                    if self.lip_gain != 1.0:  # amplify mouth motion around the track mean
                        lip = [6, 12, 14, 17, 19, 20]
                        e[lip] = exp[:, lip].mean(0) + self.lip_gain * (e[lip] - exp[:, lip].mean(0))
                    drv = dict(R=Rall[i], exp=e, t=ts[i, :3], scale=ts[i, 3], absolute=True)
                else:
                    drv = dict(R=Rall[i] @ R_ref.T, exp=exp[i] - exp[k], absolute=False)
                plan[g] = dict(rect=rect, drv=drv,
                               ok=bool(ok) and abs(ang[i, 1]) <= self.max_abs_yaw,
                               lmk=p203, yaw=float(ang[i, 1]))
            self.plan[tid] = plan
        self._log(f"re-enactment plan: {sum(len(p) for p in self.plan.values())} track-frames, "
                  f"{len(self.src)} synthetic identities")
        if self.realism == "v2":
            t0 = time.time()
            self._prerender_v2()
            self.render_seconds += time.time() - t0

    def _mask_v2(self, po, pr):
        """uint8 HxWx2 at 128 px: [..., 0] alpha mask, [..., 1] forehead band (the
        part above the brows that is only replaced where both faces show skin)."""
        q = LP_D // 4
        m = inner_face_mask(po / 4, pr / 4, q, forehead=self.mask_forehead, erode=self.mask_erode,
                            feather=self.mask_feather)
        # core = everything up to well above the brows (brows are identity: always replaced)
        core = cv2.bitwise_and(_hull_mask(po / 4, q, 0.15), _hull_mask(pr / 4, q, 0.15))
        core = cv2.dilate(core, np.ones((3, 3), np.uint8)).astype(np.float32) / 255
        band = m * (1 - cv2.GaussianBlur(core, (0, 0), 1.0))
        return (np.stack([m, band], -1) * 255 + 0.5).astype(np.uint8)

    def _hybrid_drv(self, src_raw, R, exp_i, exp_med, ts_i, sc_med):
        """pose / position absolute; expression and scale as deviations from the
        ORIGINAL's track median added to the SYNTHETIC face's own (removes the
        original's person-specific keypoint offsets)"""
        e = src_raw["exp"][0] + self.lip_gain * (exp_i - exp_med)
        return dict(R=R, exp=e, t=ts_i[:3], scale=float(src_raw["scale"][0]) * ts_i[3] / sc_med,
                    absolute=True)

    def _candidate_scorer(self, tid, rect_kf, p203_kf, lm5_kf, mot, crops256, rects_t, track, kf,
                          n_samples: int = 16):
        """Score(candidate SD image) = 75th percentile of the identity-guard cosine
        (SFace, both 5-pt alignments) between the original and the candidate
        re-enacted + pasted (v2 mask, colour matched) on up to `n_samples` frames
        of the track, in 256-px crop space.  (The per-frame guard handles the
        tail; the percentile picks the identity that trips it least often.)"""
        from .face_reenact.engine import m_o2c, tpts
        if not self._guard_ready():
            return None
        E = self.lp
        fs, ang, exp, ts = mot
        ok = [i for i, g in enumerate(fs) if abs(ang[i, 1]) <= self.max_abs_yaw and rects_t[g][2]]
        if not ok:
            return None
        pick = sorted(set(ok[j] for j in np.linspace(0, len(ok) - 1, min(n_samples, len(ok))).round().astype(int)))
        Rall = E.rot(ang[:, 0], ang[:, 1], ang[:, 2]).cpu().numpy()
        exp_med, sc_med = np.median(exp, 0), float(np.median(ts[:, 3]))
        S = 256
        samples = []
        for i in pick:
            g = fs[i]
            rect, p203 = rects_t[g][0], rects_t[g][1]
            M = m_o2c(rect, S)
            lm = track.dense[g][1] if g in track.dense and track.dense[g][1] is not None else kf[1].lm
            samples.append((i, crops256[i], tpts(p203, M), tpts(np.asarray(lm, np.float64), M)))
        box = [0, 0, S, S]

        def score(im):
            self._make_source(tid, im, rect_kf, p203_kf, lm5_kf)
            src = self.src[tid]
            cs = []
            for i, oc, po, lm5 in samples:
                out = cv2.resize(E.render(src, **self._hybrid_drv(src["raw"], Rall[i], exp[i], exp_med,
                                                                   ts[i], sc_med)), (S, S),
                                 interpolation=cv2.INTER_AREA)
                pr = E.landmarks203(out, prev203=po)
                if self.outline_align > 0:
                    A = (1 - self.outline_align) * np.float64([[1, 0, 0], [0, 1, 0]]) + self.outline_align * _fit_outline(pr, po)
                    out = cv2.warpAffine(out, A, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    pr = tpts(pr, A)
                mb = self._mask_v2(po * (LP_D / S), pr * (LP_D / S))
                mb = cv2.resize(mb, (S, S), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
                bl = self._composite(out, oc, mb[..., 0], mb[..., 1], ("eval", tid))
                cs.append(self._id_cos(oc, bl, box, lm5.astype(np.float32), {"lmk": po}))
            self._tex.pop(("eval", tid), None)
            return float(np.percentile(cs, 75))
        return score

    # ------------------------------------------------------------------ v2 pre-render
    def _prerender_v2(self):
        """Render every planned frame once, landmark the rendered face, align it to
        the original's inner face (similarity), build the inner-face mask, and
        smooth alignment + landmarks zero-phase over each contiguous run."""
        from .face_reenact.engine import m_o2c, tpts, smooth_series
        E = self.lp
        n_bad = 0
        for tid, plan in self.plan.items():
            fs = [g for g in sorted(plan) if plan[g].get("ok") and not plan[g].get("skip")]
            if not fs:
                continue
            PO, PR, okr = [], [], []
            for g in fs:
                st = plan[g]
                out = E.render(self.src[tid], **st["drv"])
                po = tpts(st["lmk"], m_o2c(st["rect"], LP_D))
                pr = E.landmarks203(out, prev203=po)
                iod_o = np.linalg.norm(po[EYE_L].mean(0) - po[EYE_R].mean(0))
                iod_r = np.linalg.norm(pr[EYE_L].mean(0) - pr[EYE_R].mean(0))
                okr.append(0.7 < iod_r / (iod_o + 1e-6) < 1.4 and
                           np.abs(pr[INNER] - po[INNER]).mean() < 0.5 * iod_o)
                st["img"] = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
                PO.append(po); PR.append(pr)
            PO, PR = np.stack(PO), np.stack(PR)
            okr = np.array(okr)
            n_bad += int((~okr).sum())
            PR[~okr] = PO[~okr]          # implausible render landmarks: trust the original's
            brk = [0] + [i for i in range(1, len(fs)) if fs[i] - fs[i - 1] > 1] + [len(fs)]
            prm = np.zeros((len(fs), 4))
            for i in range(len(fs)):
                A = _fit_outline(PR[i], PO[i])
                if A is None:
                    A = np.float64([[1, 0, 0], [0, 1, 0]])
                prm[i] = [A[0, 0], A[1, 0], A[0, 2], A[1, 2]]
            for a, b in zip(brk[:-1], brk[1:]):
                prm[a:b] = smooth_series(prm[a:b], self.pose_sigma)
                PR[a:b] = smooth_series(PR[a:b], 1.0)
            q = LP_D // 4
            for i, g in enumerate(fs):
                c, s, tx, ty = prm[i]
                A = np.array([[c, -s, tx], [s, c, ty]], np.float64)
                # outline alignment: the rendered head is scaled / moved so its jaw and
                # cheek outline matches the original's (fixes LivePortrait's size /
                # offset mismatch that showed as a doubled hairline and dark smear),
                # while eyes / nose / mouth keep the surrogate's own layout
                A = (1 - self.outline_align) * np.float64([[1, 0, 0], [0, 1, 0]]) + self.outline_align * A
                st = plan[g]
                st["A"], st["f"], st["po"], st["pr"] = A, g, PO[i], PR[i]
                st["mask_v2"] = self._mask_v2(PO[i], tpts(PR[i], A))
        self._log(f"v2 pre-render: {sum(len(p) for p in self.plan.values())} track-frames "
                  f"({n_bad} with implausible render landmarks)")

    def _make_source(self, tid, sd_crop, rect_kf, l203_kf, lm5_kf):
        """Canonicalise the SD face and build the LivePortrait source; also the
        TEMPLATE-aligned static surrogate used as per-frame fallback."""
        from .face_reenact.engine import m_o2c, rect_from_pts, tpts, inv2x3
        E = self.lp
        Mk = m_o2c(rect_kf, LP_D)
        seed203 = tpts(l203_kf, Mk)
        p = E.landmarks203(sd_crop, prev203=seed203)
        # sanity: SD face must sit roughly where the original was
        iod = np.linalg.norm(seed203[[0, 6, 12, 18]].mean(0) - seed203[[24, 30, 36, 42]].mean(0))
        if np.abs(p - seed203).mean() > 0.6 * iod:
            self._log(f"track {tid}: SD face misplaced; using keyframe landmarks for its crop")
            p = seed203
        r2 = rect_from_pts(p, LP_SCALE, LP_VY)
        canon = cv2.warpAffine(sd_crop, m_o2c(r2, LP_D), (LP_D, LP_D), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
        self.src[tid] = E.source(canon)
        self.src[tid]["img"] = canon
        lm5c = tpts(np.asarray(lm5_kf, np.float64), Mk).astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(lm5c, TEMPLATE, method=cv2.LMEDS)
        if M is not None:
            self.surr[tid] = cv2.warpAffine(sd_crop, M, (512, 512), flags=cv2.INTER_CUBIC,
                                            borderMode=cv2.BORDER_REFLECT)

    # ------------------------------------------------------------------ apply
    def _frame_of(self, track, box):
        f = self._fidx.get((track.tid, id(box)))
        if f is not None:
            return f
        for g, v in track.dense.items():
            if v[0] is box or np.array_equal(v[0], box):
                return g
        return None

    def apply(self, frame, track, box, lm) -> bool:
        if self.realism == "v2" and not self._v2_gate(box, lm):
            self.stats["none"] += 1
            return False          # fail closed: caller blurs
        self._guard_failed = False
        if self.motion == "reenact" and track.tid in self.plan:
            f = self._frame_of(track, box)
            step = self.plan[track.tid].get(f) if f is not None else None
            if step is not None and step.get("skip"):
                self.stats["none"] += 1
                return False
            if step is not None and step["ok"]:
                t0 = time.time()
                done = self._apply_reenact(frame, track.tid, step, box, lm)
                self.render_seconds += time.time() - t0
                if done:
                    self.stats["reenact"] += 1
                    return True
                if self.realism == "v2" and self._guard_failed:
                    # fail closed: too similar to the original even after covering more
                    # and damping the expression -> blur, never an unverified static warp
                    self.stats["none"] += 1
                    self.stats["guard_blur"] = self.stats.get("guard_blur", 0) + 1
                    return False
        ok = self._apply_static(frame, track, box, lm)
        self.stats["static" if ok else "none"] += 1
        return ok

    def _paste_reenact(self, frame, tid, step, drv) -> bool:
        from .face_reenact.engine import m_o2c, inv2x3, tpts, face_mask_from_lmk
        out = self.lp.render(self.src[tid], **drv)
        Mo2c = m_o2c(step["rect"], LP_D)
        Mc2o = inv2x3(Mo2c)
        if "mask" not in step:   # cached per frame (cheap: built at 128 px, upsampled)
            m = face_mask_from_lmk(tpts(step["lmk"], Mo2c) / 4, LP_D // 4, grow=self.mask_grow)
            step["mask"] = cv2.resize(m, (LP_D, LP_D), interpolation=cv2.INTER_LINEAR)
        mask = step["mask"]
        # colour: LAB statistics at 128 px in crop space, EMA over time
        small = LP_D // 4
        Ms = Mo2c / 4
        o_s = cv2.warpAffine(frame, Ms, (small, small), flags=cv2.INTER_AREA,
                             borderMode=cv2.BORDER_REPLICATE)
        sel = cv2.resize(mask, (small, small)) > 0.5
        if sel.sum() > 20:
            g_s = cv2.resize(out, (small, small), interpolation=cv2.INTER_AREA)
            out = _lab_transfer(out, o_s, sel, self._col.setdefault(tid, {}), src_small=g_s)
        # paste only the mask's bounding box
        ys, xs = np.nonzero(mask > 0.004)
        if len(xs) == 0:
            return False
        H, W = frame.shape[:2]
        corners = tpts(np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.min(), ys.max()],
                                 [xs.max(), ys.max()]], np.float64), Mc2o)
        X1, Y1 = np.floor(corners.min(0)).astype(int)
        X2, Y2 = np.ceil(corners.max(0)).astype(int) + 1
        X1, Y1, X2, Y2 = max(0, X1), max(0, Y1), min(W, X2), min(H, Y2)
        if X2 - X1 < 8 or Y2 - Y1 < 8:
            return False
        Mo = Mc2o.copy(); Mo[:, 2] -= [X1, Y1]
        ws = cv2.warpAffine(out, Mo, (X2 - X1, Y2 - Y1), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        wm = cv2.warpAffine(mask, Mo, (X2 - X1, Y2 - Y1))[..., None]
        roi = frame[Y1:Y2, X1:X2]
        roi[:] = (ws * wm + roi.astype(np.float32) * (1 - wm) + 0.5).astype(np.uint8)
        return True

    def _paste(self, frame, tid, step, drv) -> bool:
        if self.realism == "v2" and "mask_v2" in step:
            out = None
            if drv is not step["drv"] or "img" not in step:
                out = self.lp.render(self.src[tid], **drv)   # guard retry: live render
            return self._paste_v2(frame, tid, step, out)
        return self._paste_reenact(frame, tid, step, drv)

    def _paste_v2(self, frame, tid, step, out=None) -> bool:
        from .face_reenact.engine import m_o2c, inv2x3, tpts
        if out is None:
            out = cv2.imdecode(step["img"], cv2.IMREAD_COLOR)
        Mc2o = np.vstack([inv2x3(m_o2c(step["rect"], LP_D)), [0, 0, 1]])
        A = np.vstack([step["A"], [0, 0, 1]])
        Mr2o = Mc2o @ A                            # rendered crop -> frame
        q = LP_D // 4
        Mq2o = Mc2o @ np.diag([4.0, 4.0, 1.0])     # 128-px mask -> frame
        mq = step["mask_v2"][..., 0]
        bq = step["mask_v2"][..., 1]
        ys, xs = np.nonzero(mq > 2)
        if len(xs) == 0:
            return False
        H, W = frame.shape[:2]
        pad = 2
        corners = tpts(np.array([[xs.min() - pad, ys.min() - pad], [xs.max() + pad, ys.min() - pad],
                                 [xs.min() - pad, ys.max() + pad], [xs.max() + pad, ys.max() + pad]],
                                np.float64), Mq2o[:2])
        X1, Y1 = np.floor(corners.min(0)).astype(int)
        X2, Y2 = np.ceil(corners.max(0)).astype(int) + 1
        X1, Y1, X2, Y2 = max(0, X1), max(0, Y1), min(W, X2), min(H, Y2)
        if X2 - X1 < 8 or Y2 - Y1 < 8:
            return False
        sz = (int(X2 - X1), int(Y2 - Y1))
        T = np.array([[1, 0, -X1], [0, 1, -Y1], [0, 0, 1]], np.float64)
        ws = cv2.warpAffine(out, (T @ Mr2o)[:2], sz, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        m = cv2.warpAffine(mq.astype(np.float32) / 255, (T @ Mq2o)[:2], sz, flags=cv2.INTER_LINEAR)
        band = cv2.warpAffine(bq.astype(np.float32) / 255, (T @ Mq2o)[:2], sz, flags=cv2.INTER_LINEAR)
        roi = frame[Y1:Y2, X1:X2]
        roi[:] = self._composite(ws, roi, m, band, tid, self._frame_seed(tid, step))
        return True

    def _composite(self, ws, roi, m, band, tid, seed=0):
        """Appearance-match the surrogate `ws` to the original `roi` and blend it
        with alpha `m`; in the forehead `band` only where the surrogate shows skin
        (its hair is never pasted; the brows are outside the band)."""
        ws = self._match_appearance(ws, roi, m * (1 - band), tid, seed)
        if band.max() > 0.05 and self.band_gate != "none":
            core = (m > 0.5) & (band < 0.05)
            if core.sum() > 30:
                lo, ls = (cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32) for x in (roi, ws))
                ref = np.median(lo[core], 0)
                w = np.array([0.5, 1.0, 1.0], np.float32)
                sk = lambda L: np.exp(-0.5 * (np.sqrt((((L - ref) * w) ** 2).sum(-1)) / 14.0) ** 2)
                skin = sk(ls) if self.band_gate == "render" else np.minimum(sk(lo), sk(ls))
                skin = cv2.GaussianBlur(skin, (0, 0), 1.5)
                m = m * (1 - band + band * skin)
        if self.blend == "poisson":
            mb = (m > 0.5).astype(np.uint8) * 255
            ys2, xs2 = np.nonzero(mb)
            h, w_ = mb.shape
            if len(xs2) > 20 and xs2.min() > 0 and ys2.min() > 0 and xs2.max() < w_ - 1 and ys2.max() < h - 1:
                ctr = (int((xs2.min() + xs2.max() + 1) // 2), int((ys2.min() + ys2.max() + 1) // 2))
                try:
                    ws = cv2.seamlessClone(ws, roi.copy(), mb, ctr, cv2.NORMAL_CLONE)
                except cv2.error:
                    pass
        mm = m[..., None]
        return (ws.astype(np.float32) * mm + roi.astype(np.float32) * (1 - mm) + 0.5).astype(np.uint8)

    @staticmethod
    def _frame_seed(tid, step):
        return (int(tid) * 1000003 + int(step.get("f", 0)) * 7919) & 0x7FFFFFFF

    def _match_appearance(self, ws, roi, m, tid, seed=0):
        """Colour (LAB stats), low-frequency illumination and texture (sharpness +
        grain) of the surrogate `ws` matched to the original `roi` inside mask `m`."""
        sel = m > 0.5
        if sel.sum() < 30:
            return ws
        st = self._tex.setdefault(tid, {})
        # 1) colour: LAB mean fully, L contrast 80 %, chroma contrast fully (light EMA)
        a = cv2.cvtColor(ws, cv2.COLOR_BGR2LAB).astype(np.float32)
        b = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        s = np.stack([a[sel].mean(0), a[sel].std(0) + 1e-3, b[sel].mean(0), b[sel].std(0) + 1e-3])
        if "lab" in st:
            s = 0.5 * st["lab"] + 0.5 * s
        st["lab"] = s
        ma, sa, mb, sb = s
        tgt = sb * np.array([0.8, 1.0, 1.0]) + sa * np.array([0.2, 0.0, 0.0])
        a = (a - ma) / sa * tgt + mb
        # 2) illumination: low-frequency residual (normalised convolution in the mask)
        size = np.sqrt(sel.sum())
        sig = max(2.0, self.illum_sigma * size)
        k = max(1, int(sig / 3))
        h, w = m.shape
        ds = lambda x: cv2.resize(x, (max(4, w // k), max(4, h // k)), interpolation=cv2.INTER_AREA)
        ms = ds(m)
        den = cv2.GaussianBlur(ms, (0, 0), sig / k) + 1e-4
        lo_o = cv2.GaussianBlur(ds(b) * ms[..., None], (0, 0), sig / k) / den[..., None]
        lo_s = cv2.GaussianBlur(ds(a) * ms[..., None], (0, 0), sig / k) / den[..., None]
        corr = cv2.resize(lo_o - lo_s, (w, h), interpolation=cv2.INTER_LINEAR)
        a += self._illum * corr * np.array([1.0, 0.5, 0.5], np.float32)
        ws = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        # 3) texture: sharpness (blur down to the original's) + luma grain
        go = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gs = cv2.cvtColor(ws, cv2.COLOR_BGR2GRAY).astype(np.float32)
        eo, es = _hf_std(go, sel), _hf_std(gs, sel)
        if "eo" in st:
            eo, es = 0.7 * st["eo"] + 0.3 * eo, 0.7 * st["es"] + 0.3 * es
        st["eo"], st["es"] = eo, es
        if es > 1.15 * eo:
            if "sig" not in st or st.get("n", 0) % 15 == 0:
                best = (abs(es - eo), 0.0)
                for sg in (0.5, 0.75, 1.0, 1.4, 2.0, 2.8):
                    e = _hf_std(cv2.GaussianBlur(gs, (0, 0), sg), sel)
                    best = min(best, (abs(e - 0.9 * eo), sg))
                st["sig"] = best[1] if "sig" not in st else 0.5 * st["sig"] + 0.5 * best[1]
            if st["sig"] > 0.3:
                ws = cv2.GaussianBlur(ws, (0, 0), st["sig"])
                gs = cv2.GaussianBlur(gs, (0, 0), st["sig"])
        st["n"] = st.get("n", 0) + 1
        if self.grain:
            es2 = _hf_std(gs, sel)
            deficit = np.sqrt(max(0.0, eo ** 2 - es2 ** 2))
            if deficit > 0.3:
                rng = np.random.default_rng(seed + st["n"])
                n = cv2.GaussianBlur(rng.standard_normal(gs.shape).astype(np.float32), (0, 0), 0.6)
                n *= deficit / (_hf_std(n, np.ones_like(sel)) + 1e-6)
                ws = np.clip(ws.astype(np.float32) + n[..., None], 0, 255).astype(np.uint8)
        return ws

    def _apply_reenact(self, frame, tid, step, box=None, lm=None) -> bool:
        guard = self._guard_ready() and lm is not None
        if guard:
            orig = frame.copy()
        if not self._paste(frame, tid, step, step["drv"]):
            return False
        if not guard:
            return True
        cache = {}
        c = self._id_cos(orig, frame, box, lm, step, cache)
        if c <= self._thr_for(box):
            return True
        # identity guard tripped: retry with damped expression, then give up on re-enactment
        self.stats["guard_retry"] = self.stats.get("guard_retry", 0) + 1
        frame[:] = orig
        if self.realism == "v2" and "po" in step:
            # v2: first cover more of the original (outline / forehead), same motion
            from .face_reenact.engine import tpts
            step = dict(step)
            e, fh = self.mask_erode, self.mask_forehead
            self.mask_erode, self.mask_forehead = e - 0.05, fh + 0.15
            step["mask_v2"] = self._mask_v2(step["po"], tpts(step["pr"], step["A"]))
            self.mask_erode, self.mask_forehead = e, fh
            if self._paste_v2(frame, tid, step) and \
                    self._id_cos(orig, frame, box, lm, step, cache) <= self._thr_for(box):
                self.stats["guard_grow"] = self.stats.get("guard_grow", 0) + 1
                return True
            frame[:] = orig
        d = dict(step["drv"])
        if d.get("absolute") and self.drive == "hybrid":
            e0 = self.src[tid]["raw"]["exp"][0]
            for k in (0.5, 0.0):     # damp expression; 0.0 = synthetic face's own expression
                d["exp"] = e0 + k * (step["drv"]["exp"] - e0)
                if self._paste(frame, tid, step, d) and \
                        self._id_cos(orig, frame, box, lm, step, cache) <= self._thr_for(box):
                    return True
                frame[:] = orig
        self._guard_failed = True
        if self.realism != "v2":
            self.stats["guard_static"] = self.stats.get("guard_static", 0) + 1
        return False   # v1: caller falls back to the static surrogate; v2: blur

    # ---------------- identity guard (OpenCV-zoo SFace, Apache-2.0; ONNX Runtime)
    def _guard_ready(self) -> bool:
        if self.guard_thr is None:
            return False
        if self._rec is None:
            p = os.path.expanduser(os.environ.get(
                "CUSTODIAN_SFACE", "~/.cache/custodian_video/face_recognition_sface_2021dec.onnx"))
            if not os.path.exists(p):
                self._log(f"identity guard disabled ({p} missing)")
                self.guard_thr = None
                return False
            import onnxruntime as ort
            so = ort.SessionOptions(); so.log_severity_level = 3
            prov = ["CPUExecutionProvider"]
            if self.lp is not None and self.lp.device == "cuda" and \
                    "CUDAExecutionProvider" in ort.get_available_providers():
                prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._rec = ort.InferenceSession(p, sess_options=so, providers=prov)
            self._rec_in = self._rec.get_inputs()[0].name
            self._rec_cv = cv2.FaceRecognizerSF.create(p, "")   # only for alignCrop
        return True

    def _emb(self, img, row):
        al = self._rec_cv.alignCrop(img, row)
        x = np.ascontiguousarray(al[:, :, ::-1].transpose(2, 0, 1)[None], np.float32)
        e = self._rec.run(None, {self._rec_in: x})[0][0]
        return e / (np.linalg.norm(e) + 1e-9)

    def _guard_rows(self, box, lm, step, img=None):
        """5-point alignments: the detector's points, points derived from the
        dense 203 landmarks (eye centres, detector nose, mouth corners) and, in v2,
        YuNet's own points on the ORIGINAL around the box (the alignment an
        attacker / the evaluation uses)."""
        x1, y1, x2, y2 = [float(v) for v in box]
        lm = np.asarray(lm, np.float32).reshape(5, 2)
        rows = [lm]
        p = step.get("lmk")
        if p is not None:
            rows.append(np.stack([p[[0, 6, 12, 18]].mean(0), p[[24, 30, 36, 42]].mean(0),
                                  lm[2], p[48], p[66]]).astype(np.float32))
        if img is not None and self.realism == "v2":
            y = self._yunet_lm(img, box)
            if y is not None:
                rows.append(y)
        return [np.array([[x1, y1, x2 - x1, y2 - y1, *r.reshape(-1), 1.0]], np.float32) for r in rows]

    def _yunet_lm(self, img, box):
        """YuNet (OpenCV zoo, MIT) 5 points of the face inside `box` on `img`, or None."""
        p = os.path.expanduser(os.environ.get(
            "CUSTODIAN_YUNET", "~/.cache/custodian_video/face_detection_yunet_2023mar.onnx"))
        if not os.path.exists(p):
            return None
        H, W = img.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        bw, bh = x2 - x1, y2 - y1
        X1, Y1 = int(max(0, x1 - 0.5 * bw)), int(max(0, y1 - 0.5 * bh))
        X2, Y2 = int(min(W, x2 + 0.5 * bw)), int(min(H, y2 + 0.5 * bh))
        if X2 - X1 < 16 or Y2 - Y1 < 16:
            return None
        crop = img[Y1:Y2, X1:X2]
        k = max(1.0, 160.0 / max(crop.shape[:2]))    # small faces: upscale for the detector
        crop = cv2.resize(crop, None, fx=k, fy=k) if k > 1 else crop
        if getattr(self, "_yn", None) is None:
            self._yn = cv2.FaceDetectorYN.create(p, "", (crop.shape[1], crop.shape[0]), 0.5, 0.3, 50)
        self._yn.setInputSize((crop.shape[1], crop.shape[0]))
        _, f = self._yn.detect(crop)
        if f is None or not len(f):
            return None
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        f = min(f, key=lambda r: (X1 + (r[0] + r[2] / 2) / k - cx) ** 2 + (Y1 + (r[1] + r[3] / 2) / k - cy) ** 2)
        return (f[4:14].reshape(5, 2) / k + [X1, Y1]).astype(np.float32)

    def _thr_for(self, box) -> float:
        """v2: stricter guard for small faces (recognisers are less reliable there,
        so keep a larger margin to the 0.363 match threshold)."""
        if self.realism == "v2" and min(box[2] - box[0], box[3] - box[1]) < 120:
            return min(self.guard_thr, 0.25)
        return self.guard_thr

    def _id_cos(self, a, b, box, lm, step=None, cache=None) -> float:
        if cache is not None and "rows" in cache:
            rows = cache["rows"]
        else:
            rows = self._guard_rows(box, lm, step or {}, a)
            if cache is not None:
                cache["rows"] = rows
        if cache is not None and "orig" in cache:
            ea = cache["orig"]
        else:
            ea = [self._emb(a, r) for r in rows]
            if cache is not None:
                cache["orig"] = ea
        return max(float(e @ self._emb(b, r)) for e, r in zip(ea, rows))

    def _apply_static(self, frame, track, box, lm) -> bool:
        s = self.surr.get(track.tid)
        if s is None or lm is None or min(box[2] - box[0], box[3] - box[1]) < self.min_face * 0.6:
            return False
        M, _ = cv2.estimateAffinePartial2D(TEMPLATE, lm.astype(np.float32), method=cv2.LMEDS)
        if M is None:
            return False
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        bw, bh = x2 - x1, y2 - y1
        X1, Y1 = max(0, x1 - bw), max(0, y1 - bh)
        X2, Y2 = min(W, x2 + bw), min(H, y2 + bh)
        if X2 - X1 < 4 or Y2 - Y1 < 4:
            return False
        Mo = M.copy(); Mo[:, 2] -= [X1, Y1]
        ws = cv2.warpAffine(s, Mo, (X2 - X1, Y2 - Y1), flags=cv2.INTER_LINEAR)
        mask = self.mask
        if self.realism == "v2" and min(bw, bh) < 96:
            mask = self._mask_full    # small face: cover the whole face (hair edge is not visible)
        wm = cv2.warpAffine(mask, Mo, (X2 - X1, Y2 - Y1))[..., None]
        roi = frame[Y1:Y2, X1:X2]
        guard = self.realism == "v2" and self._guard_ready()
        if guard:
            before = roi.copy()
        # LAB colour transfer: match surrogate stats to the current frame's face
        sel = wm[..., 0] > 0.5
        if self.realism == "v2":
            ws = self._match_appearance(ws, roi, wm[..., 0], track.tid,
                                        (int(track.tid) * 1000003 + int(x1 * 7 + y1)) & 0x7FFFFFFF)
        elif sel.sum() > 20:
            ws = _lab_transfer(ws, roi, sel)
        roi[:] = (ws * wm + roi * (1 - wm)).astype(np.uint8)
        if guard:
            # v2: the static fallback is identity-guarded too (v1 had no check here)
            Xo = np.array([X1, Y1], np.float32)
            bx = np.array([box[0] - X1, box[1] - Y1, box[2] - X1, box[3] - Y1], np.float32)
            if self._id_cos(before, roi, bx, np.asarray(lm, np.float32).reshape(5, 2) - Xo) > self._thr_for(bx):
                roi[:] = before
                self.stats["static_guard_blur"] = self.stats.get("static_guard_blur", 0) + 1
                return False
        return True
