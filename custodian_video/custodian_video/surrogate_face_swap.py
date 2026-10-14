"""RESEARCH-ONLY face anonymisation backend (non-commercial licences):
synthetic-identity face swap.  Opt-in (`--face-backend research-swap`); the
default `sd15-keyframe` backend is unchanged.

Every face TRACK gets one SYNTHETIC identity (never a real person):

  1. coarse attributes of the track's keyframe face (age band, gender
     presentation, skin tone, hair, facial hair, glasses) from the local
     Qwen3-VL (face_attributes.py, Apache-2.0), kept in memory only;
  2. TEXT-to-image renders `n_cand` portraits of people who do not exist from
     that attribute prompt plus seed-chosen facial-feature words (nothing of the
     original face is given to the generator).  Default generator (since
     2026-10-08, experiment face_h): RealVisXL V5.0, an SDXL photoreal
     fine-tune (CreativeML OpenRAIL++-M), 16 candidates at 1024 px, every 2nd
     one without the apparent-ancestry word (more diverse identities);
     `generator="sd15"` restores Stable Diffusion 1.5 (CreativeML OpenRAIL-M).
     The local Qwen3-VL then reads every candidate portrait and REJECTS those
     whose apparent gender presentation, age band (+-1) or skin tone (+-1 step)
     differ from the person's, or with major artefacts (one more batch if fewer
     than 2 match; none -> the track is blurred), and ranks the rest side by
     side by how natural and good-looking they are (round robin of pairwise
     comparisons, each pair shown in both orders).  The guard-based choice below only picks among these;
  3. the synthetic portrait's identity embedding (InsightFace ArcFace
     w600k_r50, buffalo_l: non-commercial research licence) drives a face
     swapper on every frame of the track: HyperSwap 1a/1b/1c 256 (FaceFusion,
     ResearchRAIL) or InsightFace inswapper_128 (non-commercial), with
     FaceFusion's "pixel boost" (the aligned crop is swapped as k x k
     sub-sampled 256 px tiles) so a large face is rendered at 512 px.  Pose,
     expression, lighting, skin tone and the head's outline come from the
     frame; the identity (eyes / nose / mouth shape, proportions, skin detail)
     from the synthetic portrait.  `push` > 0 moves the source embedding away
     from the original's (FaceFusion's swapper weight);
  4. paste-back mask (crop space) = feathered box  x  BiSeNet face parsing of
     the ORIGINAL crop (FaceFusion bisenet_resnet_34, MIT; CelebAMask-HQ
     classes: skin, brows, eyes, nose, mouth, lips, optionally glasses)  x
     XSeg occlusion mask (FaceFusion xseg_3, GPL-3.0: hands, microphones, hair
     strands in front of the face stay original)  x  the swapper's own mask
     (HyperSwap).  The original's hair, ears, neck and background are never
     replaced, so no ghost hair / background bleed;
  5. optional face restoration of the swapped face (GFPGAN 1.4, Apache-2.0
     code with NVIDIA StyleGAN2 ops; CodeFormer, S-Lab 1.0 non-commercial;
     GPEN-BFR, non-commercial) blended at `enhance_blend`;
  6. motion: the swap runs per frame on the ORIGINAL frame (no re-enactment),
     aligned by YuNet's 5 points (MIT) on that frame, so pose / gaze / blinks
     come from the original.  The swapper (esp. with `push`) tends to close the
     mouth, so by default (`mouth_keep=1`) the mouth and lips (BiSeNet classes
     11-13, slightly dilated) stay the ORIGINAL's: speech is followed exactly.
     Optional zero-phase smoothing of the landmarks (`lm_sigma`) and of masks /
     crops over +-`t_sigma` frames exists but is off by default: it cost blink
     fidelity on the demo clips (experiments/face_g).
  7. guard trips are consolidated into contiguous blur segments (+-4 frames,
     gaps <= 15 frames filled; a track tripping on > 50 % of its frames is
     blurred entirely), so a face does not flicker between swap and blur.

Stability (`stabilize=True`, default since 2026-10-07; False = configuration R):
identity + render level per track chosen by the guard-trip rate on a dense
sample (since face_h: among the candidates passing at a level, the best-ranked
looking one; levels push +0 / +0.3 / +0.6 / +0.9); a tripping frame is retried with a stronger push before it is blurred;
a track still tripping on > `track_blur_trip` of the sample is blurred entirely;
blur segments are padded / merged, short swap runs next to blur are blurred,
swap <-> blur is cross-faded over `fade` frames; landmarks are smoothed with a
zero-phase one-euro filter.

Defaults (apart from the stability block) = configuration "R" of experiments/face_g (hyperswap_1b_256, push 0.5,
mouth_keep 1, 8 candidates, no temporal smoothing).  It did NOT meet the bar for
the public demo page (see experiments/face_g/summary.md): it is research-only
and opt-in.

Privacy (fail closed, like realism v2): a face gets a swap only if its box is
>= 1.25 x v2_min_face, it has landmarks and the SFace guard (OpenCV zoo,
Apache-2.0) is available; every rendered frame is checked against the
original with SFace at the evaluator's YuNet alignment (thr 0.30, 0.25 under
120 px) and BLURRED if it trips.  The synthetic identity is chosen among the
candidates by the lowest 75th-percentile SFace similarity to the original on
sampled frames (the ArcFace attacker used in evaluation is NOT used for
selection; it is, however, the swapper's own identity space, so the eval
also reports an independent recogniser).

Model files live in $CUSTODIAN_RESEARCH_FACES (default
~/.cache/custodian_video/research_faces; FaceFusion asset releases) and
$CUSTODIAN_ARCFACE (default ~/.insightface/models/buffalo_l/w600k_r50.onnx).
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from .surrogate_face import SDKeyframeSurrogate, feature_descriptors

RESEARCH_DIR = os.path.expanduser(os.environ.get("CUSTODIAN_RESEARCH_FACES",
                                                 "~/.cache/custodian_video/research_faces"))
ARCFACE = os.path.expanduser(os.environ.get("CUSTODIAN_ARCFACE",
                                            "~/.insightface/models/buffalo_l/w600k_r50.onnx"))

TEMPLATES = {
    "arcface_112_v2": np.array([[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.50022500, 0.64050536],
                                [0.37097589, 0.82469196], [0.63151696, 0.82325089]], np.float32),
    "arcface_128": np.array([[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
                             [0.38710391, 0.72160547], [0.61507734, 0.72034453]], np.float32),
    "ffhq_512": np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                          [0.39308822, 0.72541100], [0.61150205, 0.72490465]], np.float32),
}

LICENCES = {
    "hyperswap": "HyperSwap 256 (FaceFusion, ResearchRAIL - research only)",
    "inswapper_128": "InsightFace inswapper_128 (non-commercial research only)",
    "arcface": "InsightFace ArcFace w600k_r50 / buffalo_l (non-commercial research only)",
    "bisenet": "BiSeNet face parsing, FaceFusion bisenet_resnet_34 (MIT; trained on CelebAMask-HQ, "
               "non-commercial dataset)",
    "xseg": "XSeg occlusion mask, FaceFusion xseg_3 (GPL-3.0)",
    "gfpgan_1.4": "GFPGAN 1.4 (Apache-2.0; NVIDIA StyleGAN2 ops)",
    "codeformer": "CodeFormer (S-Lab License 1.0, non-commercial)",
    "gpen_bfr_512": "GPEN-BFR 512 (non-commercial)",
    "restoreformer_plus_plus": "RestoreFormer++ (Apache-2.0)",
    "sd15": "Stable Diffusion 1.5 text-to-image (CreativeML OpenRAIL-M)",
    "realvisxl": "RealVisXL V5.0 (SDXL photoreal fine-tune, CreativeML OpenRAIL++-M) + SDXL-VAE fp16-fix (MIT)",
    "qwen3vl": "Qwen3-VL-8B attributes (Apache-2.0)",
    "sface_yunet": "SFace guard + YuNet landmarks (OpenCV zoo, Apache-2.0 / MIT)",
}

PARSE_FACE = [1, 2, 3, 4, 5, 10, 11, 12, 13]   # skin, brows, eyes, nose, mouth, lips
PARSE_GLASSES = 6


def _session(path, device):
    if device != "cpu":
        try:
            import torch  # noqa: F401  (loads the cuDNN 9 / CUDA 12 libraries ONNX Runtime's CUDA EP needs)
        except Exception:
            pass
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    prov = ["CPUExecutionProvider"]
    if device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers():
        prov = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=so, providers=prov)


def align_matrix(lm5, template, size):
    M, _ = cv2.estimateAffinePartial2D(np.asarray(lm5, np.float32), TEMPLATES[template] * size, method=cv2.LMEDS)
    return M


def _gauss_smooth(x, sigma):
    """zero-phase Gaussian smoothing along axis 0 (edge-padded)."""
    if sigma <= 0 or len(x) < 3:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    xp = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)], 0)
    return np.stack([np.tensordot(k, xp[i:i + 2 * r + 1], axes=(0, 0)) for i in range(len(x))])


class SwapNets:
    """ONNX Runtime wrappers for the research-only face models."""

    def __init__(self, device="cuda", swapper="hyperswap_1b_256", enhancer=None, occluder=True):
        self.device = device
        self.swapper_name = swapper
        self.arc = _session(ARCFACE, device)
        self.sw = _session(os.path.join(RESEARCH_DIR, f"{swapper}.onnx"), device)
        self.kind = "inswapper" if swapper.startswith("inswapper") else "hyperswap"
        self.size = 128 if self.kind == "inswapper" else 256
        if self.kind == "inswapper":
            import onnx
            from onnx import numpy_helper
            m = onnx.load(os.path.join(RESEARCH_DIR, f"{swapper}.onnx"))
            self.emap = numpy_helper.to_array(m.graph.initializer[-1])
        self.parser = _session(os.path.join(RESEARCH_DIR, "bisenet_resnet_34.onnx"), device)
        self.occl = _session(os.path.join(RESEARCH_DIR, "xseg_3.onnx"), device) if occluder else None
        self.enh = _session(os.path.join(RESEARCH_DIR, f"{enhancer}.onnx"), device) if enhancer else None
        self.enhancer_name = enhancer

    # identity embedding (raw, un-normalised) -------------------------------
    def embed(self, img, lm5):
        M = align_matrix(lm5, "arcface_112_v2", 112)
        c = cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        x = ((c[:, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        return self.arc.run(None, {self.arc.get_inputs()[0].name: x})[0][0]

    def source_vec(self, emb, target_emb=None, push=0.0):
        e = emb / np.linalg.norm(emb)
        if target_emb is not None and push:
            t = target_emb / np.linalg.norm(target_emb)
            e = e * (1 + push) - t * push
        if self.kind == "inswapper":
            v = emb.reshape(1, -1) if not push else (e * np.linalg.norm(emb)).reshape(1, -1)
            return (v @ self.emap / np.linalg.norm(v)).astype(np.float32)
        return e.reshape(1, -1).astype(np.float32)

    # swap one aligned crop (any size = k * model size) ----------------------
    def swap(self, crop, src):
        n = self.size; k = crop.shape[0] // n
        out = np.zeros(crop.shape, np.float32); msk = np.zeros(crop.shape[:2], np.float32)
        for i in range(k):
            for j in range(k):
                sub = crop[i::k, j::k]
                x = sub[:, :, ::-1].astype(np.float32) / 255.0
                if self.kind == "hyperswap":
                    x = (x - 0.5) / 0.5
                x = x.transpose(2, 0, 1)[None]
                r = self.sw.run(None, {"source": src, "target": x})
                o = r[0][0].transpose(1, 2, 0)
                if self.kind == "hyperswap":
                    o = o * 0.5 + 0.5
                out[i::k, j::k] = o.clip(0, 1)[:, :, ::-1] * 255
                msk[i::k, j::k] = r[1][0, 0].clip(0, 1) if len(r) > 1 else 1.0
        return out, msk

    def parse(self, crop):
        x = cv2.resize(crop, (512, 512))[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32).transpose(2, 0, 1)[None]
        return self.parser.run(None, {"input": x})[0][0].argmax(0).astype(np.uint8)

    def occlusion(self, crop):
        if self.occl is None:
            return np.ones(crop.shape[:2], np.float32)
        x = cv2.resize(crop, (256, 256))[None].astype(np.float32) / 255.0
        m = self.occl.run(None, {"input": x})[0][0, :, :, 0].clip(0, 1)
        return cv2.resize(m, crop.shape[:2][::-1])

    def enhance(self, crop512, weight=0.5):
        x = ((crop512[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        feed = {"input": x}
        if any(i.name == "weight" for i in self.enh.get_inputs()):
            feed["weight"] = np.array([weight], np.float64)
        o = self.enh.run(None, feed)[0][0].clip(-1, 1)
        return ((o.transpose(1, 2, 0) + 1) / 2 * 255)[:, :, ::-1]


class ResearchSwapSurrogate(SDKeyframeSurrogate):
    """Opt-in research backend (see module docstring)."""

    def __init__(self, device: str = "cuda", min_face: int = 64, min_score: float = 0.6,
                 swapper: str | None = None, enhancer: str | None = None, enhance_blend: float = 0.5,
                 boost: int = 2, push: float = 0.5, n_cand: int = 16, glasses: bool = True,
                 t_sigma: float = 0.0, lm_sigma: float = 0.0, mask_erode: float = 0.02,
                 mask_blur: float = 0.025, guard_thr: float | None = 0.30, n_score: int = 12,
                 sd_steps: int = 30, seed_salt: str = "custodian-research", verbose: bool = True,
                 swap_mask: bool = True, parse_swap: bool = True, iters: int = 1,
                 mouth_keep: float = 1.0, img_smooth: bool = False, eye_keep: float = 0.0, brow_keep: float = 0.0,
                 block_pad: int = 4, block_gap: int = 15, block_track: float = 0.5,
                 stabilize: bool = True, n_dense: int = 48, retry_push: tuple = (0.3, 0.6),
                 track_levels: tuple = ((0.0, None), (0.3, None), (0.6, None), (0.9, None)),
                 level_ok: float = 0.10, track_blur_trip: float = 0.25, min_swap_run: int = 12, fade: int = 4,
                 euro: tuple | None = (1.5, 0.01), generator: str | None = None, verify: bool = True,
                 n_quality: int = 12, p75_margin: float = 0.03, p75_target: float = 0.18, **kw):
        super().__init__(device, min_face=min_face, min_score=min_score, seed_salt=seed_salt,
                         realism="v2", guard_thr=guard_thr, verbose=verbose, motion="reenact")
        self.swapper = swapper or os.environ.get("CUSTODIAN_RESEARCH_SWAPPER", "hyperswap_1b_256")
        self.enhancer = enhancer if enhancer is not None else (os.environ.get("CUSTODIAN_RESEARCH_ENHANCER") or None)
        self.enhance_blend, self.boost, self.push, self.n_cand = enhance_blend, boost, push, n_cand
        self.glasses, self.t_sigma, self.lm_sigma = glasses, t_sigma, lm_sigma
        self.m_erode, self.m_blur, self.n_score, self.sd_steps = mask_erode, mask_blur, n_score, sd_steps
        self.swap_mask, self.parse_swap, self.iters = swap_mask, parse_swap, iters
        self.mouth_keep, self.img_smooth, self.eye_keep, self.brow_keep = mouth_keep, img_smooth, eye_keep, brow_keep
        self.block_pad, self.block_gap, self.block_track = block_pad, block_gap, block_track
        # stabilize (default): identity + render level chosen by guard trips on a dense sample,
        # per-frame retries with a stronger push, swap/blur cross-fades, no short swap runs,
        # one-euro smoothed landmarks.  stabilize=False = configuration "R" of face_g.
        self.stabilize, self.n_dense, self.retry_push = stabilize, n_dense, tuple(retry_push)
        self.track_levels, self.level_ok = [tuple(x) for x in track_levels], level_ok
        self.min_swap_run, self.fade, self.euro = min_swap_run, fade, euro
        # a track whose best identity still trips the guard on more than this share of the
        # dense sample is blurred ENTIRELY (one consistent blur instead of swap/blur toggling)
        self.track_blur_trip = track_blur_trip
        # synthetic identity generator: "realvisxl" (SDXL photoreal, default since 2026-10-08) | "sd15"
        self.generator = generator or os.environ.get("CUSTODIAN_RESEARCH_GENERATOR", "realvisxl")
        # Qwen3-VL check of every candidate portrait: attribute match (gender presentation,
        # age band, skin tone) BEFORE ranking; then quality (realism + appeal), then guard trips
        self.verify, self.n_quality, self.p75_margin = verify, n_quality, p75_margin
        # a level is accepted only if the pick's 75th-percentile SFace similarity on the dense sample is
        # <= p75_target (attribute-matched photoreal identities sit closer to the person than the old
        # SD-1.5 ones; 0.18 ~ the previous clip-4 pick), else the next, stronger level is tried
        self.p75_target = p75_target
        self.cand_quality: dict[int, dict] = {}   # tid -> {candidate index: quality}
        self.cand_info: dict[int, dict] = {}      # tid -> summary (no attributes of the real face)
        self.weight: dict[tuple, float] = {}   # (tid, frame) -> swap weight (cross-fade to blur)
        self.frame_state: dict[int, dict] = {}  # tid -> {frame: weight}, for evaluation
        self._mk = mouth_keep
        self.nets = None
        self.out: dict[tuple, tuple] = {}    # (tid, frame) -> (M, swapped crop uint8, alpha float16)
        self.surr: dict[int, np.ndarray] = {}  # tid -> synthetic portrait (for the log / never written)
        self.gen_info: dict[int, dict] = {}
        lic = [LICENCES["hyperswap" if self.swapper.startswith("hyperswap") else "inswapper_128"],
               LICENCES["arcface"], LICENCES["bisenet"], LICENCES["xseg"],
               LICENCES["realvisxl" if self.generator == "realvisxl" else "sd15"], LICENCES["qwen3vl"],
               LICENCES["sface_yunet"]]
        if self.enhancer:
            lic.append(LICENCES.get(self.enhancer, self.enhancer))
        self.name = (f"research-swap ({self.swapper}; RESEARCH-ONLY models, non-commercial licences: "
                     + "; ".join(lic) + ")")
        self.licences = lic

    # --------------------------------------------------------------- identity
    def _sd_txt2img(self):
        import torch
        from diffusers import AutoencoderKL, StableDiffusionPipeline
        dev = "cuda" if self.device != "cpu" and torch.cuda.is_available() else "cpu"
        dt = torch.float16 if dev == "cuda" else torch.float32
        if self.generator == "realvisxl":
            from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dt)
            p = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V5.0", vae=vae, torch_dtype=dt,
                                                          variant="fp16" if dt == torch.float16 else None).to(dev)
            p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config, use_karras_sigmas=True)
            p.set_progress_bar_config(disable=True)
            return p, dev
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dt)
        p = StableDiffusionPipeline.from_pretrained(self.model_id, vae=vae, torch_dtype=dt, safety_checker=None,
                                                    requires_safety_checker=False).to(dev)
        p.set_progress_bar_config(disable=True)
        return p, dev

    def _portraits(self, attrs, seed, n):
        """n synthetic portraits (people who do not exist) for one track:
        [(bgr 512, 5-point landmarks)] -- candidates without a detectable
        frontal face are dropped."""
        import torch
        from PIL import Image  # noqa: F401
        from .face_attributes import prompt_from_attributes, prompt_sdxl
        if self.generator == "realvisxl":
            res = []
            for i in range(n):
                # every 2nd candidate without the apparent-ancestry word: more diverse identities
                # (further from the person in face-recogniser space); gender presentation, age band and
                # skin tone stay in the prompt and are verified for every candidate
                pos, neg = prompt_sdxl(attrs or {}, feature_descriptors(seed + i), appearance=i % 2 == 0)
                g = torch.Generator(device=self._gen_dev).manual_seed((seed + 7919 * i) % 2**31)
                im = self.t2i(prompt=pos, negative_prompt=neg,
                              num_inference_steps=self.sd_steps, guidance_scale=5.0, height=1024, width=1024,
                              generator=g).images[0]
                bgr = cv2.resize(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR), (512, 512),
                                 interpolation=cv2.INTER_AREA)
                lm = self._yunet_lm(bgr, [96, 96, 416, 416])
                if lm is not None:
                    res.append((bgr, lm))
            return res
        pos, neg = prompt_from_attributes(attrs or {})
        pos = pos.replace("relaxed pleasant expression, mouth closed",
                          "front-facing head and shoulders portrait, looking straight at the camera, "
                          "neutral expression, mouth closed, hair pulled back, clear forehead, even soft light, "
                          "plain background")
        neg += (", profile, side view, tilted head, hands, hat, sunglasses, occluded face, multiple people, "
                "bangs, fringe, hair over forehead, hair over face")
        res = []
        for i in range(n):
            g = torch.Generator(device=self._gen_dev).manual_seed((seed + 7919 * i) % 2**31)
            im = self.t2i(prompt=f"{pos}, {feature_descriptors(seed + i)}", negative_prompt=neg,
                          num_inference_steps=self.sd_steps, guidance_scale=6.5, height=512, width=512,
                          generator=g).images[0]
            bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
            lm = self._yunet_lm(bgr, [96, 96, 416, 416])
            if lm is not None:
                res.append((bgr, lm))
        return res

    def _verify_candidates(self, video_path, todo, attrs, ports):
        """Qwen3-VL reads every candidate portrait; candidates whose apparent gender
        presentation, age band or skin tone does not match the person (or with major
        artefacts) are REJECTED before any ranking.  If fewer than 2 match, one more
        batch (other seeds) is generated.  A track with attributes but no matching
        candidate keeps only same-gender candidates, or none (-> whole track blurred).
        The VLM unavailable -> candidates are kept unverified (logged)."""
        import gc
        from .face_attributes import FaceAttributes, attribute_mismatch, quality_score
        t_gen = self.t2i
        out = {}
        for rnd in range(2):
            self.t2i = None
            try:   # free the GPU for the VLM worker: the text-to-image pipeline waits on the CPU
                t_gen.to("cpu")
            except Exception:
                pass
            gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
            crops = {f"{tid}_{ci}": bgr for tid, L in ports.items() for ci, (bgr, _) in enumerate(L)}
            try:
                ver = FaceAttributes().verify(crops)
            except Exception as e:  # noqa: BLE001
                ver = {}
                self._log(f"candidate check unavailable ({type(e).__name__}); candidates kept unverified")
            if not any(ver.values()):
                if rnd == 0:
                    # fail closed on appearance: an unverified identity could be a mismatch (e.g. another
                    # gender presentation), so tracks whose attributes are known get no synthetic face
                    self._log("candidate check unavailable -> tracks with attributes are blurred "
                              "(no unverified synthetic identity)")
                    for t in todo:
                        if attrs.get(t.tid):
                            self.cand_info.setdefault(t.tid, {})["check_unavailable"] = True
                    self.t2i = t_gen
                    try:
                        t_gen.to(self._gen_dev)
                    except Exception:
                        pass
                    return {t.tid: ([] if attrs.get(t.tid) else ports.get(t.tid, [])) for t in todo}
                self._log("candidate check (round 1) returned nothing; round-0 matches kept")
                break
            more = {}
            for t in todo:
                tid, a = t.tid, attrs.get(t.tid, {})
                L = ports.get(tid, [])
                keep, q, why = [], [], {}
                for ci, c in enumerate(L):
                    v = ver.get(f"{tid}_{ci}", {})
                    bad = attribute_mismatch(a, v) if a else []
                    for b in bad:
                        why[b] = why.get(b, 0) + 1
                    if not bad:
                        keep.append(c); q.append(quality_score(v))
                prev = out.get(tid, ([], []))
                keep, q = prev[0] + keep, prev[1] + q
                out[tid] = (keep, q)
                info = self.cand_info.setdefault(tid, {"generated": 0, "rejected": {}})
                info["generated"] += len(L)
                for b, k in why.items():
                    info["rejected"][b] = info["rejected"].get(b, 0) + k
                self._log(f"track {tid}: candidate check round {rnd}: {len(keep) - len(prev[0])}/{len(L)} match "
                          f"the person's attributes (rejection reasons: {why or 'none'})")
                if len(keep) < 2 and rnd == 0 and a:
                    more[tid] = a
            if not more:
                break
            # second batch for tracks with < 2 matching candidates (other seeds)
            self.t2i = t_gen
            try:
                t_gen.to(self._gen_dev)
            except Exception:
                pass
            ports = {tid: self._portraits(a, self._seed(video_path, tid) + 104729, self.n_cand)
                     for tid, a in more.items()}
        self.t2i = t_gen
        # quality: Qwen3-VL compares the matching candidates of a track in pairs, each pair in
        # both orders (A/B swapped) so the judge's position bias cancels; score = wins
        pairs, n_top = {}, 8
        for tid, (keep, q) in out.items():
            if len(keep) >= 2:
                top = sorted(range(len(keep)), key=lambda i: (-q[i], i))[:n_top]
                tiles = {i: cv2.resize(keep[i][0], (384, 384), interpolation=cv2.INTER_AREA) for i in top}
                for x in range(len(top)):
                    for y in range(x + 1, len(top)):
                        i, j = top[x], top[y]
                        pairs[f"{tid}|{i}|{j}"] = np.hstack([tiles[i], np.full((384, 16, 3), 255, np.uint8), tiles[j]])
                        pairs[f"{tid}|{j}|{i}"] = np.hstack([tiles[j], np.full((384, 16, 3), 255, np.uint8), tiles[i]])
        if pairs:
            try:
                ans = FaceAttributes().compare(pairs)
            except Exception as e:  # noqa: BLE001
                ans = {}
                self._log(f"quality comparison unavailable ({type(e).__name__}); VLM ratings used")
            wins: dict = {}
            for k, v in ans.items():
                tid_, i, j = k.rsplit("|", 2)
                tid_ = next(t for t in out if str(t) == tid_)
                w = wins.setdefault(tid_, {})
                if v == "A":
                    w[int(i)] = w.get(int(i), 0) + 1
                elif v == "B":
                    w[int(j)] = w.get(int(j), 0) + 1
            for tid, w in wins.items():
                keep, q = out[tid]
                # pairwise wins dominate; the per-image ratings (/100) break ties and order the rest
                q = [q[i] / 100.0 + w.get(i, 0) for i in range(len(keep))]
                out[tid] = (keep, q)
                self._log(f"track {tid}: quality wins by Qwen3-VL (pairs in both orders): "
                          + ", ".join(f"{i}:{v}" for i, v in sorted(w.items(), key=lambda x: -x[1])))
        res = {}
        for t in todo:
            keep, q = out.get(t.tid, ([], []))
            if not keep and attrs.get(t.tid):
                self._log(f"track {t.tid}: no synthetic candidate matches the person's attributes -> whole track blurred")
                self.cand_info.setdefault(t.tid, {})["no_match"] = True
            res[t.tid] = keep
            self.cand_quality[t.tid] = {ci: qq for ci, qq in enumerate(q)}
            self.cand_info.setdefault(t.tid, {})["matched"] = len(keep)
        return res

    # ---------------------------------------------------------------- prepare
    def prepare(self, video_path: str, tracks) -> None:
        t0 = time.time()
        tracks = list(tracks)
        for t in tracks:
            for f, v in t.dense.items():
                self._fidx[(t.tid, id(v[0]))] = f
        if not self._guard_ready():
            self._log("SFace guard unavailable -> every face is blurred (fail closed)")
            return
        want = self._keyframes(tracks)
        kf = {t.tid: (f, d) for f, L in want.items() for t, d in L}
        todo = [t for t in tracks if t.tid in kf]
        if not todo:
            self.prep_seconds = time.time() - t0
            return
        # pass 1: per-frame YuNet points on the original (+ keyframe crops)
        frames_needed = {}
        for t in todo:
            for f, (box, lm, src) in t.dense.items():
                frames_needed.setdefault(f, []).append((t, box, lm))
        pts: dict[int, dict] = {t.tid: {} for t in todo}
        kcrop = {}
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        f = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            for t, box, lm in frames_needed.get(f, []):
                if lm is None:
                    continue
                y = self._yunet_lm(fr, box)
                pts[t.tid][f] = (y if y is not None else np.asarray(lm, np.float32).reshape(5, 2), box, lm)
                if f == kf[t.tid][0]:
                    x1, y1, x2, y2 = box; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    s = max(x2 - x1, y2 - y1) * 0.95
                    H, W = fr.shape[:2]
                    kcrop[t.tid] = fr[int(max(0, cy - s)):int(min(H, cy + s)), int(max(0, cx - s)):int(min(W, cx + s))].copy()
            f += 1
        cap.release()
        # smooth the 5 points per contiguous run (zero-phase)
        for tid, d in pts.items():
            fs = sorted(d)
            runs, cur = [], []
            for g in fs:
                if cur and g != cur[-1] + 1:
                    runs.append(cur); cur = []
                cur.append(g)
            if cur:
                runs.append(cur)
            for r in runs:
                P = np.stack([d[g][0] for g in r]).astype(np.float64)
                if self.stabilize and self.euro:
                    P = _one_euro_zero_phase(P, self.fps, *self.euro)
                P = _gauss_smooth(P, self.lm_sigma)
                for g, p in zip(r, P):
                    d[g] = (p.astype(np.float32),) + d[g][1:]
        # attributes -> synthetic portraits
        attrs = {}
        if self.use_attributes and kcrop:
            try:
                from .face_attributes import FaceAttributes
                fa = FaceAttributes()
                attrs = fa(kcrop)
                self._log(f"face attributes for {sum(bool(a) for a in attrs.values())}/{len(kcrop)} tracks "
                          f"in {fa.seconds:.1f}s (kept in memory only)")
            except Exception as e:
                self._log(f"face attributes unavailable ({type(e).__name__}: {e}); neutral prompt")
        self.t2i, self._gen_dev = self._sd_txt2img()
        ports = {t.tid: self._portraits(attrs.get(t.tid, {}), self._seed(video_path, t.tid), self.n_cand)
                 for t in todo}
        if self.verify and attrs:
            ports = self._verify_candidates(video_path, todo, attrs, ports)
        self.t2i = None
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
        self.nets = SwapNets(self.device, self.swapper, self.enhancer)
        if self.stabilize:
            self._prepare_stable(video_path, todo, pts, ports)
            self.nets = None
            self.prep_seconds = time.time() - t0
            nb = sum(len(v) for v in self.frame_state.values())
            self._log(f"research swap (stable): {len(self.out)} frames rendered, {len(self.blocked)} blurred "
                      f"(guard + consolidation), {sum(0 < w < 1 for v in self.frame_state.values() for w in v.values())} "
                      f"cross-fade frames of {nb}, {self.prep_seconds:.1f}s")
            return
        # candidate selection: lowest p75 SFace similarity on sampled frames
        frames_all = sorted({g for d in pts.values() for g in d})
        cache = {}
        cap = cv2.VideoCapture(video_path)

        def read(g):
            if g not in cache:
                cap.set(cv2.CAP_PROP_POS_FRAMES, g); cache[g] = cap.read()[1]
            return cache[g]
        srcs = {}
        for t in todo:
            d = pts[t.tid]
            fs = [g for g in sorted(d) if self._v2_gate_q(d[g][1], d[g][2])]
            if not fs or not ports[t.tid]:
                continue
            samp = [fs[int(i)] for i in np.linspace(0, len(fs) - 1, min(self.n_score, len(fs)))]
            temb = np.mean([self.nets.embed(read(g), d[g][0]) for g in samp], 0)
            best = None
            clean = [ci for ci, (bgr, lm) in enumerate(ports[t.tid]) if not self._hairy_forehead(bgr, lm)]
            if len(clean) < len(ports[t.tid]):
                self._log(f"track {t.tid}: {len(ports[t.tid]) - len(clean)} candidates skipped (hair over the forehead)")
            for ci, (bgr, lm) in enumerate(ports[t.tid]):
                if clean and ci not in clean:
                    continue
                src = self.nets.source_vec(self.nets.embed(bgr, lm), temb, self.push)
                cos = []
                for g in samp:
                    fr = read(g)
                    comp = fr.copy()
                    o = self._render(fr, d[g][0], src)
                    if o is None:
                        continue
                    self._paste(comp, *o)
                    cos.append(self._id_cos(fr, comp, d[g][1], d[g][2], {}))
                sc = float(np.percentile(cos, 75)) if cos else 9.0
                if best is None or sc < best[0]:
                    best = (sc, ci, src)
            srcs[t.tid] = best[2]
            self.surr[t.tid] = ports[t.tid][best[1]][0]
            self.gen_info[t.tid] = {"p75_sface": round(best[0], 3), "candidates": len(ports[t.tid]), "pick": best[1]}
            self._log(f"track {t.tid}: synthetic identity {best[1]}/{len(ports[t.tid])}, p75 SFace {best[0]:.3f}")
        cache.clear()
        # pass 2: render every frame (aligned crop space), then smooth over time
        raw: dict[int, dict] = {tid: {} for tid in srcs}
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for g in range(max(frames_all) + 1 if frames_all else 0):
            ok, fr = cap.read()
            if not ok:
                break
            for tid in srcs:
                p = pts[tid].get(g)
                if p is None or not self._v2_gate_q(p[1], p[2]):
                    continue
                o = self._render(fr, p[0], srcs[tid])
                if o is not None:
                    raw[tid][g] = (o[0], o[1].clip(0, 255).astype(np.uint8), o[2].astype(np.float16))
        for tid, d in raw.items():
            fs = sorted(d)
            if self.t_sigma > 0 and len(fs) > 2:
                r = int(np.ceil(2 * self.t_sigma))
                for g in fs:
                    nb = [h for h in range(g - r, g + r + 1) if h in d]
                    w = np.array([np.exp(-0.5 * ((h - g) / self.t_sigma) ** 2) for h in nb]); w /= w.sum()
                    M0 = d[g][0]
                    acc_c = np.zeros(d[g][1].shape, np.float32); acc_a = np.zeros(d[g][2].shape, np.float32)
                    for wi, h in zip(w, nb):
                        # neighbours are re-sampled into this frame's crop space
                        A = _rel(d[h][0], M0)
                        if self.img_smooth:
                            acc_c += wi * cv2.warpAffine(d[h][1], A, d[g][1].shape[:2][::-1], flags=cv2.INTER_LINEAR,
                                                         borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
                        acc_a += wi * cv2.warpAffine(d[h][2].astype(np.float32), A, d[g][2].shape[::-1],
                                                     flags=cv2.INTER_LINEAR)
                    c = np.clip(acc_c, 0, 255).astype(np.uint8) if self.img_smooth else d[g][1]
                    self.out[(tid, g)] = (M0, c, acc_a.astype(np.float16))
            else:
                for g in fs:
                    self.out[(tid, g)] = d[g]
        # guard every rendered frame (fail closed -> blur)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.blocked = set()
        nmax = max([g for _, g in self.out] or [-1])
        by_f = {}
        for (tid, g) in self.out:
            by_f.setdefault(g, []).append(tid)
        for g in range(nmax + 1):
            ok, fr = cap.read()
            if not ok:
                break
            for tid in by_f.get(g, []):
                comp = fr.copy()
                self._paste(comp, *self.out[(tid, g)])
                p = pts[tid][g]
                c = self._id_cos(fr, comp, p[1], p[2], {})
                if c > self._thr_for(p[1]):
                    self.blocked.add((tid, g))
        cap.release()
        self.blocked = self._consolidate_blocks(self.blocked, by_f)
        self.out = {k: v for k, v in self.out.items() if k not in self.blocked}
        for g, tids in by_f.items():
            for tid in tids:
                self.frame_state.setdefault(tid, {})[g] = 1.0 if (tid, g) in self.out else 0.0
        self.nets = None
        self.prep_seconds = time.time() - t0
        self._log(f"research swap: {len(self.out)} frames rendered, {len(self.blocked)} blocked by the guard, "
                  f"{self.prep_seconds:.1f}s")

    # ------------------------------------------------------------ stable mode
    def _prepare_stable(self, video_path, todo, pts, ports):
        """Identity and render level per track chosen by the number of guard-tripping
        frames on a dense sample; per-frame retries with a stronger push; blur segments
        consolidated (no short swap runs) and cross-faded."""
        cache = {}
        cap = cv2.VideoCapture(video_path)

        def read(g):
            if g not in cache:
                cap.set(cv2.CAP_PROP_POS_FRAMES, g); cache[g] = cap.read()[1]
            return cache[g]

        def trips_on(frames, d, src, mk):
            cos = []
            for g in frames:
                fr = read(g)
                o = self._render(fr, d[g][0], src, mk)
                if o is None:
                    continue
                comp = fr.copy(); self._paste(comp, *o)
                cos.append((self._id_cos(fr, comp, d[g][1], d[g][2], {}), self._thr_for(d[g][1])))
            if not cos:
                return 9.0, 9.0
            return (float(np.mean([c > th for c, th in cos])), float(np.percentile([c for c, _ in cos], 75)))

        plan = {}
        for t in todo:
            d = pts[t.tid]
            fs = [g for g in sorted(d) if self._v2_gate_q(d[g][1], d[g][2])]
            if not fs or not ports[t.tid]:
                continue
            dense = [fs[int(i)] for i in np.linspace(0, len(fs) - 1, min(self.n_dense, len(fs)))]
            samp = [fs[int(i)] for i in np.linspace(0, len(fs) - 1, min(self.n_score, len(fs)))]
            temb = np.mean([self.nets.embed(read(g), d[g][0]) for g in samp], 0)
            clean = [ci for ci, (bgr, lm) in enumerate(ports[t.tid]) if not self._hairy_forehead(bgr, lm)] \
                or list(range(len(ports[t.tid])))
            qual = self.cand_quality.get(t.tid) or {}
            if qual:   # attribute-matched candidates: the n_quality best-looking go on to the guard test
                clean = sorted(clean, key=lambda ci: -qual.get(ci, 0.0))[:self.n_quality]
                self._log(f"track {t.tid}: candidates by quality " +
                          ", ".join(f"{ci}:{qual.get(ci, 0):.2f}" for ci in clean))
            embs = {ci: self.nets.embed(*ports[t.tid][ci]) for ci in clean}
            tried, chosen = [], None
            cands = clean
            for li, (dp, mk) in enumerate(self.track_levels):
                mk = self._mk if mk is None else mk
                res = []
                for ci in cands:
                    tf, p75 = trips_on(dense, d, self.nets.source_vec(embs[ci], temb, self.push + dp), mk)
                    res.append((tf, p75, ci))
                    tried.append((tf, li, p75, ci))
                res.sort()
                self._log(f"track {t.tid}: level {li} (push +{dp}, mouth_keep {mk}): best candidate {res[0][2]} "
                          f"trips {res[0][0]:.2f} on {len(dense)} frames, p75 SFace {res[0][1]:.3f}")
                # candidates passing the guard at this level, with p75 <= p75_target, AND within `p75_margin` of the most private
                # one's 75th-percentile SFace similarity are acceptable: take the best-looking.  The margin
                # keeps the quality preference from trading away privacy headroom (face_h: without it the
                # clip-4 pick re-identified on 2.5 % of frames under the evaluation-only ArcFace)
                ok = [r for r in res if r[0] <= self.level_ok and r[1] <= res[0][1] + self.p75_margin
                      and r[1] <= self.p75_target]
                if ok:
                    best = max(ok, key=lambda r: (qual.get(r[2], 0.0), -r[0], -r[1]))
                    chosen = (best[2], li)
                    self._log(f"track {t.tid}: level {li}: {len(ok)} candidate(s) within the guard and the "
                              f"p75 margin; best-looking {best[2]} (trips {best[0]:.2f}, p75 SFace {best[1]:.3f})")
                    break
                cands = [ci for _, _, ci in res[:3]]   # stronger levels: only the 3 best identities
            if chosen is None:   # no level accepted: fewest trips, then the most private (lowest p75)
                tf, li, p75, ci = min(tried, key=lambda x: (x[0], x[2], x[1]))
                chosen = (ci, li)
            ci, li = chosen
            trip0 = [x for x in tried if x[3] == ci and x[1] == li][0][0]
            if trip0 > self.track_blur_trip:
                self.gen_info[t.tid] = {"whole_track_blur": True, "dense_trip_frac": round(trip0, 3)}
                self._log(f"track {t.tid}: best identity still trips the guard on {trip0:.0%} of the dense sample "
                          f"-> whole track blurred (consistent, no toggling)")
                continue
            dp, mk = self.track_levels[li]
            plan[t.tid] = dict(emb=embs[ci], temb=temb, dp=dp, mk=self._mk if mk is None else mk)
            self.surr[t.tid] = ports[t.tid][ci][0]
            trip = [x for x in tried if x[3] == ci and x[1] == li][0][0]
            self.gen_info[t.tid] = {"pick": ci, "level": li, "dense_trip_frac": round(trip, 3),
                                    "candidates": len(ports[t.tid]), "generator": self.generator,
                                    "quality": round(qual.get(ci, 0.0), 1) if qual else None,
                                    **{k: v for k, v in (self.cand_info.get(t.tid) or {}).items()}}
            self._log(f"track {t.tid}: synthetic identity {ci}/{len(ports[t.tid])}, level {li} "
                      f"(push {self.push + dp:.2f}, mouth_keep {plan[t.tid]['mk']}), dense trip rate {trip:.2f}")
        cache.clear()
        dd = os.environ.get("CUSTODIAN_RESEARCH_DUMP_DIR")   # debug: synthetic portraits only (no real faces)
        if dd:
            dd = os.path.join(dd, os.path.splitext(os.path.basename(video_path))[0])
            os.makedirs(dd, exist_ok=True)
            for tid, L in ports.items():
                for ci, (bgr, _) in enumerate(L):
                    q = (self.cand_quality.get(tid) or {}).get(ci)
                    tag = "chosen" if tid in plan and self.surr.get(tid) is bgr else "cand"
                    cv2.imwrite(os.path.join(dd, f"track{tid}_{ci}_{tag}_q{-1 if q is None else q:.2f}.jpg"), bgr)
        # render + guard every frame; retries with a stronger push before giving up
        self.blocked, by_f, retried = set(), {}, 0
        nmax = max([g for tid in plan for g in pts[tid]] or [-1])
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for g in range(nmax + 1):
            ok, fr = cap.read()
            if not ok:
                break
            for tid, P in plan.items():
                p = pts[tid].get(g)
                if p is None or not self._v2_gate_q(p[1], p[2]):
                    continue
                by_f.setdefault(g, []).append(tid)
                done = False
                for k, extra in enumerate((0.0,) + self.retry_push):
                    src = self.nets.source_vec(P["emb"], P["temb"], self.push + P["dp"] + extra)
                    o = self._render(fr, p[0], src, P["mk"])
                    if o is None:
                        break
                    comp = fr.copy(); self._paste(comp, *o)
                    if self._id_cos(fr, comp, p[1], p[2], {}) <= self._thr_for(p[1]):
                        self.out[(tid, g)] = (o[0], o[1].clip(0, 255).astype(np.uint8), o[2].astype(np.float16))
                        retried += k > 0
                        done = True
                        break
                if not done:
                    self.blocked.add((tid, g))
        cap.release()
        self.stats["guard_retry_ok"] = retried
        self.stats["guard_trip_raw"] = len(self.blocked)
        self.blocked = self._consolidate_stable(self.blocked, by_f)
        self.out = {k: v for k, v in self.out.items() if k not in self.blocked}
        for t in todo:   # whole-track blur (or no usable identity): one consistent blur
            if t.tid not in plan:
                self.frame_state[t.tid] = {g: 0.0 for g in pts[t.tid] if self._v2_gate_q(pts[t.tid][g][1], pts[t.tid][g][2])}
                self.blocked |= {(t.tid, g) for g in self.frame_state[t.tid]}
        # cross-fade weights: 0 on blurred frames, ramping to 1 over `fade` frames
        for tid in plan:
            fs = sorted(g for g, tids in by_f.items() if tid in tids)
            bl = [g for g in fs if (tid, g) in self.blocked]
            st = {}
            for g in fs:
                if (tid, g) in self.blocked or (tid, g) not in self.out:
                    w = 0.0
                elif not bl:
                    w = 1.0
                else:
                    dist = min(abs(g - b) for b in bl)
                    w = min(1.0, dist / (self.fade + 1))
                st[g] = w
                if w > 0:
                    self.weight[(tid, g)] = w
            self.frame_state[tid] = st

    def _consolidate_stable(self, blocked, by_f):
        """Fail-closed consolidation (only ever ADDS blur): pad each trip, merge blur
        segments closer than `block_gap`, blur swap runs shorter than `min_swap_run`
        that touch a blur segment, blur the whole track if most frames trip."""
        frames = {}
        for g, tids in by_f.items():
            for tid in tids:
                frames.setdefault(tid, []).append(g)
        out = set(blocked)
        pad = min(self.block_pad, 2)
        for tid, fs in frames.items():
            fs = sorted(fs); fset = set(fs)
            hit = [g for g in fs if (tid, g) in blocked]
            if not hit:
                continue
            if len(hit) > self.block_track * len(fs):
                out |= {(tid, g) for g in fs}
                continue
            for g in hit:
                out |= {(tid, h) for h in range(g - pad, g + pad + 1) if h in fset}
            hit = sorted(g for g in fs if (tid, g) in out)
            for a, b in zip(hit, hit[1:]):
                if 1 < b - a <= self.block_gap:
                    out |= {(tid, h) for h in range(a, b) if h in fset}
            # short swap runs touching blur (or the track edge next to blur) -> blur
            state = [(tid, g) in out for g in fs]
            i = 0
            while i < len(fs):
                if state[i]:
                    i += 1; continue
                j = i
                while j < len(fs) and not state[j]:
                    j += 1
                touches = (i > 0 and state[i - 1]) or (j < len(fs) and state[j])
                if touches and j - i < self.min_swap_run:
                    out |= {(tid, fs[k]) for k in range(i, j)}
                i = j
            if sum((tid, g) in out for g in fs) > self.block_track * len(fs):
                out |= {(tid, g) for g in fs}
        return out

    def _consolidate_blocks(self, blocked, by_f):
        """Turn isolated guard trips into contiguous blur segments (no surrogate/blur
        flicker): each trip blocks +-`block_pad` frames, gaps <= `block_gap` frames
        between blocked frames are blocked too, and a track blocked on more than
        `block_track` of its frames is blurred entirely.  Only ever ADDS blur."""
        frames = {}
        for g, tids in by_f.items():
            for tid in tids:
                frames.setdefault(tid, []).append(g)
        out = set(blocked)
        for tid, fs in frames.items():
            fs = sorted(fs); fset = set(fs)
            hit = sorted(g for g in fs if (tid, g) in blocked)
            if not hit:
                continue
            if len(hit) > self.block_track * len(fs):
                out |= {(tid, g) for g in fs}
                continue
            for g in hit:
                out |= {(tid, h) for h in range(g - self.block_pad, g + self.block_pad + 1) if h in fset}
            hit = sorted(g for g in fs if (tid, g) in out)
            for a, b in zip(hit, hit[1:]):
                if 1 < b - a <= self.block_gap:
                    out |= {(tid, h) for h in range(a, b) if h in fset}
        return out

    def _hairy_forehead(self, bgr, lm, thr=0.04):
        """True if the synthetic portrait has hair (BiSeNet class 17) over the forehead
        band (between the eyes' x-range, from the brows up 40 % of the eye-mouth distance)."""
        B = 256
        M = align_matrix(lm, "arcface_128", B)
        c = cv2.warpAffine(bgr, M, (B, B), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        lab = cv2.resize(self.nets.parse(c), (B, B), interpolation=cv2.INTER_NEAREST)
        T = TEMPLATES["arcface_128"] * B
        ey, my = T[:2, 1].mean(), T[3:, 1].mean()
        y2 = int(ey - 0.25 * (my - ey)); y1 = int(max(0, y2 - 0.55 * (my - ey)))
        x1, x2 = int(T[0, 0]), int(T[1, 0])
        band = lab[y1:y2, x1:x2]
        return band.size > 0 and float((band == 17).mean()) > thr

    def _v2_gate_q(self, box, lm):
        return lm is not None and min(box[2] - box[0], box[3] - box[1]) >= 1.25 * self.v2_min_face

    def _render(self, fr, lm5, src, mk=None):
        """swap one face: -> (M, swapped crop float32, alpha float32) in aligned crop space."""
        N = self.nets
        B = N.size * max(1, self.boost)
        M = align_matrix(lm5, "arcface_128", B)
        if M is None:
            return None
        crop = cv2.warpAffine(fr, M, (B, B), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        sw, smask = N.swap(crop, src)
        for _ in range(self.iters - 1):   # re-swap the swapped crop: stronger identity transfer
            sw, smask2 = N.swap(sw.clip(0, 255).astype(np.uint8), src)
            smask = np.minimum(smask, smask2)
        lab = cv2.resize(N.parse(crop), (B, B), interpolation=cv2.INTER_NEAREST)
        keep = PARSE_FACE + ([PARSE_GLASSES] if self.glasses else [])
        reg = np.isin(lab, keep).astype(np.float32)
        if self.parse_swap:   # never paste what the SWAPPED face shows as hair / background / clothing
            lab2 = cv2.resize(N.parse(sw.clip(0, 255).astype(np.uint8)), (B, B), interpolation=cv2.INTER_NEAREST)
            reg *= np.isin(lab2, keep + [PARSE_GLASSES]).astype(np.float32)
        k = max(1, int(self.m_erode * B))
        reg = cv2.erode(reg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        occ = N.occlusion(crop)
        box = np.zeros((B, B), np.float32)
        pad = int(0.04 * B); box[pad:-pad, pad:-pad] = 1
        parts = [reg, occ, box]
        if self.swap_mask and N.kind == "hyperswap":
            parts.append(cv2.GaussianBlur(smask, (0, 0), 2))
        a = np.minimum.reduce(parts)
        a = cv2.GaussianBlur(a, (0, 0), max(1.0, self.m_blur * B))
        mk = self.mouth_keep if mk is None else mk
        if mk > 0:   # mouth opening / lips follow the ORIGINAL speech exactly
            mo = np.isin(lab, [11, 12, 13]).astype(np.float32)
            d = max(1, int(0.015 * B))
            mo = cv2.dilate(mo, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
            mo = cv2.GaussianBlur(mo, (0, 0), max(1.0, 0.012 * B))
            a = a * (1 - mk * mo)
        if self.brow_keep > 0:    # eyebrow shape / raise follows the original (expression fidelity)
            br = np.isin(lab, [2, 3]).astype(np.float32)
            d = max(1, int(0.012 * B))
            br = cv2.dilate(br, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
            br = cv2.GaussianBlur(br, (0, 0), max(1.0, 0.01 * B))
            a = a * (1 - self.brow_keep * br)
        if self.eye_keep > 0:     # partial: eyelid closure follows the original (blink fidelity)
            ey = np.isin(lab, [4, 5]).astype(np.float32)
            d = max(1, int(0.01 * B))
            ey = cv2.dilate(ey, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
            ey = cv2.GaussianBlur(ey, (0, 0), max(1.0, 0.01 * B))
            a = a * (1 - self.eye_keep * ey)
        if N.enh is not None:
            sw = self._enhance(sw, lm5, M, B)
        return M, sw, a

    def _enhance(self, sw, lm5, M, B):
        """restore the swapped crop: re-align it to the FFHQ template, enhance, warp back, blend."""
        N = self.nets
        p = cv2.transform(np.asarray(lm5, np.float32).reshape(1, 5, 2), M).reshape(5, 2)
        E = align_matrix(p, "ffhq_512", 512)
        c = cv2.warpAffine(sw.clip(0, 255).astype(np.uint8), E, (512, 512), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)
        e = N.enhance(c)
        back = cv2.warpAffine(e.astype(np.float32), cv2.invertAffineTransform(E), (B, B), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
        return (1 - self.enhance_blend) * sw + self.enhance_blend * back

    @staticmethod
    def _paste(frame, M, crop, alpha):
        H, W = frame.shape[:2]
        B = crop.shape[0]
        Mi = cv2.invertAffineTransform(M)
        q = cv2.transform(np.array([[[0, 0], [B, 0], [B, B], [0, B]]], np.float32), Mi)[0]
        x1, y1 = np.floor(q.min(0)).astype(int); x2, y2 = np.ceil(q.max(0)).astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return
        P = Mi.copy(); P[0, 2] -= x1; P[1, 2] -= y1
        a = cv2.warpAffine(alpha.astype(np.float32), P, (x2 - x1, y2 - y1), flags=cv2.INTER_LINEAR)[..., None]
        c = cv2.warpAffine(crop.astype(np.float32), P, (x2 - x1, y2 - y1), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)
        roi = frame[y1:y2, x1:x2].astype(np.float32)
        frame[y1:y2, x1:x2] = np.clip(roi * (1 - a) + c * a, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ apply
    def apply(self, frame, track, box, lm) -> bool:
        if not self._v2_gate(box, lm):
            self.stats["none"] += 1
            return False
        f = self._frame_of(track, box)
        o = self.out.get((track.tid, f)) if f is not None else None
        if o is None:
            self.stats["none"] += 1
            if (track.tid, f) in getattr(self, "blocked", ()):
                self.stats["guard_blur"] = self.stats.get("guard_blur", 0) + 1
            return False
        w = self.weight.get((track.tid, f), 1.0) if self.stabilize else 1.0
        if w < 1.0:   # cross-fade between the (verified) swap and the blur fallback
            from .masking import redact_box
            bl = frame.copy()
            redact_box(bl, box, "blur", "ellipse")
            self._paste(frame, *o)
            frame[:] = np.clip(w * frame.astype(np.float32) + (1 - w) * bl.astype(np.float32), 0, 255).astype(np.uint8)
            self.stats["crossfade"] = self.stats.get("crossfade", 0) + 1
        else:
            self._paste(frame, *o)
        self.stats["reenact"] += 1
        return True


def _one_euro(x, fps, min_cutoff, beta, d_cutoff=1.0):
    """one-euro filter along axis 0 (causal)."""
    def alpha(c):
        tau = 1.0 / (2 * np.pi * c); te = 1.0 / fps
        return 1.0 / (1.0 + tau / te)
    y = np.empty_like(x); y[0] = x[0]; dx = np.zeros_like(x[0])
    for i in range(1, len(x)):
        d = (x[i] - x[i - 1]) * fps
        dx = dx + alpha(d_cutoff) * (d - dx)
        c = min_cutoff + beta * np.abs(dx)
        a = 1.0 / (1.0 + (1.0 / (2 * np.pi * c)) * fps)
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def _one_euro_zero_phase(x, fps, min_cutoff=1.5, beta=0.01):
    """forward + backward one-euro, averaged: no lag (the whole track is known offline)."""
    if len(x) < 3:
        return x
    return 0.5 * (_one_euro(x, fps, min_cutoff, beta) + _one_euro(x[::-1], fps, min_cutoff, beta)[::-1])


def _rel(Mh, Mg):
    """affine mapping crop-space of frame h to crop-space of frame g (both 2x3 frame->crop)."""
    A = np.vstack([Mg, [0, 0, 1]]) @ np.vstack([cv2.invertAffineTransform(Mh), [0, 0, 1]])
    return A[:2]
