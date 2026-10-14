"""EXPERIMENTAL (face_f): parsing-aware paste-back for the re-enacted surrogate
face.  Same identity generation and LivePortrait motion as
`SDKeyframeSurrogate(realism="v2")`; only what is pasted, and how, changes.
The default backend (surrogate_face.py) is untouched; this class is opt-in.

Problems of v2 it targets (visible on large faces, demo clips 1 / 4 / 5):
  * ghost hair at the forehead and background (blinds) bleeding round the head:
    v2 pastes a landmark hull + a forehead band gated by a *colour* skin test,
    so the surrogate's hair / warped background that happens to be skin-coloured
    (orange blinds) got pasted, and the alpha ramp crossed the hairline;
  * mismatched glasses: SD drew its own glasses, the original's temples stayed;
  * pasted-on look: wide alpha ramp across the outline.

v3 masks (per frame, LivePortrait crop space, 256 px):
  alpha = faceskin(original) * erode(faceskin(render)) * (1 - accessories(original))
          * (1 - accessories(render)),
  max'ed with the original's eye / brow / nose / mouth hull (identity core,
  minus original hair and accessories), so the edit ends exactly where BOTH the
  original and the surrogate show face skin: never on hair, never on background.
  Face skin / hair / accessories come from MediaPipe's selfie-multiclass
  segmenter (Apache-2.0; face_parse.py).  The masks are smoothed zero-phase over
  time (the crops are already temporally smoothed, so this is in face coordinates).
  Glasses: the surrogate is generated WITHOUT glasses (negative prompt) and the
  original's frames (accessory class) are kept on top of the new face.
  Blending: alpha (default) or a 3-level Laplacian pyramid of the surrogate-minus-
  original difference (`blend3="laplacian"`): low frequencies blend over a few px,
  edges stay sharp, nothing outside the dilated mask changes.
The identity guard and fail-closed behaviour are inherited unchanged (the
guard's "cover more" retry uses max(v3 mask, v2 core)).
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .surrogate_face import INNER, LP_D, SDKeyframeSurrogate

Q = 256  # mask / parsing resolution in crop space


def _min_filter(x, k):
    return cv2.erode(x, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) if k > 0 else x


def _tsmooth(stack, sigma):
    """zero-phase Gaussian along axis 0 (edge padded)"""
    if sigma <= 0 or len(stack) < 3:
        return stack
    r = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    p = np.pad(stack, [(r, r)] + [(0, 0)] * (stack.ndim - 1), mode="edge")
    out = np.zeros_like(stack, dtype=np.float32)
    for j, w in enumerate(k):
        out += w * p[j:j + len(stack)]
    return out


def lap_blend_diff(ws, roi, m, levels=3):
    """roi + pyramid-blended (ws - roi) with weights m (HxW, 0..1)."""
    a = ws.astype(np.float32); b = roi.astype(np.float32)
    d = a - b
    gm, gd = [m.astype(np.float32)], [d]
    for _ in range(levels):
        if min(gd[-1].shape[:2]) < 8:
            break
        gm.append(cv2.pyrDown(gm[-1])); gd.append(cv2.pyrDown(gd[-1]))
    out = gd[-1] * gm[-1][..., None]
    for i in range(len(gd) - 2, -1, -1):
        sz = gd[i].shape[1::-1]
        lap = gd[i] - cv2.pyrUp(gd[i + 1], dstsize=sz)
        out = cv2.pyrUp(out, dstsize=sz) + lap * gm[i][..., None]
    return np.clip(b + out + 0.5, 0, 255).astype(np.uint8)


class SDKeyframeSurrogateV3(SDKeyframeSurrogate):
    name_suffix = " + v3 parsing-aware paste (MediaPipe selfie-multiclass, Apache-2.0)"

    def __init__(self, *a, blend3: str = "alpha", glasses: str = "keep", render_erode: int = 2,
                 mask_tsigma: float = 1.0, mask_ssigma: float = 0.8, core: bool = True,
                 parse_scorer: bool = True, forehead_max: float | None = None,
                 fill: bool = False, source_clean: bool = False, **k):
        k.setdefault("realism", "v2")
        super().__init__(*a, **k)
        assert self.realism == "v2" and self.motion == "reenact"
        self.blend3, self.glasses = blend3, glasses
        self.render_erode, self.mask_tsigma, self.mask_ssigma = render_erode, mask_tsigma, mask_ssigma
        self.core, self.parse_scorer = core, parse_scorer
        # cap the edit at `forehead_max` x (brow-to-chin height) above the brows
        # (None = no cap): LivePortrait drags faint SD hair strands across the
        # forehead that the segmenter still calls skin
        self.forehead_max = forehead_max
        # fill=True: replace ALL of the original's face skin (no original forehead /
        # outline strip left: privacy); where the aligned render shows hair,
        # background or accessories instead of skin, its pixels are first replaced
        # by a smooth push-pull extension of its own skin (texture/grain is added
        # back by the appearance match)
        self.fill, self.source_clean = fill, source_clean
        self._fp = None
        self.name += self.name_suffix
        self.v3_stats = {"frames": 0, "alpha_mean": 0.0}

    # ------------------------------------------------------------------ parser
    def _parser(self):
        if self._fp is None:
            from .face_parse import FaceParser
            self._fp = FaceParser()
        return self._fp

    def prepare(self, video_path, tracks):
        self._video = video_path
        super().prepare(video_path, tracks)
        if self._fp is not None:
            self._fp.close(); self._fp = None

    # ------------------------------------------------------------------ generation
    def generate_v2(self, crop_bgr, seed, attrs, score_fn=None, max_tries=None):
        """glasses="keep": the surrogate is drawn without glasses (the original's
        frames are kept on top at paste time)."""
        if self.glasses != "keep":
            return super().generate_v2(crop_bgr, seed, attrs, score_fn, max_tries)
        from . import face_attributes as fa
        orig = fa.prompt_from_attributes

        def no_glasses(a):
            a = dict(a or {}); a["glasses"] = "none"
            pos, neg = orig(a)
            return pos, neg + ", glasses, eyeglasses, spectacles, sunglasses, glasses frame"
        fa.prompt_from_attributes = no_glasses
        try:
            return super().generate_v2(crop_bgr, seed, attrs, score_fn, max_tries)
        finally:
            fa.prompt_from_attributes = orig

    def _make_source(self, tid, sd_crop, rect_kf, l203_kf, lm5_kf):
        """source_clean: before the SD identity becomes the LivePortrait source,
        hair strands / glasses it drew INSIDE the face (forehead arcs that
        LivePortrait then animates as ghost lines) are replaced by a push-pull
        extension of its own skin."""
        if self.source_clean and sd_crop.shape[0] == LP_D:
            try:
                from .face_reenact.engine import m_o2c, tpts
                from .surrogate_face import _hull_mask
                P = self._parser().parse([sd_crop])[0]
                p = tpts(l203_kf, m_o2c(rect_kf, LP_D)) * (Q / LP_D)
                hull = cv2.dilate(_hull_mask(p, Q, 0.7), np.ones((5, 5), np.uint8)).astype(np.float32) / 255
                badf = np.clip(np.maximum(P[..., 1], P[..., 2]) * hull, 0, 1)
                Pf = np.stack([1 - badf, np.zeros_like(badf), np.zeros_like(badf)], -1)
                sd_crop = self._fill_render(sd_crop, Pf)
                self.v3_stats["source_cleaned_frac"] = round(float((badf > 0.35).mean()), 4)
            except Exception as e:
                self._log(f"source_clean skipped ({type(e).__name__}: {e})")
        return super()._make_source(tid, sd_crop, rect_kf, l203_kf, lm5_kf)

    # ------------------------------------------------------------------ masks
    def _bad_render(self, Pr):
        fr, hr, orr = Pr[..., 0], Pr[..., 1], Pr[..., 2]
        return np.clip(np.maximum.reduce([1 - _min_filter(fr, self.render_erode), hr, orr]), 0, 1)

    def _fill_render(self, img, Pr):
        """push-pull fill of the render's non-skin pixels from its own skin"""
        n = img.shape[0]
        bad = cv2.resize(self._bad_render(Pr), (n, n), interpolation=cv2.INTER_LINEAR)
        valid = (bad < 0.35).astype(np.float32)
        if valid.mean() > 0.995 or valid.sum() < 50:
            return img
        valid = cv2.erode(valid, np.ones((3, 3), np.uint8))
        x = img.astype(np.float32)
        fill, have = np.zeros_like(x), np.zeros_like(valid)
        for sg in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
            sg = sg * n / 256
            w = cv2.GaussianBlur(valid, (0, 0), sg)
            f = cv2.GaussianBlur(x * valid[..., None], (0, 0), sg) / (w[..., None] + 1e-6)
            take = (w > 0.05) & (have < 0.5)
            fill[take] = f[take]; have[take] = 1
        fill[have < 0.5] = x[have < 0.5]
        wv = cv2.GaussianBlur(valid, (0, 0), 1.0 * n / 256)[..., None]
        return np.clip(x * wv + fill * (1 - wv), 0, 255).astype(np.uint8)

    def _alpha(self, Po, Pr, po512, fill=None):
        """Po, Pr: (256,256,3) float [face, hair, others] of original / aligned render;
        po512: original 203 landmarks in 512 crop space."""
        fill = self.fill if fill is None else fill
        fo, ho, oo = Po[..., 0], Po[..., 1], Po[..., 2]
        fr, _, orr = Pr[..., 0], Pr[..., 1], Pr[..., 2]
        if fill:
            from .surrogate_face import _hull_mask
            hull = _hull_mask(po512 * (Q / LP_D), Q, 0.8)
            hull = cv2.GaussianBlur(cv2.dilate(hull, np.ones((7, 7), np.uint8)).astype(np.float32) / 255, (0, 0), 2.0)
            a = fo * (1 - oo) * hull
        else:
            a = fo * _min_filter(fr, self.render_erode) * (1 - oo) * (1 - orr)
        if self.core:
            p = (po512[INNER] * (Q / LP_D)).astype(np.float32)
            c = np.zeros((Q, Q), np.uint8)
            cv2.fillConvexPoly(c, cv2.convexHull(np.round(p).astype(np.int32)), 255)
            c = cv2.GaussianBlur(c.astype(np.float32) / 255, (0, 0), 1.0)
            a = np.maximum(a, c * (1 - ho) * (1 - oo))
        if self.forehead_max is not None:
            from .surrogate_face import _hull_mask
            cap = _hull_mask(po512 * (Q / LP_D), Q, self.forehead_max).astype(np.float32) / 255
            a = a * cv2.GaussianBlur(cap, (0, 0), 3.0)
        return a

    def _prerender_v2(self):
        super()._prerender_v2()
        self._parse_masks()

    def _parse_masks(self):
        from .face_reenact.engine import m_o2c
        fp = self._parser()
        need = {}
        for tid, plan in self.plan.items():
            for g, st in plan.items():
                if "A" in st and "img" in st:
                    need.setdefault(g, []).append((tid, st))
        if not need:
            return
        S2 = np.diag([Q / LP_D, Q / LP_D, 1.0])
        per = {}   # tid -> list of (g, st, crop_o, crop_r)
        cap = cv2.VideoCapture(self._video)
        f, last = 0, max(need)
        while f <= last:
            ok, fr = cap.read()
            if not ok:
                break
            for tid, st in need.get(f, []):
                co = cv2.warpAffine(fr, m_o2c(st["rect"], Q), (Q, Q), flags=cv2.INTER_AREA,
                                    borderMode=cv2.BORDER_REPLICATE)
                out = cv2.imdecode(st["img"], cv2.IMREAD_COLOR)
                A = (S2 @ np.vstack([st["A"], [0, 0, 1]]) @ np.linalg.inv(S2))[:2]
                cr = cv2.warpAffine(cv2.resize(out, (Q, Q), interpolation=cv2.INTER_AREA), A, (Q, Q),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                per.setdefault(tid, []).append((f, st, co, cr))
            f += 1
        cap.release()
        for tid, L in per.items():
            Po = fp.parse([x[2] for x in L])
            Pr = fp.parse([x[3] for x in L])
            al = np.stack([self._alpha(Po[i], Pr[i], L[i][1]["po"]) for i in range(len(L))])
            if self.fill:
                aln = np.stack([self._alpha(Po[i], Pr[i], L[i][1]["po"], fill=False) for i in range(len(L))])
                for i, (g, st, _, _) in enumerate(L):
                    out = cv2.imdecode(st["img"], cv2.IMREAD_COLOR)
                    out = cv2.warpAffine(out, st["A"], (LP_D, LP_D), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
                    st["img_fill"] = cv2.imencode(".jpg", self._fill_render(out, Pr[i]),
                                                  [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
            keep = Po[..., 2]
            fs = [x[0] for x in L]
            brk = [0] + [i for i in range(1, len(fs)) if fs[i] - fs[i - 1] > 1] + [len(fs)]
            for a_, b_ in zip(brk[:-1], brk[1:]):
                al[a_:b_] = _tsmooth(al[a_:b_], self.mask_tsigma)
                if self.fill:
                    aln[a_:b_] = _tsmooth(aln[a_:b_], self.mask_tsigma)
                keep[a_:b_] = _tsmooth(keep[a_:b_], self.mask_tsigma)
            for i, (g, st, _, _) in enumerate(L):
                a = al[i]
                if self.mask_ssigma > 0:
                    a = cv2.GaussianBlur(a, (0, 0), self.mask_ssigma)
                st["mask_v3"] = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
                if self.fill:
                    an = cv2.GaussianBlur(aln[i], (0, 0), self.mask_ssigma) if self.mask_ssigma > 0 else aln[i]
                    st["mask_v3n"] = (np.clip(an, 0, 1) * 255 + 0.5).astype(np.uint8)
                st["keep_v3"] = (np.clip(keep[i], 0, 1) * 255 + 0.5).astype(np.uint8)
                st["_v2id"] = id(st["mask_v2"])
                self.v3_stats["frames"] += 1
                self.v3_stats["alpha_mean"] += float(a.mean())
        self._log(f"v3 parsing masks: {self.v3_stats['frames']} track-frames, {fp.n} crops parsed")

    def _mask_for(self, step, live=False):
        m = step["mask_v3n" if live and "mask_v3n" in step else "mask_v3"].astype(np.float32) / 255
        if step.get("_v2id") != id(step["mask_v2"]):
            # identity-guard "cover more" retry: add the v2 landmark mask (no forehead band)
            mv = step["mask_v2"].astype(np.float32) / 255
            core = cv2.resize(mv[..., 0] * (1 - mv[..., 1]), (Q, Q), interpolation=cv2.INTER_LINEAR)
            m = np.maximum(m, core * (1 - step["keep_v3"].astype(np.float32) / 255))
        return m

    # ------------------------------------------------------------------ paste
    def _paste_v2(self, frame, tid, step, out=None) -> bool:
        if "mask_v3" not in step:
            return super()._paste_v2(frame, tid, step, out)
        from .face_reenact.engine import m_o2c, inv2x3, tpts
        live = out is not None
        Mc2o = np.vstack([inv2x3(m_o2c(step["rect"], LP_D)), [0, 0, 1]])
        if not live and "img_fill" in step:     # already aligned (A applied) and filled
            out = cv2.imdecode(step["img_fill"], cv2.IMREAD_COLOR)
            Mr2o = Mc2o
        else:
            if out is None:
                out = cv2.imdecode(step["img"], cv2.IMREAD_COLOR)
            Mr2o = Mc2o @ np.vstack([step["A"], [0, 0, 1]])
        Mq2o = Mc2o @ np.diag([LP_D / Q, LP_D / Q, 1.0])
        mq = self._mask_for(step, live)
        ys, xs = np.nonzero(mq > 0.01)
        if len(xs) == 0:
            return False
        H, W = frame.shape[:2]
        pad = 6
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
        ws = cv2.warpAffine(out, (T @ Mr2o)[:2], sz, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = cv2.warpAffine(mq, (T @ Mq2o)[:2], sz, flags=cv2.INTER_LINEAR)
        roi = frame[Y1:Y2, X1:X2]
        roi[:] = self._composite(ws, roi, m, np.zeros_like(m), tid, self._frame_seed(tid, step))
        return True

    def _composite(self, ws, roi, m, band, tid, seed=0):
        if self.blend3 != "laplacian" or band.max() > 0.05:
            return super()._composite(ws, roi, m, band, tid, seed)
        ws = self._match_appearance(ws, roi, m, tid, seed)
        # restrict the difference to the (slightly dilated) mask so no surrogate
        # background / hair can leak through the coarse pyramid levels
        sup = cv2.dilate((m > 0.02).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        ws = (ws.astype(np.float32) * sup[..., None] + roi.astype(np.float32) * (1 - sup[..., None]))
        return lap_blend_diff(ws, roi, m, levels=3)

    # ------------------------------------------------------------------ candidate scoring
    def _candidate_scorer(self, tid, rect_kf, p203_kf, lm5_kf, mot, crops256, rects_t, track, kf,
                          n_samples: int = 16):
        """As v2, but each candidate is pasted with the v3 parsing mask."""
        if not self.parse_scorer:
            return super()._candidate_scorer(tid, rect_kf, p203_kf, lm5_kf, mot, crops256, rects_t,
                                             track, kf, n_samples)
        from .face_reenact.engine import m_o2c, tpts
        from .surrogate_face import _fit_outline
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
        fp = self._parser()
        Po = fp.parse([s[1] for s in samples])
        box = [0, 0, S, S]

        def score(im):
            self._make_source(tid, im, rect_kf, p203_kf, lm5_kf)
            src = self.src[tid]
            outs = []
            for i, oc, po, lm5 in samples:
                out = cv2.resize(E.render(src, **self._hybrid_drv(src["raw"], Rall[i], exp[i], exp_med,
                                                                   ts[i], sc_med)), (S, S),
                                 interpolation=cv2.INTER_AREA)
                pr = E.landmarks203(out, prev203=po)
                A = _fit_outline(pr, po)
                if A is not None and self.outline_align > 0:
                    A = (1 - self.outline_align) * np.float64([[1, 0, 0], [0, 1, 0]]) + self.outline_align * A
                    out = cv2.warpAffine(out, A, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                outs.append(out)
            Pr = fp.parse(outs)
            cs = []
            for k, (i, oc, po, lm5) in enumerate(samples):
                m = cv2.GaussianBlur(self._alpha(Po[k], Pr[k], po * (LP_D / S)), (0, 0), self.mask_ssigma)
                ok_ = self._fill_render(outs[k], Pr[k]) if self.fill else outs[k]
                bl = self._composite(ok_, oc, m, np.zeros_like(m), ("eval", tid))
                cs.append(self._id_cos(oc, bl, box, lm5.astype(np.float32), {"lmk": po}))
            self._tex.pop(("eval", tid), None)
            return float(np.percentile(cs, 75))
        return score
