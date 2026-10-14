"""Stage 4: aggregate per-clip measurements into results/<tag>/summary.{md,json}.

  python analyze.py --tag v0.1 [--baseline-tag v0.1]

Statistics mirror the text paper: paired per-clip deltas (arm - original), 95% paired-bootstrap
CI (10,000 resamples over clips), and a TOST equivalence test with a +-2-point margin
(two one-sided z-tests on the paired mean difference; equivalent if p<0.05, which coincides
with the 90% CI lying inside (-2, +2)). Privacy rates get clip-bootstrap 95% CIs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import defaultdict

import numpy as np

from common import CACHE, EMB, RESULTS, load_manifest
from metrics import CACHE_VERSION, METRIC_VERSION

ARMS_ORDER = ["original", "reencode", "redact_blur", "redact_box", "transform"]
N_CAND = 100           # closed-set identification: own original + 99 distractor faces from other clips
MARGIN = 2.0
BIG_FACE = 96          # px, median face size of a Kinetics clip for the "large face" subset
B = 10000
RNG = np.random.default_rng(0)


def boot_ci(x, stat=np.mean, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (math.nan, math.nan)
    idx = RNG.integers(0, len(x), (B, len(x)))
    s = stat(x[idx], axis=1) if stat is np.mean else np.array([stat(x[i]) for i in idx])
    return (float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2)))


def tost(d, margin=MARGIN):
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return math.nan, False
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    se = max(se, 1e-12)
    p1 = 0.5 * math.erfc(((m + margin) / se) / math.sqrt(2))     # H0: mu <= -margin
    p2 = 0.5 * math.erfc((-(m - margin) / se) / math.sqrt(2))    # H0: mu >= +margin
    p = max(p1, p2)
    return p, p < 0.05


def load(tag):
    per = defaultdict(dict)   # arm -> clip -> res
    npz = defaultdict(dict)
    for arm in ARMS_ORDER:
        d = RESULTS / tag / "per_clip" / arm
        if not d.exists():
            continue
        for p in d.glob("*.json"):
            r = json.loads(p.read_text())
            if r.get("missing") or r.get("metric_version") != METRIC_VERSION:
                continue
            per[arm][r["clip"]] = r
            q = EMB / tag / arm / f"{p.stem}.npz"
            if q.exists():
                npz[arm][r["clip"]] = dict(np.load(q))
    return per, npz


def thresholds():
    g = np.load(CACHE / f"gallery_{CACHE_VERSION}.npz")
    out = {"gallery_ids": int(len(set(g["ids"].tolist()))), "gallery_frames": int(len(g["ids"]))}
    for k in ["arc", "sf"]:
        E = g[k]
        S = E @ E.T
        diff = g["ids"][:, None] != g["ids"][None, :]
        imp = S[np.triu(diff, 1)]
        out[f"thr_{k}_far1e-3"] = float(np.quantile(imp, 0.999))
        out[f"n_impostor_pairs_{k}"] = int(len(imp))
    out["thr_sf_opencv_default"] = 0.363
    return out


def gallery_means():
    g = np.load(CACHE / f"gallery_{CACHE_VERSION}.npz")
    ids = sorted(set(g["ids"].tolist()))
    M = np.stack([g["arc"][g["ids"] == i].mean(0) for i in ids])
    return ids, M / np.linalg.norm(M, axis=1, keepdims=True)


# ----------------------------------------------------------------------------- privacy
def privacy_clip(r, thr, min_size=0):
    fs = [f for f in r.get("faces", []) if f["size"] >= min_size]
    if not fs:
        return None
    m = lambda xs: float(np.mean(xs)) * 100
    out = dict(n_faces=len(fs),
               reid_same=m([f["sim_same"] >= thr["thr_arc_far1e-3"] for f in fs]),
               reid_same_sf=m([f["sim_same_sf"] >= thr["thr_sf_opencv_default"] for f in fs]),
               detected=m([f["detected"] for f in fs]),
               leak=m([f["detected"] and f["det_sim"] is not None and f["det_sim"] >= thr["thr_arc_far1e-3"]
                       for f in fs]),
               sim_same=float(np.mean([f["sim_same"] for f in fs])))
    cov = [f["covered"] for f in fs if f.get("covered") is not None]
    out["covered"] = m(cov) if cov else None
    return out


def identification(npz_arm, ids, G, key):
    """rank-1 over the 118-identity gallery for each vox2 probe clip; returns clip->dict"""
    out = {}
    for cid, d in npz_arm.items():
        if key not in d:          # only vox2 probes carry identification embeddings
            continue
        pid = cid.split("_")[1]
        E = d[key]
        if len(E) == 0:
            out[cid] = dict(rank1_clip=0.0, rank1_frame=0.0, n=0)
            continue
        q = E.mean(0); q /= np.linalg.norm(q) + 1e-9
        s = G @ q
        fr = (np.argmax(E @ G.T, axis=1) == ids.index(pid)).mean()
        out[cid] = dict(rank1_clip=float(ids[int(np.argmax(s))] == pid) * 100, rank1_frame=float(fr) * 100,
                        genuine_cos=float(s[ids.index(pid)]), n=len(E))
    return out


def closed_set(per, npz, arm, thr, seed=0):
    """Per masked face: is its own original the best match among N_CAND candidates (own original +
    distractor ORIGINAL faces from other clips of the same dataset)? Also the impostor match rate
    (distractor cos >= FAR-1e-3 threshold), i.e. the re-ID rate expected with no identity signal.
    Returns clip -> {rank1, impostor, rank1_ge64, impostor_ge64} (percent)."""
    rng = np.random.default_rng(seed)
    pool = defaultdict(list)
    for cid, d in npz.get("original", {}).items():
        ds = per["original"][cid]["dataset"]
        for e in d["arc_o"]:
            pool[ds].append((cid, e))
    P = {ds: (np.array([c for c, _ in v]), np.stack([e for _, e in v]).astype(np.float32)) for ds, v in pool.items()}
    out = {}
    for cid, d in npz.get(arm, {}).items():
        r = per[arm].get(cid)
        if r is None or len(d["arc_m"]) == 0 or r["dataset"] not in P:
            continue
        cl, E = P[r["dataset"]]
        others = np.where(cl != cid)[0]
        if len(others) < N_CAND - 1:
            continue
        M = d["arc_m"].astype(np.float32); O = d["arc_o"].astype(np.float32)
        sizes = np.array([f["size"] for f in r["faces"]])
        rk, imp = [], []
        for k in range(len(M)):
            dist = E[rng.choice(others, N_CAND - 1, replace=False)]
            ds_ = dist @ M[k]
            rk.append(float(M[k] @ O[k] > ds_.max())); imp.append(float((ds_ >= thr).mean()))
        rk, imp = np.array(rk), np.array(imp)
        e = dict(rank1=rk.mean() * 100, impostor=imp.mean() * 100)
        if (sizes >= 64).any():
            e.update(rank1_ge64=rk[sizes >= 64].mean() * 100, impostor_ge64=imp[sizes >= 64].mean() * 100)
        out[cid] = e
    return out


# ----------------------------------------------------------------------------- utility
UTILITY = [  # (dataset, name, getter, scale, higher_is_better)
    ("k400", "Action top-1, VideoMAE-B (%)", lambda r: r["videomae"]["top1"], 100, True),
    ("k400", "Action top-5, VideoMAE-B (%)", lambda r: r["videomae"]["top5"], 100, True),
    ("k400", "Action top-1, X-CLIP B/32 (%)", lambda r: r["xclip"]["top1"], 100, True),
    ("k400", "Action top-5, X-CLIP B/32 (%)", lambda r: r["xclip"]["top5"], 100, True),
    ("k400", "VideoMAE prediction = original's (%)", lambda r: r["videomae"]["agree_orig"], 100, True),
    ("k400", "Pose OKS vs original, all 17 kpts (x100)", lambda r: r["pose"]["oks"], 100, True),
    ("k400", "Pose OKS vs original, 5 head kpts (x100)", lambda r: r["pose"]["oks_head"], 100, True),
    ("k400", "Person recall vs original (%)", lambda r: r["pose"]["person_recall"], 100, True),
    ("mot17", "Person AP50 vs MOT17 GT (x100)", lambda r: r["mot"]["ap50"], 100, True),
    ("mot17", "Person recall@0.5 vs GT (%)", lambda r: r["mot"]["recall"], 100, True),
    ("mot17", "Person count MAE / frame (lower=better)", lambda r: r["mot"]["count_mae"], 1, False),
]


def get(fn, r):
    try:
        v = fn(r)
        return None if v is None else float(v)
    except (KeyError, TypeError):
        return None


def paired(per, arm, ds, fn, scale, keep=None):
    xs, ys = [], []
    for cid, r0 in per["original"].items():
        if r0["dataset"] != ds or cid not in per.get(arm, {}) or (keep is not None and cid not in keep):
            continue
        a, b = get(fn, r0), get(fn, per[arm][cid])
        if a is None or b is None:
            continue
        xs.append(a * scale); ys.append(b * scale)
    return np.array(xs), np.array(ys)


def fmt(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--baseline-tag", help="also tabulate headline numbers of an earlier tag")
    a = ap.parse_args()
    per, npz = load(a.tag)
    arms = [x for x in ARMS_ORDER if x in per]
    thr = thresholds()
    ids, G = gallery_means()
    manifest = load_manifest()
    S = dict(tag=a.tag, created=dt.datetime.now().isoformat(timespec="seconds"), thresholds=thr,
             n_clips={ds: sum(1 for c in manifest if c["dataset"] == ds) for ds in ("k400", "mot17", "vox2")},
             privacy={}, identification={}, utility={}, transform_vs_redact={}, audit={}, runtime={})
    info_p = RESULTS / a.tag / "run_info.json"
    run_info = json.loads(info_p.read_text()) if info_p.exists() else {}

    # privacy
    for arm in arms:
        cs = closed_set(per, npz, arm, thr["thr_arc_far1e-3"])
        for ds in ["k400", "mot17", "vox2", "all"]:
            for min_size, sub in [(0, "all faces"), (64, "faces >=64px")]:
                rows = []
                for r in per[arm].values():
                    if ds != "all" and r["dataset"] != ds:
                        continue
                    x = privacy_clip(r, thr, min_size)
                    if not x:
                        continue
                    c = cs.get(r["clip"], {})
                    sfx = "" if min_size == 0 else "_ge64"
                    x["rank1_100"] = c.get("rank1" + sfx)
                    x["impostor"] = c.get("impostor" + sfx)
                    rows.append(x)
                if not rows:
                    continue
                e = dict(n_clips=len(rows), n_faces=int(sum(x["n_faces"] for x in rows)))
                for k in ["reid_same", "reid_same_sf", "leak", "detected", "covered", "sim_same", "rank1_100", "impostor"]:
                    v = [x[k] for x in rows if x[k] is not None]
                    if v:
                        e[k] = float(np.mean(v)); e[k + "_ci"] = boot_ci(v)
                S["privacy"][f"{arm}|{ds}|{sub}"] = e
        if arm in npz:
            for key, name in [("probe_same", "same-location attacker"), ("probe_det", "detector attacker")]:
                idr = identification(npz[arm], ids, G, key)
                if not idr:
                    continue
                e = {}
                for k in ["rank1_clip", "rank1_frame"]:
                    v = [x[k] for x in idr.values()]
                    e[k] = float(np.mean(v)); e[k + "_ci"] = boot_ci(v)
                e["n_probes"] = len(idr)
                S["identification"][f"{arm}|{name}"] = e

    # utility (+ Kinetics subset whose median face is large: where masking could matter most)
    big = {c["id"] for c in manifest if c["dataset"] == "k400" and c.get("face_px_median", 0) >= BIG_FACE}
    S["n_k400_bigface"] = len(big)
    util = list(UTILITY) + [(ds, f"{name} [face>={BIG_FACE}px subset]", fn, scale, hib, big)
                            for ds, name, fn, scale, hib in UTILITY if ds == "k400" and "top-1" in name]
    for ds, name, fn, scale, hib, *kp in util:
        keep = kp[0] if kp else None
        for arm in arms[1:]:
            x, y = paired(per, arm, ds, fn, scale, keep)
            if len(x) == 0:
                continue
            d = y - x
            p, eq = tost(d) if scale == 100 else (math.nan, None)
            S["utility"][f"{name}|{arm}"] = dict(
                dataset=ds, n=len(d), orig=float(x.mean()), arm=float(y.mean()), delta=float(d.mean()),
                ci95=boot_ci(d), ci90=boot_ci(d, alpha=0.10), tost_p=p, equivalent=eq, higher_is_better=hib)
        if "transform" in per and "redact_blur" in per:
            _, yt = paired(per, "transform", ds, fn, scale, keep)
            _, yb = paired(per, "redact_blur", ds, fn, scale, keep)
            if len(yt) == len(yb) and len(yt):
                d = yt - yb
                S["transform_vs_redact"][name] = dict(n=len(d), delta=float(d.mean()), ci95=boot_ci(d))

    # audit diagnostics + runtime
    for arm in arms[1:]:
        for ds in ["k400", "mot17", "vox2"]:
            sts = [r.get("audit", {}) for r in per[arm].values() if r["dataset"] == ds]
            ntr = sum(s.get("face_tracks", 0) for s in sts)
            S["audit"][f"{arm}|{ds}"] = dict(
                face_tracks=ntr, tracks_surrogate=sum(s.get("tracks_surrogate", 0) for s in sts),
                tracks_blur_fallback=sum(s.get("tracks_blur_fallback", 0) for s in sts),
                fps_median=float(np.median([s["timing"]["fps_end_to_end"] for s in sts
                                            if s.get("timing", {}).get("fps_end_to_end")] or [math.nan])))
    rp = RESULTS / a.tag / "runs.jsonl"
    if rp.exists():
        runs = [json.loads(l) for l in rp.read_text().splitlines() if l.strip()]
        last = {}
        for r in runs:
            last[(r["clip"], r["arm"])] = r
        for arm in arms[1:]:
            rr = [r for r in last.values() if r["arm"] == arm]
            if rr:
                ok = [r for r in rr if r["rc"] == 0]
                S["runtime"][arm] = dict(runs=len(rr), failed=len(rr) - len(ok),
                                         wall_s_total=float(sum(r["wall_s"] for r in ok)),
                                         wall_s_median=float(np.median([r["wall_s"] for r in ok])) if ok else None,
                                         frames=int(sum(r["frames"] for r in ok)))
    (RESULTS / a.tag / "summary.json").write_text(json.dumps(S, indent=1))
    md = render(S, run_info, a.baseline_tag)
    (RESULTS / a.tag / "summary.md").write_text(md)
    print(md)


def render(S, info, baseline_tag=None):
    L = []
    w = L.append
    w(f"# Video Guardian benchmark: results for pipeline tag `{S['tag']}`\n")
    w(f"Generated {S['created']} by `custodian_video/eval/analyze.py` (metric version {METRIC_VERSION}). "
      f"Pipeline snapshot: source `{info.get('pipeline', {}).get('source', '?')}`, commit `{info.get('pipeline', {}).get('commit', '?')[:10]}`, "
      f"files sha256 `{info.get('pipeline', {}).get('files_sha256', '?')}`, reported version `{info.get('custodian_video_version', '?').split(' ')[0]}`. Only the face channel is enabled (`--channels faces`).\n")
    n = S["n_clips"]
    w(f"Clips: Kinetics-400 val **{n['k400']}** (action, pose), MOT17 **{n['mot17']}** (person detection/count), "
      f"VoxCeleb2 test **{n['vox2']}** probes vs a {S['thresholds']['gallery_ids']}-identity gallery (identification). "
      f"Arms: original, `reencode` (null arm: same pipeline + encoder, nothing masked = codec-noise floor), `redact_blur` (mosaic+blur), `redact_box` (solid ellipse), `transform` (SD-1.5 surrogate face per track; blur fallback for faces <64 px or score <0.6).\n")
    t = S["thresholds"]
    w(f"ArcFace-R50 threshold at FAR=1e-3 (calibrated on {t['n_impostor_pairs_arc']:,} impostor pairs from the gallery): "
      f"**{t['thr_arc_far1e-3']:.3f}**; SFace uses OpenCV's 0.363.\n")

    w("\n## 1. Privacy (lower is better)\n")
    w("Per reference face (YuNet on the original frame, >=24 px), pooled as the mean of per-clip rates with 95% clip-bootstrap CIs.\n")
    w("- **Re-ID (same location)**: ArcFace cosine between the original face and the output pixels at the same landmarks >= FAR-1e-3 threshold. The attacker knows exactly where the face is, so this is a worst case.")
    w("- **Leak**: a face detector still finds a face there *and* it matches the original identity.")
    w("- **Face present**: a face detector still finds *a* face there. For privacy this alone is harmless; it is the utility goal of `transform`.")
    w("- **Impostor %**: how often the output face matches a *different* person's original face (from another clip of the same dataset) at the same threshold, i.e. the re-ID rate expected with no identity signal left. Re-ID should be compared with this floor, which is high for small, low-detail faces.")
    w(f"- **Rank-1 of {N_CAND}**: closed-set identification. Is the true original the best match among itself plus {N_CAND - 1} distractor originals? This is threshold-free; chance is {100 / N_CAND:.0f}%.")
    w("- **Mask coverage**: the audit log has a face mask containing the face centre (detection recall).\n")
    w(f"| Arm | Subset | clips / faces | Re-ID ArcFace % | Impostor % | Rank-1 of {N_CAND} % | Re-ID SFace % | Leak % | Face present % | Mask coverage % | mean cos |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, e in S["privacy"].items():
        arm, ds, sub = key.split("|")
        if ds != "all":
            continue
        ci = lambda k: f"{fmt(e.get(k))} [{fmt(e[k + '_ci'][0])}, {fmt(e[k + '_ci'][1])}]" if k in e else "–"
        w(f"| {arm} | {sub} | {e['n_clips']} / {e['n_faces']} | {ci('reid_same')} | {ci('impostor')} | {ci('rank1_100')} | {ci('reid_same_sf')} | {ci('leak')} | "
          f"{ci('detected')} | {ci('covered')} | {fmt(e.get('sim_same'), 3)} |")
    w(f"\nBy dataset (all faces): re-ID ArcFace % / impostor % / rank-1 of {N_CAND} % / leak % / face present %\n")
    w("| Arm | Kinetics-400 | MOT17 | VoxCeleb2 |")
    w("|---|---|---|---|")
    arms = sorted({k.split("|")[0] for k in S["privacy"]}, key=ARMS_ORDER.index)
    for arm in arms:
        cells = []
        for ds in ["k400", "mot17", "vox2"]:
            e = S["privacy"].get(f"{arm}|{ds}|all faces")
            cells.append("–" if not e else f"{fmt(e.get('reid_same'))} / {fmt(e.get('impostor'))} / {fmt(e.get('rank1_100'))} / "
                         f"{fmt(e.get('leak'))} / {fmt(e.get('detected'))}")
        w(f"| {arm} | " + " | ".join(cells) + " |")
    if S["identification"]:
        w(f"\n**Cross-video identification (VoxCeleb2).** Each probe clip is identified against a gallery built from a *different* "
          f"YouTube video of each of {t['gallery_ids']} people (chance = {100 / t['gallery_ids']:.2f}%). "
          "Rank-1 % with 95% CI. The clip-level score uses the mean embedding over 16 frames.\n")
        w("| Arm | Attacker | probes | rank-1 per clip % | rank-1 per frame % |")
        w("|---|---|---|---|---|")
        for key, e in S["identification"].items():
            arm, att = key.split("|")
            w(f"| {arm} | {att} | {e['n_probes']} | {fmt(e['rank1_clip'])} [{fmt(e['rank1_clip_ci'][0])}, {fmt(e['rank1_clip_ci'][1])}] | "
              f"{fmt(e['rank1_frame'])} [{fmt(e['rank1_frame_ci'][0])}, {fmt(e['rank1_frame_ci'][1])}] |")

    w("\n## 2. Utility: original vs each arm (paired per clip)\n")
    w(f"Δ = arm − original (points). 95% paired-bootstrap CI. TOST equivalence margin ±{MARGIN:.0f} points (α=0.05): "
      "**EQ** = equivalent; ✗ = not shown equivalent.\n")
    w("| Metric | n | Original | Arm | Value | Δ [95% CI] | TOST p | Equiv. ±2 |")
    w("|---|---|---|---|---|---|---|---|")
    for key, e in S["utility"].items():
        name, arm = key.split("|")
        eq = "–" if e["equivalent"] is None else ("**EQ**" if e["equivalent"] else "✗")
        w(f"| {name} | {e['n']} | {fmt(e['orig'])} | {arm} | {fmt(e['arm'])} | {fmt(e['delta'], 2)} "
          f"[{fmt(e['ci95'][0], 2)}, {fmt(e['ci95'][1], 2)}] | {fmt(e['tost_p'], 3)} | {eq} |")
    if S["transform_vs_redact"]:
        w("\n**Transform − redact_blur** (paired, 95% CI):\n")
        w("| Metric | n | Δ [95% CI] |")
        w("|---|---|---|")
        for name, e in S["transform_vs_redact"].items():
            w(f"| {name} | {e['n']} | {fmt(e['delta'], 2)} [{fmt(e['ci95'][0], 2)}, {fmt(e['ci95'][1], 2)}] |")

    w("\n## 3. Pipeline diagnostics\n")
    w("| Arm | Dataset | face tracks | tracks given a surrogate | tracks blur fallback | median fps (audit) |")
    w("|---|---|---|---|---|---|")
    for key, e in S["audit"].items():
        arm, ds = key.split("|")
        w(f"| {arm} | {ds} | {e['face_tracks']} | {e['tracks_surrogate']} | {e['tracks_blur_fallback']} | {fmt(e['fps_median'])} |")
    if S["runtime"]:
        w("\n| Arm | runs | failed | total wall (min) | median wall / clip (s) | frames |")
        w("|---|---|---|---|---|---|")
        for arm, e in S["runtime"].items():
            w(f"| {arm} | {e['runs']} | {e['failed']} | {e['wall_s_total'] / 60:.1f} | {fmt(e['wall_s_median'])} | {e['frames']} |")
    if baseline_tag:
        bp = RESULTS / baseline_tag / "summary.json"
        if bp.exists():
            Bs = json.loads(bp.read_text())
            w(f"\n## 4. Change vs `{baseline_tag}`\n")
            w(f"| Headline | {baseline_tag} | {S['tag']} |")
            w("|---|---|---|")
            for arm in ["redact_blur", "redact_box", "transform"]:
                for k, lab in [("reid_same", "re-ID %"), ("leak", "leak %"), ("detected", "face present %")]:
                    o = Bs["privacy"].get(f"{arm}|all|all faces", {}).get(k)
                    nw = S["privacy"].get(f"{arm}|all|all faces", {}).get(k)
                    w(f"| {arm} {lab} | {fmt(o)} | {fmt(nw)} |")
                for name in [u[1] for u in UTILITY]:
                    o = Bs["utility"].get(f"{name}|{arm}", {}).get("delta")
                    nw = S["utility"].get(f"{name}|{arm}", {}).get("delta")
                    if o is not None or nw is not None:
                        w(f"| {arm} Δ {name} | {fmt(o, 2)} | {fmt(nw, 2)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
