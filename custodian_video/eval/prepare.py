"""Stage 1: select evaluation clips (deterministic, seed 0) and normalise them to mp4.

Produces DATA/clips/<dataset>/<id>.mp4, DATA/clips/mot17_gt/<id>.json and the
committed clip list custodian_video/eval/manifest.json.

Datasets
  k400  : Kinetics-400 *validation* shard 0 (1,000 clips). Keep clips where YuNet finds a
          face (score>=0.8, min side >=40 px) in >=50% of 8 sampled frames; pick N with at
          most one clip per class first (class diversity), then fill. Trim to <=10 s, height<=720.
  mot17 : MOT17 train sequences (FRCNN copy; same frames for every detector variant),
          3 evenly spaced 150-frame windows per sequence, GT boxes re-indexed per window.
  vox2  : VoxCeleb2 *test* split (118 identities). For N identities pick a probe utterance
          (trimmed to 6 s, processed by every arm) and, from a DIFFERENT YouTube video of the
          same person, a gallery utterance (kept original). All 118 identities get a gallery
          entry, so the identification gallery has 118 subjects (chance rank-1 = 0.85%).
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import CLIPS, MANIFEST, RAW, FaceModels, read_frames, uniform_idxs, video_info, write_video


def _resize(fr, max_h=720):
    h, w = fr.shape[:2]
    if h <= max_h:
        return fr if (w % 2 == 0 and h % 2 == 0) else fr[: h - h % 2, : w - w % 2]
    s = max_h / h
    nw = int(round(w * s / 2) * 2)
    return cv2.resize(fr, (nw, max_h), interpolation=cv2.INTER_AREA)


def _k400_candidates(fm: FaceModels, shard_dirs: list[str]) -> list[dict]:
    root = RAW / "k400"
    meta = {}
    with open(root / "annotations" / "val.csv") as f:
        for r in csv.DictReader(f):
            vid = f"{r['youtube_id']}_{int(r['time_start']):06d}_{int(r['time_end']):06d}"
            meta[vid] = r
    cands = []
    files = sorted(p for d in shard_dirs for p in (root / d).glob("*.mp4"))
    for i, p in enumerate(files):
        vid = p.stem
        if vid not in meta:
            continue
        info = video_info(p)
        if info["n"] < 32 or info["h"] < 200:
            continue
        frs = read_frames(p, uniform_idxs(min(info["n"], int(info["fps"] * 10)), 8))
        hits, sizes = 0, []
        for fr in frs.values():
            faces = fm.detect(fr, 0.8)
            big = [min(r[2], r[3]) for r in faces if min(r[2], r[3]) >= 40]
            if big:
                hits += 1
                sizes.append(max(big))
        if frs and hits / len(frs) >= 0.5:
            cands.append(dict(vid=vid, label=meta[vid]["label"], is_cc=meta[vid].get("is_cc"),
                              face_px=float(np.median(sizes)), shard=p.parent.name))
        if i % 100 == 0:
            print(f"  k400 screened {i}/{len(files)}  candidates={len(cands)}", flush=True)
    print(f"  k400: {len(cands)} face-visible candidates of {len(files)}")
    return cands


def _k400_write(chosen: list[dict]) -> list[dict]:
    root = RAW / "k400"
    out = []
    for c in chosen:
        src = root / c.get("shard", "val_part0") / f"{c['vid']}.mp4"
        info = video_info(src)
        n = min(info["n"], int(round(info["fps"] * 10)))
        frs = read_frames(src, range(n))
        cid = f"k400_{c['vid']}"
        dst = CLIPS / "k400" / f"{cid}.mp4"
        if not dst.exists():
            write_video((_resize(frs[i]) for i in sorted(frs)), dst, info["fps"])
        vi = video_info(dst)
        out.append(dict(id=cid, dataset="k400", label=c["label"], is_cc=c["is_cc"],
                        face_px_median=c["face_px"], source=f"kinetics400/val/{c['vid']}", **vi))
    return out


def prep_k400_extend(fm: FaceModels, existing: list[dict], n_total: int, seed: int,
                     shards: list[int]) -> list[dict]:
    """Keep the existing K400 clips (so earlier tags stay comparable) and add new ones
    from the given shards up to n_total, preferring classes not yet represented."""
    have = {c["source"].split("/")[-1] for c in existing}
    have_labels = {c["label"] for c in existing}
    cands = [c for c in _k400_candidates(fm, [f"val_part{k}" for k in shards]) if c["vid"] not in have]
    rng = random.Random(seed + 1)
    rng.shuffle(cands)
    seen, first, rest = set(have_labels), [], []
    for c in cands:
        (rest if c["label"] in seen else first).append(c)
        seen.add(c["label"])
    need = max(0, n_total - len(existing))
    added = _k400_write((first + rest)[:need])
    print(f"  k400 extend: kept {len(existing)}, added {len(added)} "
          f"({len({c['label'] for c in added} - have_labels)} new classes)")
    return existing + added


def prep_k400(fm: FaceModels, n_target: int, seed: int) -> list[dict]:
    cands = _k400_candidates(fm, ["val_part0"])
    rng = random.Random(seed)
    rng.shuffle(cands)
    seen, first, rest = set(), [], []
    for c in cands:
        (rest if c["label"] in seen else first).append(c)
        seen.add(c["label"])
    chosen = (first + rest)[:n_target]
    return _k400_write(chosen)


def prep_mot17(win: int = 150, per_seq: int = 3) -> list[dict]:
    root = RAW / "mot17" / "train"
    out = []
    for sd in sorted(root.glob("MOT17-*-FRCNN")):
        ini = configparser.ConfigParser()
        ini.read(sd / "seqinfo.ini")
        L, fps = int(ini["Sequence"]["seqLength"]), float(ini["Sequence"]["frameRate"])
        gt = np.loadtxt(sd / "gt" / "gt.txt", delimiter=",")
        seq = sd.name.replace("-FRCNN", "")
        for k in range(per_seq):
            s = int(round(k * (L - win) / max(1, per_seq - 1))) + 1   # 1-based frame ids
            cid = f"mot17_{seq[6:]}_{s:04d}"
            dst = CLIPS / "mot17" / f"{cid}.mp4"
            if not dst.exists():
                frames = (_resize(cv2.imread(str(sd / "img1" / f"{f:06d}.jpg")), 1080)
                          for f in range(s, s + win))
                write_video(frames, dst, fps)
            g = gt[(gt[:, 0] >= s) & (gt[:, 0] < s + win)]
            boxes = defaultdict(list)
            for r in g:
                # cols: frame,id,x,y,w,h,consider,class,visibility
                boxes[int(r[0]) - s].append([round(float(v), 1) for v in r[2:6]] +
                                            [int(r[6]), int(r[7]), round(float(r[8]), 3)])
            gpath = CLIPS / "mot17_gt" / f"{cid}.json"
            gpath.parent.mkdir(parents=True, exist_ok=True)
            gpath.write_text(json.dumps({str(k2): v for k2, v in boxes.items()}))
            out.append(dict(id=cid, dataset="mot17", label=None, source=f"MOT17/train/{seq} frames {s}-{s + win - 1}",
                            gt=str(gpath.relative_to(CLIPS)), **video_info(dst)))
    return out


def prep_vox2(n_probe: int, seed: int, probe_s: float = 6.0) -> list[dict]:
    root = RAW / "vox2" / "test_mp4"
    mp4 = next(p for p in [root / "mp4", root] if any(p.glob("id*")))
    ids = sorted(p.name for p in mp4.glob("id*"))
    rng = random.Random(seed)
    probe_ids = set(rng.sample(ids, min(n_probe, len(ids))))
    out = []
    for pid in ids:
        vids = sorted(p.name for p in (mp4 / pid).iterdir() if p.is_dir())
        if len(vids) < 2:
            continue
        r = random.Random(f"{seed}-{pid}")
        g_vid, p_vid = r.sample(vids, 2)
        g_utt = sorted((mp4 / pid / g_vid).glob("*.mp4"))[0]
        roles = [("gallery", g_vid, g_utt)]
        if pid in probe_ids:
            # longest utterance of the probe video (more frames for the attacker)
            utts = sorted((mp4 / pid / p_vid).glob("*.mp4"), key=lambda p: -video_info(p)["n"])
            roles.append(("probe", p_vid, utts[0]))
        for role, yt, src in roles:
            info = video_info(src)
            n = min(info["n"], int(round(info["fps"] * probe_s)))
            cid = f"vox2_{pid}_{role}"
            dst = CLIPS / ("vox2" if role == "probe" else "vox2_gallery") / f"{cid}.mp4"
            if not dst.exists():
                frs = read_frames(src, range(n))
                write_video((frs[i] for i in sorted(frs)), dst, info["fps"])
            out.append(dict(id=cid, dataset="vox2" if role == "probe" else "vox2_gallery",
                            label=pid, role=role, youtube_id=yt,
                            source=f"voxceleb2/test/{pid}/{yt}/{src.name}", **video_info(dst)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k400", type=int, default=100)
    ap.add_argument("--vox2", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="mot17,vox2,k400")
    ap.add_argument("--force", action="store_true", help="re-select even if all clips are present")
    ap.add_argument("--k400-extend", type=int, help="keep current K400 clips and add more up to this total")
    ap.add_argument("--k400-shards", default="0,1,2", help="shards scanned by --k400-extend")
    ap.add_argument("--manifest-out", help="write the manifest here instead of manifest.json (staging)")
    a = ap.parse_args()
    clips = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    if a.k400_extend:
        fm = FaceModels(gpu=False)
        k4 = prep_k400_extend(fm, [c for c in clips if c["dataset"] == "k400"], a.k400_extend, a.seed,
                              [int(x) for x in a.k400_shards.split(",")])
        clips = [c for c in clips if c["dataset"] != "k400"] + k4
        out = Path(a.manifest_out) if a.manifest_out else MANIFEST
        out.write_text(json.dumps(clips, indent=1))
        from collections import Counter
        print("manifest ->", out, Counter(c["dataset"] for c in clips))
        raise SystemExit(0)
    have = {c["dataset"].replace("_gallery", "") for c in clips
            if (CLIPS / c["dataset"] / f"{c['id']}.mp4").exists()}
    missing = {c["dataset"].replace("_gallery", "") for c in clips
               if not (CLIPS / c["dataset"] / f"{c['id']}.mp4").exists()}
    fm = FaceModels(gpu=False)
    for ds in a.only.split(","):
        if ds in have and ds not in missing and not a.force:
            print(f"== {ds}: already prepared ({sum(c['dataset'].startswith(ds) for c in clips)} clips)")
            continue
        print(f"== {ds}", flush=True)
        new = {"mot17": lambda: prep_mot17(), "vox2": lambda: prep_vox2(a.vox2, a.seed),
               "k400": lambda: prep_k400(fm, a.k400, a.seed)}[ds]()
        clips = [c for c in clips if not c["dataset"].startswith(ds)] + new
        MANIFEST.write_text(json.dumps(clips, indent=1))
    from collections import Counter
    print("manifest:", Counter(c["dataset"] for c in clips))
