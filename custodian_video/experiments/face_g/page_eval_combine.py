"""Combine page_eval.sh outputs into <root>/surrogate_research_eval.json (numbers only)."""
import json, sys
from pathlib import Path
W, out = Path(sys.argv[1]), Path(sys.argv[2])
ld = lambda p: json.loads(p.read_text()) if p.exists() else {}
cross = ld(W / "cross_floor.json").get("surrogate_research", {})
res = {"method": "Same-place re-identification of the original face in the output (YuNet reference faces inside the "
                 "masked face boxes, aligned at the original's landmarks) with SFace (thr 0.363), ArcFace-R50 "
                 "(InsightFace buffalo_l, eval only, thr 0.217 = FAR 1e-3) and glintr100 (antelopev2, eval only, "
                 "independent of the swapper's identity space; thr = FAR 1e-3 on VoxCeleb2 gallery impostor pairs). "
                 "Floors: cross-clip (original faces of one clip vs the output faces of another, i.e. different "
                 "people; clips 1/4/5) and in-frame (another person's face in the same output frame; clips 2/6). "
                 "Motion: per-track correlation of MediaPipe mouth opening / blink / yaw / brow raise between the "
                 "original and the output.", "clips": {}}
for p in sorted(W.glob("*.id3.json")):
    c = p.name[:-len(".id3.json")]
    d = ld(p); r = {}
    for k in ("surrogate_research", "blur"):
        if k in d:
            r[k] = {m: {"reid": v["reid_rate"], "max": v["max"], "n": v["n"]} for m, v in d[k].items()}
    inf = ld(W / f"{c}.inframe.json").get("surrogate_research")
    if inf:
        r["floor_inframe"] = {m: v["inframe_floor"] for m, v in inf.items()}
        r["floor_inframe_pairs"] = inf["sface"]["n_pairs"]
    elif cross:
        r["floor_cross"] = {m: v["floor"] for m, v in cross.items()}
        r["floor_cross_pairs"] = cross["sface"]["pairs"]
    mo = ld(W / f"{c}.motion.json").get("surrogate_research")
    if mo:
        r["motion"] = {k[5:]: mo.get(k) for k in ("corr_mouth_open", "corr_eyeBlink", "corr_yaw", "corr_browInnerUp")}
    res["clips"][c] = r
out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)[:3000])
