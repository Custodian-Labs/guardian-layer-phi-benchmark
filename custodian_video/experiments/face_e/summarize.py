"""Collect face_e results into summary.json + summary.md (before = v1, after = v2)."""
import json
from pathlib import Path

E = Path(__file__).resolve().parent
OUT = E / "out"
CLIPS = ["clip1", "clip4", "clip5", "clip6"]


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def main():
    S = {}
    for k in CLIPS:
        id2, bd, mo, vlm = (load(OUT / f"{k}.{x}.json") for x in ("id2", "boundary", "motion", "vlm"))
        for v in ("v1", "v2"):
            t = load(OUT / f"{k}_{v}.timing.json")
            if not t:
                continue
            n = t["render_fps_incl_decode_encode"] * t["render_s"]
            st = t["backend_stats"]
            tot = max(1, sum(st.get(x, 0) for x in ("reenact", "static", "none")))
            r = {
                "sface_reid": id2.get(v, {}).get("sface", {}).get("reid_rate"),
                "sface_max": id2.get(v, {}).get("sface", {}).get("max"),
                "sface_mean": id2.get(v, {}).get("sface", {}).get("mean"),
                "arc_reid": id2.get(v, {}).get("arc", {}).get("reid_rate"),
                "arc_max": id2.get(v, {}).get("arc", {}).get("max"),
                "seam_dE_flat": bd.get(v, {}).get("dE_flat"), "seam_frac_flat": bd.get(v, {}).get("seam_flat"),
                "dE_excess": bd.get(v, {}).get("dE_excess"),
                "corr_mouth": mo.get(v, {}).get("corr_mouth_open"), "corr_blink": mo.get(v, {}).get("corr_eyeBlink"),
                "corr_yaw": mo.get(v, {}).get("corr_yaw"), "corr_pitch": mo.get(v, {}).get("corr_pitch"),
                "flicker_ratio": mo.get(v, {}).get("flicker_ratio_vs_original"),
                "landmark_detect": mo.get(v, {}).get("landmarker_detect_rate_on_output"),
                "frac_reenact": round(st.get("reenact", 0) / tot, 3), "frac_static": round(st.get("static", 0) / tot, 3),
                "frac_blur": round(st.get("none", 0) / tot, 3),
                "prepare_s": round(t["prepare_s"], 1), "face_apply_fps": round(t["face_apply_fps"], 1),
                "face_channel_fps": round(n / (t["prepare_s"] + t["render_s"]), 2),
            }
            S.setdefault(k, {})[v] = r
        if vlm:
            S.setdefault(k, {})["vlm"] = {x: vlm.get(x) for x in ("n", "valid", "v2_preferred", "v1_preferred",
                                                                 "v2_win_rate", "position_bias_chose_image1")}
    (E / "summary.json").write_text(json.dumps(S, indent=1))
    cols = [("sface_reid", "SFace re-ID"), ("sface_max", "SFace max cos"), ("arc_reid", "ArcFace re-ID (eval)"),
            ("arc_max", "ArcFace max"), ("seam_dE_flat", "seam dE (flat)"), ("seam_frac_flat", "seam px >6dE"),
            ("corr_mouth", "r mouth"), ("corr_blink", "r blink"), ("corr_yaw", "r yaw"),
            ("flicker_ratio", "flicker/orig"), ("frac_reenact", "reenacted"), ("frac_static", "static"),
            ("prepare_s", "prep s"), ("face_apply_fps", "apply fps"), ("face_channel_fps", "face-ch fps")]
    L = ["| clip | var | " + " | ".join(c for _, c in cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for k in CLIPS:
        for v in ("v1", "v2"):
            r = S.get(k, {}).get(v)
            if r:
                L.append(f"| {k} | {v} | " + " | ".join("" if r[c] is None else str(r[c]) for c, _ in cols) + " |")
    L += ["", "| clip | VLM pairs (valid) | v2 preferred | v1 preferred | v2 win rate | chose image 1 |", "|---|---|---|---|---|---|"]
    for k in CLIPS:
        q = S.get(k, {}).get("vlm")
        if q:
            L.append(f"| {k} | {q['n']} ({q['valid']}) | {q['v2_preferred']} | {q['v1_preferred']} | {q['v2_win_rate']} | {q['position_bias_chose_image1']} |")
    (E / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
