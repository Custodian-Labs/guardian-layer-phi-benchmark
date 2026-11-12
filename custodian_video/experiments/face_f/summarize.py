"""Collect face_f results into summary.json + summary.md."""
import json
from pathlib import Path

E = Path(__file__).resolve().parent
OUT = E / "out"
CLIPS = ["clip1", "clip4", "clip5"]


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def main():
    S = {}
    for k in CLIPS:
        id2, bd, mo = (load(OUT / f"{k}.{x}.json") for x in ("id2", "boundary", "motion"))
        for t in sorted(OUT.glob(f"{k}_*.timing.json")):
            v = t.name[len(k) + 1:-len(".timing.json")]
            T = load(t); st = T["backend_stats"]
            tot = max(1, sum(st.get(x, 0) for x in ("reenact", "static", "none")))
            vl, vo = load(OUT / f"{k}.vlm_{v}.json"), load(OUT / f"{k}.vlm_orig_{v}.json")
            S.setdefault(k, {})[v] = {
                "sface_reid": id2.get(v, {}).get("sface", {}).get("reid_rate"),
                "sface_max": id2.get(v, {}).get("sface", {}).get("max"),
                "arc_reid": id2.get(v, {}).get("arc", {}).get("reid_rate"),
                "arc_max": id2.get(v, {}).get("arc", {}).get("max"),
                "seam_dE_flat": bd.get(v, {}).get("dE_flat"), "seam_frac_flat": bd.get(v, {}).get("seam_flat"),
                "corr_mouth": mo.get(v, {}).get("corr_mouth_open"), "corr_blink": mo.get(v, {}).get("corr_eyeBlink"),
                "corr_yaw": mo.get(v, {}).get("corr_yaw"), "flicker_ratio": mo.get(v, {}).get("flicker_ratio_vs_original"),
                "frac_surrogate": round(st.get("reenact", 0) / tot, 3), "guard_blur": st.get("guard_blur", 0),
                "prepare_s": round(T["prepare_s"], 1),
                "vlm_win_vs_v2": vl.get(f"{v}_win_rate"), "vlm_valid": vl.get("valid"),
                "vlm_beats_original": vo.get(f"{v}_win_rate"),
            }
    (E / "summary.json").write_text(json.dumps(S, indent=1))
    cols = [("sface_reid", "SFace re-ID"), ("sface_max", "SFace max"), ("arc_reid", "ArcFace re-ID (eval)"),
            ("arc_max", "ArcFace max"), ("seam_dE_flat", "seam dE flat"), ("seam_frac_flat", "seam px>6dE"),
            ("corr_mouth", "r mouth"), ("corr_blink", "r blink"), ("corr_yaw", "r yaw"), ("flicker_ratio", "flicker/orig"),
            ("frac_surrogate", "surrogate frac"), ("guard_blur", "guard blur fr"), ("prepare_s", "prep s"),
            ("vlm_win_vs_v2", "VLM win vs v2"), ("vlm_beats_original", "VLM picks it over original")]
    L = ["| clip | var | " + " | ".join(c for _, c in cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for k in CLIPS:
        for v, r in S.get(k, {}).items():
            L.append(f"| {k} | {v} | " + " | ".join("" if r[c] is None else str(r[c]) for c, _ in cols) + " |")
    (E / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
