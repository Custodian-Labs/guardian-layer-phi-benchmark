"""Position-debiased variant of face_e/vlm_pairwise.py (same crops, prompt, model):
every pair is judged in BOTH orders; B wins a frame only if it is chosen in both
orders, A likewise, otherwise the frame is a tie.  Reports B's share of decisive
frames and of all frames (ties count half), plus the raw per-order agreement.
  python vlm_both_orders.py --a v2=a.mp4 --b R=b.mp4 --audit a.audit.json --n 40 --json-out x.json"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "face_e"))
import vlm_pairwise as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--audit", required=True); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--json-out")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    a = ap.parse_args()
    na, pa = a.a.split("=", 1); nb, pb = a.b.split("=", 1)
    py = os.environ.get("CUSTODIAN_VLM_PYTHON", sys.executable)
    with tempfile.TemporaryDirectory(prefix="vlm2_") as d:
        items = V.crops(pa, pb, a.audit, a.n, a.seed, d)
        both = []
        for it in items:
            pA, pB = (it["img2"], it["img1"]) if it["b_is"] == 1 else (it["img1"], it["img2"])
            both.append({"id": it["id"] + "_ab", "frame": it["frame"], "img1": pA, "img2": pB, "b_is": 2})
            both.append({"id": it["id"] + "_ba", "frame": it["frame"], "img1": pB, "img2": pA, "b_is": 1})
        fin, fout = os.path.join(d, "in.json"), os.path.join(d, "out.json")
        json.dump({"model": a.model, "prompt": V.PROMPT, "items": both}, open(fin, "w"))
        r = subprocess.run([py, V.__file__, "--worker", fin, fout], capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr[-2000:])
        ans = json.load(open(fout))
    def pick(it):
        s = (ans.get(it["id"]) or "").strip(); dg = next((c for c in s if c in "12"), None)
        return None if dg is None else ("B" if int(dg) == it["b_is"] else "A")
    W = {"B": 0, "A": 0, "tie": 0}; first = 0; per = []
    for i in range(0, len(both), 2):
        x, y = pick(both[i]), pick(both[i + 1])
        first += sum(1 for it, p in ((both[i], x), (both[i + 1], y)) if p and (p == "B") == (it["b_is"] == 1))
        w = x if x == y and x is not None else "tie"
        W[w] += 1; per.append((both[i]["frame"], w))
    n = len(per); dec = W["A"] + W["B"]
    res = {"A": na, "B": nb, "frames": n, f"{nb}_both_orders": W["B"], f"{na}_both_orders": W["A"], "ties": W["tie"],
           f"{nb}_share_decisive": round(W["B"] / max(dec, 1), 3), f"{nb}_score_ties_half": round((W["B"] + 0.5 * W["tie"]) / max(n, 1), 3),
           "chose_image1_rate": round(first / max(2 * n, 1), 3), "per_frame": per}
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame"}))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
