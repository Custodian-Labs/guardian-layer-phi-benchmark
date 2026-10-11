#!/usr/bin/env python3
"""Gold-span re-projection check and typology-vs-outcome analysis (paper v2).

B1  gold-span re-projection quality on masked spans: how often the difflib-
    re-projected span on d' cuts inside a word (vs. the same check on d), and
    whether the exact-vs-overlap gap is concentrated in those spans
B2  typology vs. detection: join the 200-span surrogate audit to per-detector
    outcomes and report original / transformed recall by failure class
"""
import csv, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from analyze_equivalence import BENCH, load, found  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
AUDIT_BENCH = {"ASQ-PHI": "ASQ-PHI", "MEDDOCAN": "MEDDOCAN", "MultiCoNER": "MultiCoNER v2",
               "PII-en": "PII en", "PII-nl": "PII nl", "PII-fr": "PII fr", "PII-de": "PII de"}


def exact(g, preds):
    return any(p["start"] == g["start"] and p["end"] == g["end"] for p in preds)


def cuts_word(text, s, e):
    a = s > 0 and text[s - 1].isalnum() and text[s].isalnum()
    b = e < len(text) and text[e - 1].isalnum() and text[e].isalnum()
    return a or b


def texts(globs):
    t = {}
    for g in globs:
        import glob as G
        for fn in sorted(G.glob(os.path.join(ROOT, "results", g))):
            for line in open(fn):
                if line.strip():
                    r = json.loads(line)
                    t.setdefault(r["doc_id"], r.get("text", ""))
    return t


def main():
    agg = defaultdict(lambda: [0, 0, 0, 0, 0])  # clean/cut -> n, ov_o, ov_t, ex_o, ex_t
    span_cut = [0, 0, 0]  # n_masked_unique, cut_on_t, cut_on_o
    data = {}
    for name, (og, tg) in BENCH.items():
        do, dt = load(og), load(tg)
        To, Tt = texts(og), texts(tg)
        data[name] = (do, dt)
        for did, d0 in do.items():
            d1 = dt.get(did)
            if not d1 or len(d0["gold"]) != len(d1["gold"]):
                continue
            syss = set(d0["preds"]) & set(d1["preds"])
            for sg, tg_ in zip(d0["gold"], d1["gold"]):
                if sg.get("text") == tg_.get("text"):
                    continue
                ct = cuts_word(Tt.get(did, ""), tg_["start"], tg_["end"]) if Tt.get(did) else False
                co = cuts_word(To.get(did, ""), sg["start"], sg["end"]) if To.get(did) else False
                span_cut[0] += 1; span_cut[1] += ct; span_cut[2] += co
                k = "cut" if ct else "clean"
                for s in syss:
                    a = agg[k]; a[0] += 1
                    a[1] += found(sg, d0["preds"][s]); a[2] += found(tg_, d1["preds"][s])
                    a[3] += exact(sg, d0["preds"][s]); a[4] += exact(tg_, d1["preds"][s])
    print("B1  re-projected masked spans: %d unique; cut inside a word on d': %d (%.1f%%); on d: %d (%.1f%%)"
          % (span_cut[0], span_cut[1], 100 * span_cut[1] / span_cut[0], span_cut[2], 100 * span_cut[2] / span_cut[0]))
    print(f"{'subset':6s} {'outcomes':>8s} {'ov_o':>6s} {'ov_t':>6s} {'ex_o':>6s} {'ex_t':>6s} {'ex gap':>7s}")
    tot = [0] * 5
    for k in ("clean", "cut"):
        a = agg[k]; tot = [x + y for x, y in zip(tot, a)]
        print(f"{k:6s} {a[0]:8d} " + " ".join(f"{100*a[i]/a[0]:6.1f}" for i in range(1, 5))
              + f" {100*(a[3]-a[4])/a[0]:+7.1f}")
    print(f"{'all':6s} {tot[0]:8d} " + " ".join(f"{100*tot[i]/tot[0]:6.1f}" for i in range(1, 5))
          + f" {100*(tot[3]-tot[4])/tot[0]:+7.1f}")

    # ---- B2
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "annotation", "surrogate_quality_audit.csv"))))
    cls = defaultdict(lambda: [0, 0, 0, 0])  # spans_matched, outcomes, found_o, found_t
    miss = 0
    for r in rows:
        do, dt = data[AUDIT_BENCH[r["benchmark"]]]
        d0, d1 = do.get(r["doc_id"]), dt.get(r["doc_id"])
        hit = None
        if d0 and d1 and len(d0["gold"]) == len(d1["gold"]):
            for sg, tg_ in zip(d0["gold"], d1["gold"]):
                if sg.get("text") == r["original"] and tg_.get("text") == r["surrogate"]:
                    hit = (sg, tg_); break
        if not hit:
            miss += 1; continue
        syss = set(d0["preds"]) & set(d1["preds"])
        c = cls[r["failure"]]; c[0] += 1
        for s in syss:
            c[1] += 1; c[2] += found(hit[0], d0["preds"][s]); c[3] += found(hit[1], d1["preds"][s])
    print(f"\nB2  audit join: {200-miss}/200 spans matched")
    print(f"{'class':18s} {'spans':>5s} {'outc':>5s} {'rec_o':>6s} {'rec_t':>6s} {'change':>7s}")
    T = [0, 0, 0, 0]
    for k in ("none", "truncated_garbled", "x_masked", "salience_loss"):
        c = cls[k]; T = [x + y for x, y in zip(T, c)]
        if c[1]:
            print(f"{k:18s} {c[0]:5d} {c[1]:5d} {100*c[2]/c[1]:6.1f} {100*c[3]/c[1]:6.1f} {100*(c[3]-c[2])/c[1]:+7.1f}")
    d = [cls[k] for k in ("truncated_garbled", "x_masked", "salience_loss")]
    D = [sum(x[i] for x in d) for i in range(4)]
    print(f"{'any defect':18s} {D[0]:5d} {D[1]:5d} {100*D[2]/D[1]:6.1f} {100*D[3]/D[1]:6.1f} {100*(D[3]-D[2])/D[1]:+7.1f}")
    print(f"{'all':18s} {T[0]:5d} {T[1]:5d} {100*T[2]/T[1]:6.1f} {100*T[3]/T[1]:6.1f} {100*(T[3]-T[2])/T[1]:+7.1f}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- B3
def snap(text, s, e):
    """Extend a re-projected span outward to the nearest word boundaries."""
    while s > 0 and text[s - 1].isalnum() and text[s].isalnum():
        s -= 1
    while e < len(text) and text[e - 1].isalnum() and text[e].isalnum():
        e += 1
    return s, e


def snapped_load(globs, _orig_load=load):
    docs = _orig_load(globs)
    if any(g.startswith("t_") for g in globs):
        T = texts(globs)
        for did, d in docs.items():
            t = T.get(did, "")
            if not t:
                continue
            new = []
            for g in d["gold"]:
                s, e = snap(t, g["start"], g["end"])
                new.append({**g, "start": s, "end": e, "text": t[s:e]})
            d["gold"] = new
    return docs


def b3():
    """Re-run the headline + cluster bootstrap + exact gap with snapped spans."""
    import analyze_robustness as R
    import numpy as np
    R.load = snapped_load
    data = R.collect()
    print("\nB3  headline with word-boundary-snapped re-projection")
    R.r1_r2(data)
    n = ex_o = ex_t = 0
    for name, (og, tg) in BENCH.items():
        do, dt = load(og), snapped_load(tg)
        for did, d0 in do.items():
            d1 = dt.get(did)
            if not d1 or len(d0["gold"]) != len(d1["gold"]):
                continue
            syss = set(d0["preds"]) & set(d1["preds"])
            for sg, tg_ in zip(d0["gold"], d1["gold"]):
                # masked = text changed (snapping never makes an unmasked span differ)
                if sg.get("text") == tg_.get("text"):
                    continue
                for s in syss:
                    n += 1; ex_o += exact(sg, d0["preds"][s]); ex_t += exact(tg_, d1["preds"][s])
    print(f"exact-boundary recall on masked spans (snapped): {100*ex_o/n:.1f} -> {100*ex_t/n:.1f} "
          f"({100*(ex_o-ex_t)/n:+.1f} pts), n={n}")

    # how many audit 'truncated_garbled' surrogates were re-projection artefacts?
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "annotation", "surrogate_quality_audit.csv"))))
    art = tot = 0
    for r in rows:
        if r["failure"] != "truncated_garbled":
            continue
        tot += 1
        _, tg = BENCH[AUDIT_BENCH[r["benchmark"]]]
        dt0, dts = load(tg), snapped_load(tg)
        d0, d1 = dt0.get(r["doc_id"]), dts.get(r["doc_id"])
        for a, b in zip(d0["gold"], d1["gold"]):
            if a.get("text") == r["surrogate"]:
                art += a["text"] != b["text"]
                break
    print(f"audit 'truncated/garbled' surrogates that were re-projection artefacts: {art}/{tot}")


if __name__ == "__main__" and "--snap" in sys.argv:
    b3()
