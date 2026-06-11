"""Stage 1 of the transformed-data experiment.

Push every benchmark document through the Custodian Guardian Layer
(`masking_type="transform"`, top-1 output) and cache the result, remapping
gold spans onto the surrogate text.

Why remap: transform substitutes PHI with plausible surrogates of *different
length* (Anna -> Maria), so original character offsets drift. We align
original vs transformed text with difflib and move every gold span to its
new location; the span's gold label is kept, and its text is re-sliced from
the transformed document. The downstream question is "can a detector still
find the (surrogate) PHI in a structure-preserving transformed document?",
so the gold label set must stay identical.

Source of documents: the merged results JSONLs from the original runs
(results/<bench>_250.jsonl), which carry doc_id + text + gold_spans for the
exact 250-doc subsets already graded. This guarantees both experiments use
identical documents.

Output: data/transformed/<benchmark>.jsonl with rows
  {doc_id, text, gold_spans:[{start,end,label,text}],
   meta:{orig_len, new_len, n_sensitive, changed, sdk_error}}

Usage:
  python scripts/transform_corpus.py                  # all 7 benchmarks
  python scripts/transform_corpus.py --benchmark asq_phi --limit 3   # smoke
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

SOURCES = {
    "asq_phi": "results/asq_phi_250.jsonl",
    "meddocan": "results/meddocan_250.jsonl",
    "pii_masking_300k": "results/pii_masking_300k_250.jsonl",
    "pii_masking_300k_dutch": "results/pii_dutch_250.jsonl",
    "pii_masking_300k_french": "results/pii_french_250.jsonl",
    "pii_masking_300k_german": "results/pii_german_250.jsonl",
    "multiconer_v2": "results/multiconer_v2_250.jsonl",
}

OUT_DIR = ROOT / "data" / "transformed"


def remap_spans(orig: str, new: str, spans: list[dict]) -> list[dict]:
    """Move gold spans from `orig` onto `new` via difflib char alignment."""
    if orig == new:
        return [dict(s) for s in spans]
    ops = difflib.SequenceMatcher(None, orig, new, autojunk=False).get_opcodes()

    def map_start(p: int) -> int:
        for tag, i1, i2, j1, j2 in ops:
            if i1 <= p < i2:
                if tag == "equal":
                    return j1 + (p - i1)
                return j1  # start of replacement block
        return len(new)

    def map_end(p: int) -> int:
        # p is exclusive end; find the op containing p-1, map past it.
        for tag, i1, i2, j1, j2 in ops:
            if i1 <= p - 1 < i2:
                if tag == "equal":
                    return j1 + (p - i1)
                return j2  # end of replacement block
        return len(new)

    out = []
    for s in spans:
        ns, ne = map_start(s["start"]), map_end(s["end"])
        ns = max(0, min(ns, len(new)))
        ne = max(ns, min(ne, len(new)))
        if ne == ns:  # span vanished (deletion); keep a zero-len marker out
            continue
        out.append({
            "start": ns,
            "end": ne,
            "label": s["label"],
            "text": new[ns:ne],
        })
    return out


def transform_doc(guardian, text: str, max_retries: int = 4):
    """Returns (transformed_text, n_sensitive, error_str|None)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            r = guardian.deidentify_text_outputs(
                text, masking_type="transform", pii_entities=["ALL"],
            )
            outs = r.outputs or []
            if not outs:
                return text, 0, None  # nothing sensitive found
            top1 = outs[0].text
            n_sens = len((r.meta or {}).get("sensitive_words", []))
            return top1, n_sens, None
        except Exception as e:  # rate limits / transient 5xx
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2.0 * (attempt + 1))
    return text, 0, last_err


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=list(SOURCES), default=None,
                   help="Single benchmark; default all")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip doc_ids already present in the output file")
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    from custodian_labs import GuardianLayer
    guardian = GuardianLayer()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    benches = [args.benchmark] if args.benchmark else list(SOURCES)

    for bench in benches:
        src = ROOT / SOURCES[bench]
        out_path = OUT_DIR / f"{bench}.jsonl"
        done_ids = set()
        if args.resume and out_path.exists():
            with out_path.open() as f:
                for line in f:
                    try:
                        done_ids.add(json.loads(line)["doc_id"])
                    except json.JSONDecodeError:
                        pass

        rows = [json.loads(l) for l in src.open()]
        if args.limit:
            rows = rows[: args.limit]
        todo = [r for r in rows if r["doc_id"] not in done_ids]
        print(f"[{bench}] {len(todo)}/{len(rows)} docs to transform "
              f"({len(done_ids)} cached)", flush=True)

        n_changed = n_err = 0
        t0 = time.time()
        with out_path.open("a") as out_f:
            for i, r in enumerate(todo):
                text = r["text"]
                new_text, n_sens, err = transform_doc(guardian, text)
                spans = remap_spans(text, new_text, r.get("gold_spans", []))
                changed = new_text != text
                n_changed += changed
                n_err += bool(err)
                out_f.write(json.dumps({
                    "doc_id": r["doc_id"],
                    "text": new_text,
                    "gold_spans": spans,
                    "meta": {
                        "orig_len": len(text), "new_len": len(new_text),
                        "n_sensitive": n_sens, "changed": changed,
                        "sdk_error": err,
                    },
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                if (i + 1) % 25 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    print(f"[{bench}] {i+1}/{len(todo)} "
                          f"({rate:.2f} docs/s, changed={n_changed}, err={n_err})",
                          flush=True)
        print(f"[{bench}] DONE changed={n_changed}/{len(todo)} errors={n_err}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
