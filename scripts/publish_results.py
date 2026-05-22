"""Bundle benchmark results into the static dashboard's data directory.

Reads `results/*.summary.json` and the corresponding `results/*.jsonl`
detail files, then writes:

  web/data/index.json                 — manifest of available runs
  web/data/<run_id>.summary.json      — aggregate per-system metrics
  web/data/<run_id>.samples.json      — small set of representative docs

Usage
-----
# Publish everything; per-doc samples ONLY for benchmarks marked synthetic
python scripts/publish_results.py

# Force-include samples for a DUA-protected benchmark (DO NOT DO THIS for
# real EHR text — only when you have explicit redaction guarantees)
python scripts/publish_results.py --publish-samples mimic_iv_note

# Limit to a single run
python scripts/publish_results.py --run-id asq_phi_1716000000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
WEB_DATA = ROOT / "web" / "data"

# Benchmarks safe to publish per-doc samples of (synthetic / public license).
SAMPLE_SAFE = {"asq_phi", "meddocan", "pii_masking_300k"}
SAMPLE_LIMIT = 12  # docs per run in the drilldown panel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", help="Specific run id (filename stem) to publish")
    p.add_argument("--publish-samples", nargs="*", default=[],
                   help="Additional benchmark names whose per-doc samples may be bundled.")
    p.add_argument("--max-text", type=int, default=2000,
                   help="Truncate doc texts beyond this many characters in samples.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not RESULTS.exists():
        print("no results/ directory yet", file=sys.stderr)
        return 1

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    sample_allow = SAMPLE_SAFE | set(args.publish_samples)

    runs = []
    for summary_path in sorted(RESULTS.glob("*.summary.json")):
        stem = summary_path.stem.removesuffix(".summary")
        if args.run_id and stem != args.run_id:
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            print(f"[skip] bad summary {summary_path}", file=sys.stderr)
            continue
        if not summary:
            continue

        # Detect benchmark from the first row OR the filename pattern.
        benchmark = summary[0].get("benchmark") or _infer_benchmark(stem)
        mode = summary[0].get("mode", "type")
        n_docs = max((row.get("n_docs", 0) for row in summary), default=0)

        bundle = {
            "id": stem,
            "benchmark": benchmark,
            "mode": mode,
            "n_docs": n_docs,
            "timestamp": _stamp_to_iso(stem, summary_path),
            "summary_path": f"{stem}.summary.json",
        }

        # Copy summary as-is (small, safe).
        (WEB_DATA / bundle["summary_path"]).write_text(json.dumps(summary, indent=2))

        # Samples (gated by license + size).
        details_path = summary_path.with_suffix("").with_suffix(".jsonl")
        # ^^ stems already include `.summary` so this becomes `<stem>.jsonl` of the
        #    full detail file from the runner.
        if not details_path.exists():
            details_path = RESULTS / f"{stem}.jsonl"
        if benchmark in sample_allow and details_path.exists():
            samples = _pick_samples(details_path, args.max_text)
            samples_name = f"{stem}.samples.json"
            (WEB_DATA / samples_name).write_text(json.dumps(samples, ensure_ascii=False))
            bundle["samples_path"] = samples_name
        else:
            bundle["samples_path"] = None

        runs.append(bundle)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "runs": runs,
    }
    (WEB_DATA / "index.json").write_text(json.dumps(manifest, indent=2))
    print(f"published {len(runs)} run(s) to {WEB_DATA}")
    return 0


def _stamp_to_iso(stem: str, fallback_path: Path | None = None) -> str:
    m = re.search(r"(\d{10,})$", stem)
    if m:
        return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(int(m.group(1))))
    if fallback_path and fallback_path.exists():
        return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(fallback_path.stat().st_mtime))
    return ""


def _infer_benchmark(stem: str) -> str:
    # asq_phi_1716000000 -> asq_phi
    return re.sub(r"_\d{10,}$", "", stem)


def _pick_samples(details_path: Path, max_text: int) -> list[dict]:
    """Sample docs where systems disagree the most, capped to SAMPLE_LIMIT."""
    rows = []
    with details_path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return []

    def disagreement(row):
        f1s = [
            p["score"]["tp"] / max(1, p["score"]["tp"] + 0.5 * (p["score"]["fp"] + p["score"]["fn"]))
            for p in row.get("predictions", {}).values()
        ]
        if len(f1s) < 2:
            return 0.0
        return max(f1s) - min(f1s)

    rows.sort(key=disagreement, reverse=True)
    picked = rows[:SAMPLE_LIMIT]

    # The runner does not currently persist doc.text or gold_spans in the
    # detail JSONL. If your runner is patched to include them, pass them
    # through here. Otherwise we leave text="" and the dashboard renders
    # only the per-system spans without context.
    for r in picked:
        r.setdefault("text", "")
        r.setdefault("gold_spans", [])
        if r["text"] and len(r["text"]) > max_text:
            r["text"] = r["text"][:max_text] + "…"
    return picked


if __name__ == "__main__":
    raise SystemExit(main())
