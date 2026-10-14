"""Stage 2: run the de-identification pipeline (through its CLI only) on every clip x arm.

  python run_arms.py --tag v0.1 --pipeline git:4dd3004 [--arms ...] [--workers 4]
  python run_arms.py --tag v0.2 --pipeline worktree

The pipeline code is snapshotted ONCE per tag into DATA/pipeline_snapshots/<tag>/ (from a git
revision or from the current working tree) and run from there via `python -m custodian_video`
(the public CLI), so concurrent edits to the package cannot leak into a tag half-way through.

Outputs go to DATA/outputs/<tag>/<arm>/<clip>.mp4 (+ .audit.json, .log). Existing outputs
are skipped, so the command is resumable. A per-run record (wall time, exit code) is appended to
results/<tag>/runs.jsonl, and results/<tag>/run_info.json records the snapshot (source, commit,
sha256 of the snapshotted files, reported __version__).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import ARMS, DATA, REPO, RESULTS, arm_video, load_manifest

DEFAULT_PY = "/data/qbao775/miniconda3/envs/custodian-video/bin/python"
PKG = "custodian_video/custodian_video"
EVAL_DATASETS = ("k400", "mot17", "vox2")      # vox2_gallery clips stay original (attacker gallery)


def snapshot_pipeline(tag: str, source: str) -> tuple[Path, dict]:
    """Copy the pipeline package into DATA/pipeline_snapshots/<tag>/ once, so a tag is pinned to
    one code version even while the working tree is being edited. source = 'git:<rev>' or
    'worktree'. An existing snapshot is reused (and its recorded source kept)."""
    root = DATA / "pipeline_snapshots" / tag
    meta_p = root / "snapshot.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta["source"] != source:
            print(f"NOTE: tag {tag} is pinned to pipeline source {meta['source']!r}; ignoring {source!r}")
        return root, meta

    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout

    tmp = root.with_name(root.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if source.startswith("git:"):
        rev = source[4:]
        tar = git("archive", "--format=tar", rev, PKG)
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(tmp / "x")
        shutil.move(str(tmp / "x" / PKG), str(tmp / "custodian_video"))
        shutil.rmtree(tmp / "x")
        commit = git("rev-parse", rev).decode().strip()
    elif source == "worktree":
        shutil.copytree(REPO / PKG, tmp / "custodian_video", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        commit = git("rev-parse", "HEAD").decode().strip() + "+worktree"
    else:
        raise SystemExit(f"bad --pipeline {source!r}")
    h = hashlib.sha256()
    for f in sorted((tmp / "custodian_video").rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(tmp)).encode()); h.update(f.read_bytes())
    meta = dict(source=source, commit=commit, files_sha256=h.hexdigest()[:16],
                snapshotted=dt.datetime.now().isoformat(timespec="seconds"))
    (tmp / "snapshot.json").write_text(json.dumps(meta, indent=1))
    if root.exists():
        shutil.rmtree(root)
    tmp.rename(root)
    return root, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", default="reencode,redact_blur,redact_box,transform")
    ap.add_argument("--datasets", default=",".join(EVAL_DATASETS))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--python", default=os.environ.get("CUSTODIAN_VIDEO_PYTHON", DEFAULT_PY),
                    help="interpreter of the env the pipeline runs in")
    ap.add_argument("--pipeline", default="worktree",
                    help="'git:<rev>' or 'worktree': code snapshotted (once per tag) and run via "
                         "`python -m custodian_video` (the public CLI)")
    ap.add_argument("--extra-args", default="", help="extra CLI args appended for every arm")
    ap.add_argument("--limit", type=int, help="first N clips per dataset (smoke tests)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--cpus-per-worker", type=int, default=8,
                    help="each worker is pinned (taskset) to this many cores; ORT/OpenCV otherwise "
                         "spawn one thread per core of the shared host")
    ap.add_argument("--cpu-base", type=int, default=int(os.environ.get("EVAL_CPU_BASE", 160)),
                    help="first core of the pinned range")
    a = ap.parse_args()

    res_dir = RESULTS / a.tag
    res_dir.mkdir(parents=True, exist_ok=True)
    arms = [x for x in a.arms.split(",") if x and x != "original"]
    snap, meta = snapshot_pipeline(a.tag, a.pipeline)
    info = dict(tag=a.tag, started=dt.datetime.now().isoformat(timespec="seconds"), python=a.python,
                pipeline=meta, snapshot_dir=str(snap),
                arms={k: ["--channels", "faces", *ARMS[k], *a.extra_args.split()] for k in arms},
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    penv = dict(os.environ, PYTHONPATH=str(snap))
    info["custodian_video_version"] = subprocess.run(
        [a.python, "-c", "import custodian_video;print(custodian_video.__version__, custodian_video.__file__)"],
        capture_output=True, text=True, env=penv, cwd=str(snap)).stdout.strip()
    if not info["custodian_video_version"].split(" ")[-1].startswith(str(snap)):
        raise SystemExit(f"snapshot import failed: {info['custodian_video_version']}")
    prev = res_dir / "run_info.json"
    if prev.exists():
        old = json.loads(prev.read_text())
        info["history"] = old.get("history", []) + [{k: old.get(k) for k in ("started", "pipeline")}]
    prev.write_text(json.dumps(info, indent=1))

    ds = set(a.datasets.split(","))
    clips = [c for c in load_manifest() if c["dataset"] in ds]
    if a.limit:
        per = {}
        clips = [c for c in clips if per.setdefault(c["dataset"], []).append(c) or len(per[c["dataset"]]) <= a.limit]
    jobs = [(c, arm) for arm in arms for c in clips]
    todo = [(c, arm) for c, arm in jobs
            if not (arm_video(c, arm, a.tag).exists() and arm_video(c, arm, a.tag).with_suffix(".audit.json").exists())]
    print(f"{len(jobs)} clip x arm jobs, {len(todo)} to run, {a.workers} workers", flush=True)

    lock = threading.Lock()
    done = [0]
    nt = str(max(1, a.cpus_per_worker))
    env = dict(os.environ, PYTHONPATH=str(snap), HF_HUB_OFFLINE="1", OMP_NUM_THREADS=nt, MKL_NUM_THREADS=nt,
               OPENBLAS_NUM_THREADS=nt, OPENCV_FOR_THREADS_NUM=nt, TOKENIZERS_PARALLELISM="false")

    import queue
    slots = queue.Queue()
    ncpu = os.cpu_count() or 1
    for w in range(a.workers):
        lo = (a.cpu_base + w * a.cpus_per_worker) % ncpu
        slots.put(f"{lo}-{min(ncpu - 1, lo + a.cpus_per_worker - 1)}")
    has_taskset = shutil.which("taskset") is not None

    def run(job):
        c, arm = job
        src, out = arm_video(c, "original", a.tag), arm_video(c, arm, a.tag)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp.mp4")
        audit_tmp = tmp.with_suffix(".audit.json")
        cmd = [a.python, "-m", "custodian_video", str(src), "-o", str(tmp), "--audit", str(audit_tmp), "--channels", "faces",
               *ARMS[arm], *a.extra_args.split()]
        slot = slots.get()
        if has_taskset and a.cpus_per_worker > 0:
            cmd = ["taskset", "-c", slot, *cmd]
        t0 = time.time()
        with open(out.with_suffix(".log"), "w") as lf:
            try:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(snap),
                                    timeout=a.timeout).returncode
            except subprocess.TimeoutExpired:
                rc = -9
        slots.put(slot)
        wall = time.time() - t0
        if rc == 0 and tmp.exists():
            tmp.rename(out)
            audit_tmp.rename(out.with_suffix(".audit.json"))
        rec = dict(clip=c["id"], dataset=c["dataset"], arm=arm, rc=rc, wall_s=round(wall, 2),
                   frames=c["n"], w=c["w"], h=c["h"])
        with lock:
            done[0] += 1
            with open(res_dir / "runs.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            if rc != 0 or done[0] % 10 == 0:
                print(f"  [{done[0]}/{len(todo)}] {arm} {c['id']} rc={rc} {wall:.1f}s", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, todo))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
