#!/usr/bin/env bash
# One command to (re)run the whole video Guardian benchmark for a pipeline version.
#
#   CUDA_VISIBLE_DEVICES=7 custodian_video/eval/run_all.sh v0.1 git:4dd3004
#   CUDA_VISIBLE_DEVICES=7 custodian_video/eval/run_all.sh v0.2 worktree v0.1
#
# args: TAG  [PIPELINE=worktree|git:<rev>]  [BASELINE_TAG]
# env : CUDA_VISIBLE_DEVICES (required; GPU 2 is reserved), WORKERS (default 4),
#       EXTRA_ARGS (pipeline CLI args appended for every arm, e.g. "--surrogate-min-face 32"),
#       VIDEO_EVAL_DATA (default <repo>/data/video_eval)
# Every stage is idempotent/resumable; re-running a tag only fills in what is missing.
set -euo pipefail
TAG=${1:?usage: run_all.sh TAG [PIPELINE] [BASELINE_TAG]}
PIPELINE=${2:-worktree}
BASE=${3:-}
WORKERS=${WORKERS:-4}
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to a free GPU (check nvidia-smi)}"
if [[ ",$CUDA_VISIBLE_DEVICES," == *",2,"* ]]; then echo "GPU 2 is reserved" >&2; exit 1; fi
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

HERE=$(cd "$(dirname "$0")" && pwd)
EVAL_PY=${EVAL_PY:-/data/qbao775/miniconda3/envs/custodian-video-eval/bin/python}
cd "$HERE"
t0=$(date +%s)
echo "== [$(date +%T)] download (idempotent)";  $EVAL_PY download.py
echo "== [$(date +%T)] prepare (idempotent)";   $EVAL_PY prepare.py
echo "== [$(date +%T)] run arms ($PIPELINE)";   $EVAL_PY run_arms.py --tag "$TAG" --pipeline "$PIPELINE" --workers "$WORKERS" --extra-args "${EXTRA_ARGS:-}"
echo "== [$(date +%T)] metrics";                HF_HUB_OFFLINE=1 $EVAL_PY metrics.py --tag "$TAG"
echo "== [$(date +%T)] analyze";                $EVAL_PY analyze.py --tag "$TAG" ${BASE:+--baseline-tag "$BASE"} > /dev/null
echo "== done in $(( ($(date +%s) - t0) / 60 )) min -> $HERE/results/$TAG/summary.md"
