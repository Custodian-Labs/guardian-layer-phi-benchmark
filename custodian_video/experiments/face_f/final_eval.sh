#!/usr/bin/env bash
# after generation: metrics for clips 4/5, then VLM (GPU) for all clips, sequentially
cd "$(dirname "$0")/../.."
S=${SCRATCH:-/tmp/custodian_video_scratch}
export MPPY=$S/mpvenv/bin/python MPMODEL=$S/face_landmarker.task
while pgrep -f "face_f/queue7.sh" > /dev/null; do sleep 15; done
STAGE=metrics CLIPS="clip4 clip5" VARS="v3 v3x v3xg v3xc v3xs" experiments/face_f/eval_all.sh
while pgrep -f "STAGE=metrics|identity2.py|boundary_metrics.py|motion_metrics.py" > /dev/null; do sleep 15; done
STAGE=vlm CLIPS="clip1 clip4 clip5" VARS="v3 v3xc v3xs" experiments/face_f/eval_all.sh
echo done
