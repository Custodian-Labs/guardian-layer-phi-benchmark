#!/usr/bin/env bash
# Experiment face_e: surrogate realism v1 (before) vs v2 (after) on the demo clips.
#   CUDA_VISIBLE_DEVICES=3 MPPY=.../mpvenv/bin/python MPMODEL=.../face_landmarker.task \
#       experiments/face_e/eval_all.sh                   (run from custodian_video/)
# Needs out/<clip>_{v1,v2}.mp4 from run_face_e.py (--realism v1 / v2, same --cache).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
MPPY=${MPPY:?python with mediapipe}; MPMODEL=${MPMODEL:?face_landmarker.task}
export CUSTODIAN_VLM_PYTHON=${CUSTODIAN_VLM_PYTHON:-python3}
export HF_HUB_OFFLINE=1
E=experiments/face_e; IN=../data/video_demo/inputs
declare -A NAME=([clip1]=clip1_interview_synthpii [clip4]=clip4_closeup_caption
                 [clip5]=clip5_presenter_badge [clip6]=clip6_group_indoor)
declare -A TIMES=([clip1]="2 6 10 16" [clip4]="1 4 8 11" [clip5]="1 4 8 11" [clip6]="1 4 7 9")
for k in ${CLIPS:-clip1 clip4 clip5 clip6}; do
  n=${NAME[$k]}; o=$E/out/$k
  echo "== $k"
  $PY $E/identity2.py --orig $IN/$n.mp4 --outs v1=${o}_v1.mp4 v2=${o}_v2.mp4 blur=../data/video_demo/outputs/$n/blur.mp4 --audit ${o}_v2.audit.json \
      --json-out ${o}.id2.json > /dev/null 2>&1
  $PY $E/boundary_metrics.py --orig $IN/$n.mp4 --outs v1=${o}_v1.mp4 v2=${o}_v2.mp4 --audit ${o}_v2.audit.json \
      --json-out ${o}.boundary.json > /dev/null 2>&1
  $MPPY experiments/face_d/motion_metrics.py --orig $IN/$n.mp4 --outs v1=${o}_v1.mp4 v2=${o}_v2.mp4 \
      --audit ${o}_v2.audit.json --model $MPMODEL --json-out ${o}.motion.json > /dev/null 2>&1
  $PY $E/vlm_pairwise.py --a v1=${o}_v1.mp4 --b v2=${o}_v2.mp4 --audit ${o}_v2.audit.json --n 20 --seed 0 \
      --json-out ${o}.vlm.json --keep $E/media/vlm_pairs/$k > /dev/null 2>&1
  $PY $E/crops.py --orig $IN/$n.mp4 --outs old_v1=${o}_v1.mp4 new_v2=${o}_v2.mp4 --audit ${o}_v2.audit.json \
      --times ${TIMES[$k]} --dst $E/media/${k}_keyframes_original_old_new.jpg --tile 360
done
$PY $E/summarize.py
