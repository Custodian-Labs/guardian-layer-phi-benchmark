#!/usr/bin/env bash
# both-order VLM judge for v2 vs VAR on CLIPS (GPU)
cd "$(dirname "$0")/../.."
export CUSTODIAN_VLM_PYTHON=${CUSTODIAN_VLM_PYTHON:-python3} HF_HUB_OFFLINE=1
PY=${PY:-python3}; E=experiments/face_g; IN=../data/video_demo/inputs
declare -A NAME=([clip1]=clip1_interview_synthpii [clip4]=clip4_closeup_caption [clip5]=clip5_presenter_badge)
for k in $CLIPS; do for v in $VARS; do
  b=experiments/face_f/out/${k}_v2
  $PY $E/vlm_both_orders.py --a v2=$b.mp4 --b $v=$E/out/${k}_$v.mp4 --audit $b.audit.json --n 40 --seed 0 --json-out $E/out/$k.vlm2_$v.json > $E/logs/${k}_vlm2_$v.log 2>&1
done; done
