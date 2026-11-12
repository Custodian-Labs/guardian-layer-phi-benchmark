#!/usr/bin/env bash
# usage: run_one.sh <clipkey> <variant-name> <backend v2|v3> '<kw json>'   (run from custodian_video/)
set -uo pipefail
cd "$(dirname "$0")/../.."
declare -A NAME=([clip1]=clip1_interview_synthpii [clip4]=clip4_closeup_caption [clip5]=clip5_presenter_badge)
k=$1; v=$2; be=$3; kw=${4:-"{}"}
export HF_HUB_OFFLINE=1
export CUSTODIAN_VLM_PYTHON=${CUSTODIAN_VLM_PYTHON:-python3}
export CUSTODIAN_MP_PYTHON=${CUSTODIAN_MP_PYTHON:-python3}
${PY:-python3} experiments/face_f/run_face_f.py ../data/video_demo/inputs/${NAME[$k]}.mp4 \
  -o experiments/face_f/out/${k}_${v}.mp4 --cache experiments/face_f/out/${k}_tracks.pkl --realism v2 --small-extra 0.5 \
  --backend $be --kw "$kw" > experiments/face_f/logs/${k}_${v}.log 2>&1
echo "exit $?" >> experiments/face_f/logs/${k}_${v}.log
