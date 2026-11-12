#!/usr/bin/env bash
# usage: run_one.sh <clipkey> <variant-name> <backend v2|swap> '<kw json>'   (run from anywhere)
set -uo pipefail
cd "$(dirname "$0")/../.."
declare -A NAME=([clip1]=clip1_interview_synthpii [clip4]=clip4_closeup_caption [clip5]=clip5_presenter_badge [clip6]=clip6_meeting_adults)
declare -A CACHE=([clip1]=experiments/face_f/out/clip1_tracks.pkl [clip4]=experiments/face_f/out/clip4_tracks.pkl
                  [clip5]=experiments/face_f/out/clip5_tracks.pkl [clip6]=experiments/face_g/out/clip6_tracks.pkl)
k=$1; v=$2; be=$3; kw=${4:-"{}"}
export HF_HUB_OFFLINE=1
export CUSTODIAN_VLM_PYTHON=${CUSTODIAN_VLM_PYTHON:-python3}
${PY:-python3} experiments/face_g/run_face_g.py ../data/video_demo/inputs/${NAME[$k]}.mp4 \
  -o experiments/face_g/out/${k}_${v}.mp4 --cache ${CACHE[$k]} --realism v2 --small-extra 0.5 \
  --backend $be --kw "$kw" > experiments/face_g/logs/${k}_${v}.log 2>&1
echo "exit $?" >> experiments/face_g/logs/${k}_${v}.log
