#!/usr/bin/env bash
cd "$(dirname "$0")"
STAGE=metrics CLIPS="clip5 clip6" VARS="R" ./eval_all.sh
cd ../..; IN=../data/video_demo/inputs; F=experiments/face_f/out; G=experiments/face_g/out
CUDA_VISIBLE_DEVICES="" ${PY:-python3} experiments/face_g/impostor_floor3.py \
  --clips clip1=$IN/clip1_interview_synthpii.mp4,$F/clip1_v2.mp4,$G/clip1_R.mp4 clip4=$IN/clip4_closeup_caption.mp4,$F/clip4_v2.mp4,$G/clip4_R.mp4 \
          clip5=$IN/clip5_presenter_badge.mp4,$F/clip5_v2.mp4,$G/clip5_R.mp4 clip6=$IN/clip6_meeting_adults.mp4,$G/clip6_v2.mp4,$G/clip6_R.mp4 \
  --names v2 R --json-out $G/impostor_floor.json > experiments/face_g/logs/impostor_floor.log 2>&1
