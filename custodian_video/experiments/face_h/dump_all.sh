#!/usr/bin/env bash
cd "$(git rev-parse --show-toplevel)/custodian_video"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 HF_HUB_OFFLINE=1
for c in clip1_interview_synthpii clip2_street_faces clip3_traffic_plates clip4_closeup_caption clip5_presenter_badge clip6_meeting_adults; do
  ${PY:-python3} experiments/face_h/dump_fp_inputs.py ../data/video_demo/inputs/$c.mp4 experiments/face_h/out/$c.fpin.pkl
done
echo ALLDONE
