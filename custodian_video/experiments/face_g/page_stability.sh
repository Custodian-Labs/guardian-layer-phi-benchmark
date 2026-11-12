#!/usr/bin/env bash
# stability of web-demo surrogate_research outputs (state from comparison with blur.mp4). arg: outputs dir name, tag
cd "$(dirname "$0")/../../.."
PY=${PY:-python3}; R=data/video_demo; D=$1; tag=$2
for c in clip1_interview_synthpii clip4_closeup_caption clip5_presenter_badge; do
  $PY custodian_video/experiments/face_g/stability.py --orig $R/inputs/$c.mp4 --audit $R/$D/$c/surrogate_research.audit.json \
     --outs $tag=$R/$D/$c/surrogate_research.mp4 --blur-ref $R/outputs/$c/blur.mp4 \
     --json-out custodian_video/experiments/face_g/out/$c.page_stability_$tag.json > /dev/null 2>&1
done
echo done $tag
