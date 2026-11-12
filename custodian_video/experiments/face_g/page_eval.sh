#!/usr/bin/env bash
# Re-ID / floor / motion numbers for the web-demo config surrogate_research (CPU only).
#   experiments/face_g/page_eval.sh   -> ../data/video_demo/surrogate_research_eval.json
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
S=${SCRATCH:-/tmp/custodian_video_scratch}
MPPY=${MPPY:-$S/mpvenv/bin/python}; MPMODEL=${MPMODEL:-$S/face_landmarker.task}
R=../data/video_demo; W=$R/accuracy_work/surrogate_research_eval; mkdir -p $W
CL="clip1_interview_synthpii clip2_street_faces clip4_closeup_caption clip5_presenter_badge clip6_meeting_adults"
for c in $CL; do
  o=$R/outputs/$c
  CUDA_VISIBLE_DEVICES="" $PY experiments/face_g/identity3.py --orig $R/inputs/$c.mp4 --outs surrogate_research=$o/surrogate_research.mp4 blur=$o/blur.mp4 \
     --audit $o/surrogate_research.audit.json --json-out $W/$c.id3.json > $W/$c.id3.log 2>&1
done
for c in clip2_street_faces clip6_meeting_adults; do
  CUDA_VISIBLE_DEVICES="" $PY experiments/face_g/inframe_floor.py --orig $R/inputs/$c.mp4 --outs surrogate_research=$R/outputs/$c/surrogate_research.mp4 \
     --json-out $W/$c.inframe.json > $W/$c.inframe.log 2>&1
done
args=""; for c in clip1_interview_synthpii clip4_closeup_caption clip5_presenter_badge; do args="$args $c=$R/inputs/$c.mp4,$R/outputs/$c/surrogate_research.mp4"; done
CUDA_VISIBLE_DEVICES="" $PY experiments/face_g/impostor_floor3.py --clips $args --names surrogate_research --json-out $W/cross_floor.json > $W/cross_floor.log 2>&1
for c in clip1_interview_synthpii clip4_closeup_caption clip5_presenter_badge; do
  $MPPY experiments/face_g/motion_metrics2.py --orig $R/inputs/$c.mp4 --outs surrogate_research=$R/outputs/$c/surrogate_research.mp4 \
     --audit $R/outputs/$c/surrogate_research.audit.json --model $MPMODEL --json-out $W/$c.motion.json > $W/$c.motion.log 2>&1
done
$PY experiments/face_g/page_eval_combine.py $W $R/surrogate_research_eval.json
