#!/usr/bin/env bash
# Experiment face_g: v2 vs research-only swap variants, same tracks.
#   CUDA_VISIBLE_DEVICES=.. VARS="bp ap" CLIPS="clip1 clip4 clip5" STAGE=metrics|vlm|all experiments/face_g/eval_all.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
S=${SCRATCH:-/tmp/custodian_video_scratch}
MPPY=${MPPY:-$S/mpvenv/bin/python}; MPMODEL=${MPMODEL:-$S/face_landmarker.task}
export CUSTODIAN_VLM_PYTHON=${CUSTODIAN_VLM_PYTHON:-python3}
export HF_HUB_OFFLINE=1
E=experiments/face_g; EE=experiments/face_e; IN=../data/video_demo/inputs
declare -A NAME=([clip1]=clip1_interview_synthpii [clip4]=clip4_closeup_caption [clip5]=clip5_presenter_badge [clip6]=clip6_meeting_adults)
v2of() { if [ $1 = clip6 ]; then echo $E/out/clip6_v2; else echo experiments/face_f/out/${1}_v2; fi; }
VARS=${VARS:-bp}; STAGE=${STAGE:-all}
for k in ${CLIPS:-clip1 clip4 clip5}; do
  n=${NAME[$k]}; o=$E/out/$k; b=$(v2of $k)
  outs="v2=${b}.mp4"; for v in $VARS; do outs="$outs $v=${o}_$v.mp4"; done
  echo "== $k $(date)"
  if [ "$STAGE" != vlm ]; then
    CUDA_VISIBLE_DEVICES="" $PY $E/identity3.py --orig $IN/$n.mp4 --outs $outs --audit ${b}.audit.json \
        --json-out ${o}.id3.json > $E/logs/${k}_id3.log 2>&1
    $PY $EE/boundary_metrics.py --orig $IN/$n.mp4 --outs $outs --audit ${b}.audit.json \
        --json-out ${o}.boundary.json > $E/logs/${k}_boundary.log 2>&1
    $MPPY $E/motion_metrics2.py --orig $IN/$n.mp4 --outs $outs \
        --audit ${b}.audit.json --model $MPMODEL --json-out ${o}.motion.json > $E/logs/${k}_motion.log 2>&1
  fi
  if [ "$STAGE" != metrics ]; then
    for v in $VARS; do
      $PY $EE/vlm_pairwise.py --a v2=${b}.mp4 --b $v=${o}_$v.mp4 --audit ${b}.audit.json --n ${NPAIR:-40} --seed 0 \
          --json-out ${o}.vlm_$v.json --keep $E/media/vlm_pairs/${k}_$v > $E/logs/${k}_vlm_$v.log 2>&1
      $PY $EE/vlm_pairwise.py --a original=$IN/$n.mp4 --b $v=${o}_$v.mp4 --audit ${b}.audit.json --n ${NPAIR:-40} --seed 1 \
          --json-out ${o}.vlm_orig_$v.json > $E/logs/${k}_vlm_orig_$v.log 2>&1
    done
  fi
done
