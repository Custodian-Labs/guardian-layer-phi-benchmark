#!/usr/bin/env bash
# Experiment face_f: current v2 vs v3 variants (parsing-aware paste) on clips 1/4/5, same tracks.
#   CUDA_VISIBLE_DEVICES=.. MPPY=.../mpvenv/bin/python MPMODEL=.../face_landmarker.task VARS="v3 v3f" experiments/face_f/eval_all.sh
# STAGE=metrics (CPU: identity, seam, motion) | vlm (GPU, Qwen3-VL-8B) | all
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
export CUSTODIAN_VLM_PYTHON=${CUSTODIAN_VLM_PYTHON:-python3}
export HF_HUB_OFFLINE=1
E=experiments/face_f; EE=experiments/face_e; IN=../data/video_demo/inputs
declare -A NAME=([clip1]=clip1_interview_synthpii [clip4]=clip4_closeup_caption [clip5]=clip5_presenter_badge)
VARS=${VARS:-v3}; STAGE=${STAGE:-all}
for k in ${CLIPS:-clip1 clip4 clip5}; do
  n=${NAME[$k]}; o=$E/out/$k
  outs="v2=${o}_v2.mp4"; for v in $VARS; do outs="$outs $v=${o}_$v.mp4"; done
  echo "== $k $(date)"
  if [ "$STAGE" != vlm ]; then
    CUDA_VISIBLE_DEVICES="" $PY $EE/identity2.py --orig $IN/$n.mp4 --outs $outs --audit ${o}_v2.audit.json \
        --json-out ${o}.id2.json > $E/logs/${k}_id2.log 2>&1
    $PY $EE/boundary_metrics.py --orig $IN/$n.mp4 --outs $outs --audit ${o}_v2.audit.json \
        --json-out ${o}.boundary.json > $E/logs/${k}_boundary.log 2>&1
    ${MPPY:?} $EE/../face_d/motion_metrics.py --orig $IN/$n.mp4 --outs $outs \
        --audit ${o}_v2.audit.json --model ${MPMODEL:?} --json-out ${o}.motion.json > $E/logs/${k}_motion.log 2>&1
  fi
  if [ "$STAGE" != metrics ]; then
    for v in $VARS; do
      $PY $EE/vlm_pairwise.py --a v2=${o}_v2.mp4 --b $v=${o}_$v.mp4 --audit ${o}_v2.audit.json --n ${NPAIR:-40} --seed 0 \
          --json-out ${o}.vlm_$v.json --keep $E/media/vlm_pairs/${k}_$v > $E/logs/${k}_vlm_$v.log 2>&1
    done
    for v in v2 $VARS; do   # how often is the edit judged MORE natural than the untouched original?
      $PY $EE/vlm_pairwise.py --a original=$IN/$n.mp4 --b $v=${o}_$v.mp4 --audit ${o}_v2.audit.json --n ${NPAIR:-40} --seed 1 \
          --json-out ${o}.vlm_orig_$v.json > $E/logs/${k}_vlm_orig_$v.log 2>&1
    done
  fi
done
