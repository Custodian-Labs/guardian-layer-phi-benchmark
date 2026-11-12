#!/usr/bin/env bash
# CPU metrics for before/after stabilization: identity3 (+dump), motion2, stability.  args: clipkey vars...
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
S=${SCRATCH:-/tmp/custodian_video_scratch}
declare -A NAME=([clip1]=clip1_interview_synthpii [clip4]=clip4_closeup_caption [clip5]=clip5_presenter_badge)
k=$1; shift; n=${NAME[$k]}; E=experiments/face_g; IN=../data/video_demo/inputs; A=experiments/face_f/out/${k}_v2.audit.json
outs=""; for v in "$@"; do outs="$outs $v=$E/out/${k}_$v.mp4"; done
tag=$(echo "$@" | tr ' ' '_')
CUDA_VISIBLE_DEVICES="" $PY $E/identity3.py --orig $IN/$n.mp4 --outs $outs --audit $A --json-out $E/out/$k.id3_$tag.json --dump $E/out/$k.id3dump_$tag.json > $E/logs/${k}_id3_$tag.log 2>&1
$S/mpvenv/bin/python $E/motion_metrics2.py --orig $IN/$n.mp4 --outs $outs --audit $A --model $S/face_landmarker.task --json-out $E/out/$k.motion_$tag.json > $E/logs/${k}_motion_$tag.log 2>&1
$PY $E/stability.py --orig $IN/$n.mp4 --audit $E/out/${k}_$1.audit.json --outs $outs --json-out $E/out/$k.stability_$tag.json > $E/logs/${k}_stab_$tag.log 2>&1
echo done $k
