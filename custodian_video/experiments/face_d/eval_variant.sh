#!/usr/bin/env bash
# usage: eval_variant.sh NAME [run_face_d.py args...]   (run from custodian_video/)
set -euo pipefail
NAME=$1; shift
PY=${PY:-python3}
MPPY=${MPPY:?set MPPY to a python with mediapipe}
MPMODEL=${MPMODEL:?face_landmarker.task}
E=experiments/face_d; IN=demo/inputs/clip1_interview_synthpii.mp4
export HF_HUB_OFFLINE=1
$PY $E/run_face_d.py $IN -o $E/out/clip1_$NAME.mp4 --cache $E/out/clip1_tracks.pkl "$@" > $E/logs/clip1_$NAME.log 2>&1
$MPPY $E/motion_metrics.py --orig $IN --outs $NAME=$E/out/clip1_$NAME.mp4 --audit $E/out/clip1_$NAME.audit.json \
   --model $MPMODEL --json-out $E/out/clip1_$NAME.motion.json --series-out $E/out/clip1_$NAME.series.json > /dev/null 2>&1
$PY $E/identity_metrics.py --orig $IN --outs $NAME=$E/out/clip1_$NAME.mp4 --audit $E/out/clip1_$NAME.audit.json \
   --json-out $E/out/clip1_$NAME.identity.json > /dev/null 2>&1
echo "== $NAME"; python3 - "$E/out/clip1_$NAME" "$NAME" <<'PY'
import json,sys
b,n=sys.argv[1:]; m=json.load(open(b+'.motion.json'))[n]; i=json.load(open(b+'.identity.json'))[n]; t=json.load(open(b+'.timing.json'))
print({k:v for k,v in m.items() if k!='per_track'}); print({k:(v['corr_mouth_open'],v['mae_mouth_open'],v['std_mouth_open_orig'],v['std_mouth_open_out']) for k,v in m['per_track'].items()})
print({k:v for k,v in i.items() if k!='reid_frames'}, i['reid_frames'][:6]); print({k:t[k] for k in ('face_apply_fps','render_fps_incl_decode_encode','backend_stats')})
PY
