#!/usr/bin/env bash
# Web demo clip 1 -> first $SECS s (user requests 2026-10-09: 12 s, then 11 s): trim input, rerun the 5 page configs on
# GPUs 3/4 (one job each), then rebuild every page number that involves clip 1.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)/custodian_video"
PY=${PY:-python3}
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 CUSTODIAN_VLM_PYTHON=${CUSTODIAN_VLM_PYTHON:-python3}
R=../data/video_demo; C=clip1_interview_synthpii; L=experiments/face_i/logs; mkdir -p $L
FF=$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
free_of() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g{print $2}'; }
gpu() { while :; do for c in 4 3; do [[ $(free_of $c) -ge 25600 ]] && { echo $c; return; }; done; sleep 60; done; }
step() { echo "[$(date +%T)] $*"; }

# web-demo clip 1 = the first $SECS s of demo/inputs/$C.mp4 (12 s on 2026-10-09, then 11 s)
SECS=${SECS:-11}
cur=$($FF -i $R/inputs/$C.mp4 2>&1 | sed -n 's/.*Duration: \([0-9:.]*\).*/\1/p' | awk -F: '{printf "%d", $1*3600+$2*60+$3+0.5}')
if [[ -L $R/inputs/$C.mp4 || $cur != $SECS ]]; then
  step "trim input to $SECS s (was ${cur} s)"
  $FF -v error -y -i demo/inputs/$C.mp4 -t $SECS -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -c:a aac $R/inputs/_$C.mp4
  rm $R/inputs/$C.mp4 && mv $R/inputs/_$C.mp4 $R/inputs/$C.mp4
  mkdir -p $R/outputs_pre_clip1_trim/${cur}s && mv $R/outputs/$C $R/outputs_pre_clip1_trim/${cur}s/
fi
step "rerun page configs"
GPUS="${CHAIN_GPUS:-3 4}" PER_GPU=1 ONLY=$C CONFIGS="blur box blur_fake_pii blur_guardian surrogate_research" \
  CUSTODIAN_GUARDIAN_TOTAL_CAP=${CUSTODIAN_GUARDIAN_TOTAL_CAP:-233} demo/run_web_demo.sh
for k in blur box blur_fake_pii blur_guardian surrogate_research; do
  [[ -s $R/outputs/$C/$k.audit.json ]] || { step "MISSING $k -> stop"; exit 1; }
done
step "assets + overlays"
$PY demo/build_web_assets.py --root $R --web ../web/video --force > $L/assets.log 2>&1; echo "rc=$?"
$PY demo/build_web_overlays.py
step "face privacy (clip 1)"
CUDA_VISIBLE_DEVICES="" $PY experiments/face_e/privacy_report.py --root $R --step 2 --clips $C --configs blur box surrogate_research \
  --json-out experiments/face_i/out_privacy_clip1.json > $L/privacy.log 2>&1; echo "rc=$?"
$PY - <<'PY'
import json
p="../data/video_demo/face_privacy.json"; d=json.load(open(p))
d["clip1_interview_synthpii"]=json.load(open("experiments/face_i/out_privacy_clip1.json"))["clip1_interview_synthpii"]
open(p,"w").write(json.dumps(d,indent=1)); print("merged face_privacy")
PY
step "surrogate eval (re-ID / motion)"
experiments/face_g/page_eval.sh > $L/page_eval.log 2>&1; echo "rc=$?"
ALL="blur box blur_fake_pii blur_guardian surrogate_research"
step "accuracy: faces"; CUDA_VISIBLE_DEVICES="" $PY demo/web_accuracy.py faces --root $R --configs $ALL > $L/acc_faces.log 2>&1; echo "rc=$?"
g=$(gpu); step "accuracy: vlm gpu$g"; CUDA_VISIBLE_DEVICES=$g $PY demo/web_accuracy.py vlm --root $R > $L/acc_vlm.log 2>&1; echo "rc=$?"
step "accuracy: text"; CUDA_VISIBLE_DEVICES="" $PY demo/web_accuracy.py text --root $R --configs $ALL > $L/acc_text.log 2>&1; echo "rc=$?"
g=$(gpu); step "accuracy: speech gpu$g"; CUDA_VISIBLE_DEVICES=$g $PY demo/web_accuracy.py speech --root $R --configs $ALL > $L/acc_speech.log 2>&1; echo "rc=$?"
g=$(gpu); step "checklist gpu$g"; CUDA_VISIBLE_DEVICES=$g $PY demo/web_pii_checklist.py --root $R --json-out $R/checklist.json --configs original,${ALL// /,} > $L/checklist.log 2>&1; echo "rc=$?"
step "accuracy: assemble"; $PY demo/web_accuracy.py assemble --root $R > $L/acc_assemble.log 2>&1; echo "rc=$?"
$PY demo/build_web_assets.py --root $R --web ../web/video > $L/assets2.log 2>&1; echo "assets rc=$?"
step "CLIP1 TRIM DONE"
