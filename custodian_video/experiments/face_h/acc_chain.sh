#!/usr/bin/env bash
# accuracy chain after the face_h regeneration: GPU stages sequential on one GPU (4, else 3), >= 25 GB free
cd "$(git rev-parse --show-toplevel)/custodian_video"
PY=${PY:-python3}
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 CUSTODIAN_VLM_PYTHON=${CUSTODIAN_VLM_PYTHON:-python3}
L=experiments/face_h/logs
free_of() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g{print $2}'; }
gpu() { while :; do for c in 4 3; do [[ $(free_of $c) -ge 25600 ]] && { echo $c; return; }; done; sleep 60; done; }
while pgrep -f "demo/web_accuracy.py faces" >/dev/null; do sleep 15; done
echo "[$(date +%T)] faces stage finished"
g=$(gpu); echo "[$(date +%T)] vlm on gpu$g"; CUDA_VISIBLE_DEVICES=$g $PY demo/web_accuracy.py vlm --root ../data/video_demo > $L/acc_vlm.log 2>&1; echo "rc=$?"
CUDA_VISIBLE_DEVICES="" $PY demo/web_accuracy.py text --root ../data/video_demo --configs surrogate_research > $L/acc_text_sr.log 2>&1; echo "[$(date +%T)] text rc=$?"
g=$(gpu); echo "[$(date +%T)] speech on gpu$g"; CUDA_VISIBLE_DEVICES=$g $PY demo/web_accuracy.py speech --root ../data/video_demo --configs blur box blur_fake_pii blur_guardian surrogate_research > $L/acc_speech.log 2>&1; echo "rc=$?"
g=$(gpu); echo "[$(date +%T)] checklist on gpu$g"; CUDA_VISIBLE_DEVICES=$g $PY demo/web_pii_checklist.py --root ../data/video_demo --json-out ../data/video_demo/checklist.json --configs blur,box,blur_fake_pii,blur_guardian,surrogate_research > $L/checklist.log 2>&1; echo "rc=$?"
echo "[$(date +%T)] ACC CHAIN DONE"
