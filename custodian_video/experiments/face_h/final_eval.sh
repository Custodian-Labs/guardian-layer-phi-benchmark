#!/usr/bin/env bash
cd "$(git rev-parse --show-toplevel)/custodian_video"
PY=${PY:-python3}
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 CUSTODIAN_VLM_PYTHON=${CUSTODIAN_VLM_PYTHON:-python3}
L=experiments/face_h/logs
free_of() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g{print $2}'; }
gpu() { while :; do for c in 4 3; do [[ $(free_of $c) -ge 25600 ]] && { echo $c; return; }; done; sleep 60; done; }
( experiments/face_g/page_eval.sh; experiments/face_g/page_stability.sh outputs face_h ) > $L/page_eval_final.log 2>&1 &
CUDA_VISIBLE_DEVICES="" $PY experiments/face_e/privacy_report.py --root ../data/video_demo --step 2 --configs surrogate_research --json-out experiments/face_h/out/face_privacy_sr_final.json --md-out experiments/face_h/out/privacy_sr_final.md > $L/privacy_sr.log 2>&1
CUDA_VISIBLE_DEVICES="" $PY demo/web_accuracy.py faces --root ../data/video_demo --configs surrogate_research > $L/acc_faces_sr.log 2>&1; echo "faces rc=$?"
g=$(gpu); CUDA_VISIBLE_DEVICES=$g $PY demo/web_accuracy.py vlm --root ../data/video_demo > $L/acc_vlm2.log 2>&1; echo "vlm gpu$g rc=$?"
CUDA_VISIBLE_DEVICES="" $PY demo/web_accuracy.py text --root ../data/video_demo --configs surrogate_research > $L/acc_text_sr2.log 2>&1; echo "text rc=$?"
g=$(gpu); CUDA_VISIBLE_DEVICES=$g $PY demo/web_accuracy.py speech --root ../data/video_demo --configs surrogate_research > $L/acc_speech_sr.log 2>&1; echo "speech gpu$g rc=$?"
g=$(gpu); CUDA_VISIBLE_DEVICES=$g $PY demo/web_pii_checklist.py --root ../data/video_demo --json-out ../data/video_demo/checklist.json --configs surrogate_research > $L/checklist_sr.log 2>&1; echo "checklist gpu$g rc=$?"
wait
echo "[$(date +%T)] FINAL EVAL DONE"
