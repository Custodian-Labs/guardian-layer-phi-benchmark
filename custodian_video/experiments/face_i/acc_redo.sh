#!/usr/bin/env bash
# redo the accuracy stages after the _write_stage fix (stale clip-1 refs from the 18 s input)
cd "$(git rev-parse --show-toplevel)/custodian_video"
PY=${PY:-python3}
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 CUSTODIAN_VLM_PYTHON=${CUSTODIAN_VLM_PYTHON:-python3}
R=../data/video_demo; L=experiments/face_i/logs; ALL="blur box blur_fake_pii blur_guardian surrogate_research"
echo "[$(date +%T)] faces"; CUDA_VISIBLE_DEVICES="" $PY demo/web_accuracy.py faces --root $R --configs $ALL > $L/acc_faces2.log 2>&1; echo rc=$?
echo "[$(date +%T)] plates"; CUDA_VISIBLE_DEVICES="" $PY demo/web_accuracy.py plates --root $R --configs $ALL > $L/acc_plates2.log 2>&1; echo rc=$?
echo "[$(date +%T)] vlm"; CUDA_VISIBLE_DEVICES=4 $PY demo/web_accuracy.py vlm --root $R > $L/acc_vlm2.log 2>&1; echo rc=$?
echo "[$(date +%T)] text"; CUDA_VISIBLE_DEVICES="" $PY demo/web_accuracy.py text --root $R --configs $ALL > $L/acc_text2.log 2>&1; echo rc=$?
echo "[$(date +%T)] speech"; CUDA_VISIBLE_DEVICES=4 $PY demo/web_accuracy.py speech --root $R --configs $ALL > $L/acc_speech2.log 2>&1; echo rc=$?
echo "[$(date +%T)] assemble"; $PY demo/web_accuracy.py assemble --root $R > $L/acc_assemble2.log 2>&1; echo rc=$?
$PY demo/build_web_assets.py --root $R --web ../web/video > $L/assets3.log 2>&1; echo assets rc=$?
echo "[$(date +%T)] ACC REDO DONE"
