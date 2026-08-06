#!/bin/bash
cd /data/qbao775/custodianai
PY=/data/qbao775/miniconda3/envs/gemma4unified/bin/python
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. $PY scripts/score_corpus.py --system qwen3_5_9b --all >> results/c1c2/log_qwen3_5_9b.log 2>&1 &
Q=$!
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. $PY scripts/score_corpus.py --system gemma_4_31b --all >> results/c1c2/log_gemma_4_31b.log 2>&1 &
G=$!
wait $Q $G
echo "[$(date +%H:%M:%S)] LOCAL LLM BATCH DONE"
