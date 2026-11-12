#!/usr/bin/env bash
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=3 setsid nohup bash -c "while kill -0 1973913 2>/dev/null; do sleep 20; done; CLIPS=clip6 ./queue_a.sh 'mk1={\"push\":0.35,\"mouth_keep\":1.0}'; CLIPS='clip4 clip5' ./queue_a.sh 'mg={\"push\":0.35,\"mouth_keep\":1.0,\"enhancer\":\"gfpgan_1.4\",\"enhance_blend\":0.35}'" > logs/chain3.log 2>&1 &
