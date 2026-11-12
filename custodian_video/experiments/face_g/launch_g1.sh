#!/usr/bin/env bash
cd "$(dirname "$0")"
M='mk1={"push":0.35,"mouth_keep":1.0}'
G='mg={"push":0.35,"mouth_keep":1.0,"enhancer":"gfpgan_1.4","enhance_blend":0.35}'
CUDA_VISIBLE_DEVICES=4 setsid nohup ./chain.sh 'queue_a.sh bq=' "clip5 clip1" "$M" > logs/chain4.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 setsid nohup bash -c "./chain.sh 'queue_a.sh mk1=' clip6 '$M'; CLIPS='clip4 clip5' ./queue_a.sh '$G'" > logs/chain3.log 2>&1 &
