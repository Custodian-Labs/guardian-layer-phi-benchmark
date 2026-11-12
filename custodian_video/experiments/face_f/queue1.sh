#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
while pgrep -f "run_face_f.py.*clip4_v2" > /dev/null; do sleep 10; done
for k in clip1 clip5; do
  experiments/face_f/run_one.sh $k v3 v3 '{}'
  experiments/face_f/run_one.sh $k v2 v2 '{}'
done
