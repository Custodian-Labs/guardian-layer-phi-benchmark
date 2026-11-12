#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
while pgrep -f "face_f/queue1.sh" > /dev/null; do sleep 10; done
for k in clip4 clip1 clip5; do
  experiments/face_f/run_one.sh $k v3f v3 '{"forehead_max": 0.2}'
  experiments/face_f/run_one.sh $k v3fl v3 '{"forehead_max": 0.2, "blend3": "laplacian"}'
done
