#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
while pgrep -f "face_f/queue[12].sh" > /dev/null; do sleep 10; done
for k in clip4 clip1 clip5; do
  experiments/face_f/run_one.sh $k v3p v3 '{"forehead_max": 0.2, "gen_thr": 0.12}'
  experiments/face_f/run_one.sh $k v3pl v3 '{"forehead_max": 0.2, "gen_thr": 0.12, "blend3": "laplacian"}'
done
