#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
while pgrep -f "run_face_f.py" > /dev/null; do sleep 10; done
G='"guard_thr": 0.24, "gen_thr": 0.15, "gen_tries": 16'
for k in clip1 clip4 clip5; do
  experiments/face_f/run_one.sh $k v3x v3 '{"fill": true, "blend3": "laplacian"}'
  experiments/face_f/run_one.sh $k v3xg v3 "{\"fill\": true, \"blend3\": \"laplacian\", $G}"
done
for k in clip4 clip5; do
  experiments/face_f/run_one.sh $k v3gl v3 "{\"blend3\": \"laplacian\", $G}"
done
