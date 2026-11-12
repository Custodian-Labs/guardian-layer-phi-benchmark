#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
while pgrep -f "face_f/queue1.sh" > /dev/null; do sleep 10; done
G='"guard_thr": 0.24, "gen_thr": 0.15, "gen_tries": 16'
for k in clip1 clip4 clip5; do
  experiments/face_f/run_one.sh $k v3fl v3 '{"forehead_max": 0.2, "blend3": "laplacian"}'
  experiments/face_f/run_one.sh $k v3gl v3 "{\"blend3\": \"laplacian\", $G}"
  experiments/face_f/run_one.sh $k v3fgl v3 "{\"forehead_max\": 0.2, \"blend3\": \"laplacian\", $G}"
done
