#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
while pgrep -f "face_f/queue5.sh" > /dev/null; do sleep 10; done
S='"fill": true, "blend3": "laplacian", "v2_strength": 0.92, "control_scale": 0.6, "gen_thr": 0.15, "gen_tries": 16'
for k in clip1 clip4 clip5; do
  experiments/face_f/run_one.sh $k v3xs v3 "{$S}"
done
