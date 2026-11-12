#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
while pgrep -f "run_face_f.py" > /dev/null; do sleep 10; done
X='"fill": true, "blend3": "laplacian"'
G='"guard_thr": 0.24, "gen_thr": 0.15, "gen_tries": 16'
experiments/face_f/run_one.sh clip1 v3xc v3 "{$X, \"source_clean\": true}"
experiments/face_f/run_one.sh clip5 v3xg v3 "{$X, $G}"
experiments/face_f/run_one.sh clip4 v3xc v3 "{$X, \"source_clean\": true}"
experiments/face_f/run_one.sh clip5 v3xc v3 "{$X, \"source_clean\": true}"
for k in clip1 clip4 clip5; do
  experiments/face_f/run_one.sh $k v3xs v3 "{$X, \"source_clean\": true, \"v2_strength\": 0.92, \"control_scale\": 0.6, \"gen_thr\": 0.15, \"gen_tries\": 16}"
done
