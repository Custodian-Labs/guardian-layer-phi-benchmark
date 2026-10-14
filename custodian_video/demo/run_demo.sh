#!/usr/bin/env bash
# Reproduce the v0.2 demo (inputs -> outputs -> eval -> keyframes).
# Needs: the custodian-video conda env, Ollama with qwen2.5:7b on localhost:11434,
# EgoBlur weights in ~/.cache/custodian_video/egoblur_gen1 (see README), and an
# interpreter with transformers>=4.57 for the Qwen3-VL scene gate ($CUSTODIAN_VLM_PYTHON).
#   OUT=demo/outputs/v02 BACKEND=local demo/run_demo.sh
#   OUT=demo/outputs/v02_guardian BACKEND=guardian CHANNELS=text,audio demo/run_demo.sh
# BACKEND=guardian|both sends OCR lines / transcripts to the Custodian Guardian Layer
# (hosted endpoint; research on synthetic/public demo data only).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/data/qbao775/miniconda3/envs/custodian-video/bin/python}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export CUSTODIAN_VLM_PYTHON=${CUSTODIAN_VLM_PYTHON:-/data/qbao775/miniconda3/envs/gemma4-rl/bin/python}
VOICE=${VOICE:-$HOME/.cache/custodian_video/voices/en_US-amy-medium.onnx}
OUT=${OUT:-demo/outputs/v02}
BACKEND=${BACKEND:-local}
CHANNELS=${CHANNELS:-faces,plates,text,audio}
EXTRA=${EXTRA:-}
GCACHE=${GCACHE:-$HOME/.cache/custodian_video/guardian_cache_demo.json}
mkdir -p "$OUT"
for c in clip1_interview_synthpii clip2_street_faces clip3_traffic_plates; do
  for m in redact transform; do
    $PY -m custodian_video demo/inputs/$c.mp4 -o $OUT/${c%%_*}_$m.mp4 --mode $m --tts-voice "$VOICE" \
        --channels "$CHANNELS" --pii-backend "$BACKEND" --guardian-cache "$GCACHE" $EXTRA
  done
done
$PY demo/evaluate.py --orig demo/inputs/clip1_interview_synthpii.mp4 \
    --outs redact=$OUT/clip1_redact.mp4 transform=$OUT/clip1_transform.mp4 \
    --gt demo/inputs/ground_truth.json --step 5 --json-out $OUT/clip1_eval.json
if [[ "$CHANNELS" == *faces* ]]; then
  for c in clip2_street_faces clip3_traffic_plates; do
    k=${c%%_*}
    $PY demo/evaluate.py --orig demo/inputs/$c.mp4 --outs redact=$OUT/${k}_redact.mp4 \
        transform=$OUT/${k}_transform.mp4 --step 5 --min-face 16 --json-out $OUT/${k}_eval.json
  done
  $PY demo/keyframes.py --orig demo/inputs/clip1_interview_synthpii.mp4 \
      --outs redact=$OUT/clip1_redact.mp4 transform=$OUT/clip1_transform.mp4 \
      --times 5 16.5 --dst demo/keyframes/v02_clip1_full --width 640
fi
