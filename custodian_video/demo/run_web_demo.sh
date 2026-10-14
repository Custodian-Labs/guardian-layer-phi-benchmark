#!/usr/bin/env bash
# Precompute every clip x configuration for the web video demo (web/video.html).
#   GPUS="3 4" PER_GPU=2 demo/run_web_demo.sh            # all jobs, skips finished outputs
#   ONLY="clip4_closeup_caption" demo/run_web_demo.sh     # one clip
# Inputs : $ROOT/inputs/*.mp4 (demo/make_web_demo_inputs.py + symlinks to demo/inputs/clip1-3)
# Outputs: $ROOT/outputs/<clip>/<config>.mp4 (+ .audit.json, .log)   -- kept out of git
# Then   : demo/build_web_assets.py transcodes them into web/video/ and writes manifest.json;
#          demo/web_accuracy.py {faces,plates,vlm,text,speech,assemble} writes web/video/accuracy.json.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/data/qbao775/miniconda3/envs/custodian-video/bin/python}
ROOT=${ROOT:-$(cd .. && pwd)/data/video_demo}
GPUS=${GPUS:-"3 4"}
PER_GPU=${PER_GPU:-2}
ONLY=${ONLY:-}
FP_FILTER=${FP_FILTER:-}   # e.g. FP_FILTER=none to override the --face-fp-filter default
CONFIGS=${CONFIGS:-"blur box blur_fake_pii blur_guardian"}  # page configs; surrogate_*/gen2/guardian: opt-in, not on the page
export HF_HUB_OFFLINE=1
export CUSTODIAN_VLM_PYTHON=${CUSTODIAN_VLM_PYTHON:-/data/qbao775/miniconda3/envs/gemma4-rl/bin/python}
# Guardian Layer: per-run cap below, plus a global hard cap (persistent counter in
# ~/.cache/custodian_video/guardian_usage.json) passed in by the caller.
export CUSTODIAN_GUARDIAN_TOTAL_CAP=${CUSTODIAN_GUARDIAN_TOTAL_CAP:-233}
VOICE=${VOICE:-$HOME/.cache/custodian_video/voices/en_US-amy-medium.onnx}
GCACHE=${GCACHE:-$HOME/.cache/custodian_video/guardian_cache_demo.json}
GEN2=$HOME/.cache/custodian_video/egoblur_gen2

run_job() {  # clip config gpu
  local clip=$1 cfg=$2 gpu=$3
  local out=$ROOT/outputs/$clip/$cfg.mp4
  [[ -s $out && -s ${out%.mp4}.audit.json ]] && return 0
  mkdir -p "$ROOT/outputs/$clip"
  local args=(--tts-voice "$VOICE" --channels faces,plates,text,audio)
  [[ -n $FP_FILTER ]] && args+=(--face-fp-filter "$FP_FILTER")
  local envs=(CUDA_VISIBLE_DEVICES=$gpu)
  case $cfg in
    blur)                  args+=(--mode redact) ;;
    blur_fake_pii)         args+=(--mode transform --face-mode redact) ;;   # faces blurred; screen text + speech get fake values
    blur_guardian)         args+=(--mode transform --face-mode redact --pii-backend guardian-replace
                                  --guardian-cache "$GCACHE" --guardian-max-calls 40) ;;  # values from Guardian Layer
    surrogate_research)    args+=(--mode transform --face-backend research-swap --pii-backend guardian-replace
                                  --guardian-cache "$GCACHE" --guardian-max-calls 40) ;;  # research-only face swap (experiments/face_g)
    box)                   args+=(--mode redact --face-style box) ;;
    surrogate_static)      args+=(--mode transform); envs+=(CUSTODIAN_FACE_MOTION=static) ;;
    surrogate_animated)    args+=(--mode transform) ;;
    surrogate_recommended) args+=(--mode transform --surrogate-min-face 32) ;;
    gen2)                  args+=(--mode transform --surrogate-min-face 32 --face-threshold 0.1)
                           envs+=(EGOBLUR_DIR=$GEN2) ;;
    guardian)              args+=(--mode transform --surrogate-min-face 32 --pii-backend both
                                  --guardian-cache "$GCACHE" --guardian-max-calls 30) ;;
    *) echo "unknown config $cfg"; return 1 ;;
  esac
  echo "[$(date +%T)] gpu$gpu start $clip/$cfg"
  env "${envs[@]}" $PY -m custodian_video "$ROOT/inputs/$clip.mp4" -o "$out" "${args[@]}" \
      > "${out%.mp4}.log" 2>&1
  echo "[$(date +%T)] gpu$gpu done  $clip/$cfg rc=$?"
}

jobs=()
for clip in $(ls "$ROOT/inputs" | sed -n 's/\.mp4$//p'); do
  [[ -n $ONLY && " $ONLY " != *" $clip "* ]] && continue
  for cfg in $CONFIGS; do jobs+=("$clip $cfg"); done
done
slots=()
for g in $GPUS; do for ((k = 0; k < PER_GPU; k++)); do slots+=("$g"); done; done
n=${#slots[@]}
for ((w = 0; w < n; w++)); do
  (
    for ((j = w; j < ${#jobs[@]}; j += n)); do run_job ${jobs[$j]} "${slots[$w]}"; done
  ) &
done
wait
echo "[$(date +%T)] all done"
