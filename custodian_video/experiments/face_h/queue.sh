#!/usr/bin/env bash
# Sequential regeneration queue: one job at a time on GPU 4 (or 3), >= 25 GB free required.
cd "$(git rev-parse --show-toplevel)/custodian_video"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUSTODIAN_GUARDIAN_TOTAL_CAP=92          # 62 used so far + at most 30 new calls
export CUSTODIAN_RESEARCH_DUMP_DIR=custodian_video/experiments/face_h/out/portraits
free_of() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g{print $2}'; }
for job in "$@"; do
  clip=${job%%/*}; cfg=${job##*/}
  while :; do
    g=""
    for c in 4 3; do [[ $(free_of $c) -ge 25600 ]] && { g=$c; break; }; done
    [[ -n $g ]] && break
    echo "[$(date +%T)] waiting for a GPU with >= 25 GB free"; sleep 60
  done
  echo "[$(date +%T)] gpu$g ($(free_of $g) MiB free) $clip/$cfg"
  GPUS=$g PER_GPU=1 ONLY=$clip CONFIGS=$cfg bash demo/run_web_demo.sh
done
echo "[$(date +%T)] QUEUE DONE"
