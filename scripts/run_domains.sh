#!/bin/bash
# Reproduce the non-medical domain study end to end (see docs/multidomain_results.md).
#   bash scripts/run_domains.sh                 # data + C1/C2 + all detectors + analysis
#   bash scripts/run_domains.sh custodian       # + Custodian transform arm (needs a valid CUSTODIAN_SDK_API_KEY)
# GPUs are shared: set GPU_QWEN / GPU_LLAMA / GPU_GEMMA to devices with >= 24 GB free.
set -e
cd "$(dirname "$0")/.."
PY_CPU=${PY_CPU:-python}     # presidio, spaCy en_core_web_lg, faker, custodian_labs
PY_GPU=${PY_GPU:-python}  # transformers>=5.13 (Qwen3.5 / Gemma 4)
export PYTHONPATH=.
mkdir -p results/domains/logs
CORPORA="original faker redacted"

if [ "$1" = "custodian" ]; then
  # Custodian Guardian Layer transform arm: transform, then score every detector on it.
  for b in finance_gretel legal_tab enterprise_gretel; do
    $PY_CPU scripts/transform_corpus.py --benchmark $b
  done
  CORPORA="custodian"
else
  $PY_CPU scripts/build_domain_corpora.py
  $PY_CPU scripts/build_domain_c1c2.py
fi

CUDA_VISIBLE_DEVICES="" $PY_CPU scripts/run_domain_eval.py --system presidio --corpora $CORPORA > results/domains/logs/presidio_$1.log 2>&1
CUDA_VISIBLE_DEVICES="" $PY_CPU scripts/run_domain_eval.py --system obi      --corpora $CORPORA > results/domains/logs/obi_$1.log 2>&1
CUDA_VISIBLE_DEVICES=${GPU_QWEN:-1}  $PY_GPU scripts/run_domain_eval.py --system qwen3_5_9b  --batch-size 8 --corpora $CORPORA > results/domains/logs/qwen3_5_9b_$1.log 2>&1 &
CUDA_VISIBLE_DEVICES=${GPU_LLAMA:-2} $PY_GPU scripts/run_domain_eval.py --system llama3_1_8b --batch-size 8 --corpora $CORPORA > results/domains/logs/llama3_1_8b_$1.log 2>&1 &
CUDA_VISIBLE_DEVICES=${GPU_GEMMA:-3} $PY_GPU scripts/run_domain_eval.py --system gemma_e4b   --batch-size 8 --corpora $CORPORA > results/domains/logs/gemma_e4b_$1.log 2>&1 &
wait
$PY_CPU scripts/analyze_domains.py
