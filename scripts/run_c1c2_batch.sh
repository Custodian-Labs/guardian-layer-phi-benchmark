#!/bin/bash
cd /data/qbao775/custodianai
BENCHES="asq_phi meddocan multiconer_v2 pii_masking_300k pii_dutch pii_french pii_german"
for sys in obi presidio openai; do
  for corpus in redacted faker; do
    for b in $BENCHES; do
      echo "[$(date +%H:%M:%S)] START $sys $corpus $b"
      PYTHONPATH=. .venv/bin/python scripts/score_corpus.py --system "$sys" --corpus "$corpus" --benchmark "$b" >> "results/c1c2/log_${sys}.log" 2>&1 \
        && echo "[$(date +%H:%M:%S)] OK $sys $corpus $b" || echo "[$(date +%H:%M:%S)] FAIL $sys $corpus $b"
    done
  done
done
echo "[$(date +%H:%M:%S)] BATCH COMPLETE"
