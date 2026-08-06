#!/bin/bash
cd /data/qbao775/custodianai
for corpus in redacted faker; do for b in asq_phi meddocan multiconer_v2 pii_masking_300k pii_dutch pii_french pii_german; do
  echo "[$(date +%H:%M:%S)] presidio $corpus $b"
  PYTHONPATH=. .venv/bin/python scripts/score_corpus.py --system presidio --corpus "$corpus" --benchmark "$b" >> results/c1c2/log_presidio_redo.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] OK presidio $corpus $b" || echo "[$(date +%H:%M:%S)] FAIL presidio $corpus $b"
done; done
echo "[$(date +%H:%M:%S)] PRESIDIO REDO DONE"
