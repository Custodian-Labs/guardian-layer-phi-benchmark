#!/usr/bin/env bash
cd "$(dirname "$0")"
STAGE=metrics CLIPS="clip4" VARS="bq mk1 mk6 iq" ./eval_all.sh
STAGE=metrics CLIPS="clip5" VARS="bq mk1 iq" ./eval_all.sh
