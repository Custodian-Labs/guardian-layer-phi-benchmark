#!/usr/bin/env bash
cd "$(dirname "$0")"
for k in $CLIPS; do
  for spec in "$@"; do v=${spec%%=*}; kw=${spec#*=}; ./run_one.sh $k $v swap "$kw"; done
done
