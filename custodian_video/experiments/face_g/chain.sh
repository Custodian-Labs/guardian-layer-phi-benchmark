#!/usr/bin/env bash
# wait until no OTHER process matches the pattern, then run a queue: chain.sh "<pattern>" "<clips>" specs...
cd "$(dirname "$0")"
while pgrep -u $USER -f "$1" | grep -vx "$$" | grep -q .; do sleep 20; done
shift; CLIPS="$1"; shift; CLIPS="$CLIPS" ./queue_a.sh "$@"
