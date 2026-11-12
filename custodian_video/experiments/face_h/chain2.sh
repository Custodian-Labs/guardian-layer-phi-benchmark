#!/usr/bin/env bash
# after the first queue: re-run clip1/clip4 surrogate_research with the final (pairwise) quality ranking
cd "$(git rev-parse --show-toplevel)/custodian_video"
until grep -q "QUEUE DONE" experiments/face_h/logs/queue.log; do sleep 30; done
R=../data/video_demo
mkdir -p $R/outputs_face_h_trials/top3vote/clip1 $R/outputs_face_h_trials/top3vote/clip4
mv $R/outputs/clip1_interview_synthpii/surrogate_research.* $R/outputs_face_h_trials/top3vote/clip1/
mv $R/outputs/clip4_closeup_caption/surrogate_research.* $R/outputs_face_h_trials/top3vote/clip4/
rm -rf experiments/face_h/out/portraits/clip1_interview_synthpii experiments/face_h/out/portraits/clip4_closeup_caption
experiments/face_h/queue.sh clip1_interview_synthpii/surrogate_research clip4_closeup_caption/surrogate_research > experiments/face_h/logs/queue2.log 2>&1
