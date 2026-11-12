# face_i — research swap on the low-resolution meeting clip (clip 6), 2026-10-08

Question: why does `surrogate_research` blur every face on clip 6 (faces ~70-150 px, median 109 px)
although they are above the 64 px swap floor?

Findings (GPU 4, `restore_small_faces.patch` = the experimental code, NOT merged):

| run | change | alpha pasted | SFace p75 vs original (guard thr 0.25) | result |
|---|---|---|---|---|
| baseline | — | mean ~0.02, max ~0.15 | 1.000 (nothing pasted) | all blurred |
| 1 | GFPGAN-restored crop for parsing/occlusion only | still ~0.08 max 0.16 | 0.91-0.97 | all blurred |
| 2 | swap + segment the restored crop, downsample back, yaw gate 0.45, restore for faces < ~200 px | ~0.32, max 1.0 | 0.38-0.54 | all blurred |
| 3 | + replace the mouth too on these faces (mouth_keep 0) | ~0.32, max 1.0 | 0.30-0.54 | all blurred |

Causes: on blurry ~100 px faces BiSeNet finds 1-17 % of the crop as face (vs ~40 % after
restoration) and HyperSwap's own mask peaks at ~0.5 (1.0 on close-ups), so nothing was pasted.
Restoring first fixes the pasting, but even full-face swaps stay matchable by SFace
(p75 0.30-0.54 > the 0.25 small-face margin and mostly > the 0.363 match threshold).
Strongly turned faces ghost and must be blurred anyway.

Decision: keep fail-closed blur on clip 6; the page text now gives this reason instead of
"faces under 64 px". Clip 2 (median face 51 px) is below the 64 px floor.

## Clip 1 trimmed to 12 s (2026-10-09, user request)

`clip1_12s.sh`: the web-demo input `data/video_demo/inputs/clip1_interview_synthpii.mp4` is now the
first 12 s (was a symlink to the 18 s `demo/inputs` clip, which is unchanged); the second speaker,
the remaining curtain false positive and the spoken account number (12.5-15 s) are gone, so planted
speech PII on the page is 2/2. Surrogate: 295/300 frames swapped, last 5 (head tilted back) blurred;
re-ID 0 % on SFace / ArcFace / Glint. `acc_redo.sh`: rerun after fixing `web_accuracy._write_stage`,
which kept a clip's stale reference data (18 s refs) when merging with `--configs`.

Then 11 s (same day, user request; `SECS=11 experiments/face_i/clip1_trim.sh`, GPU 4 only): the
surrogate is swapped on all 275/275 frames (no blurred frames), re-ID 0 % on all three recognisers.
Clip 1 is the default clip for every visitor (remembered-clip key bumped to `vdemo4-clip`).
