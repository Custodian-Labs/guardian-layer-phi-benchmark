# Video Guardian benchmark: results for pipeline tag `v0.2-gen2`

Generated 2026-10-03T13:20:06 by `custodian_video/eval/analyze.py` (metric version m2). Pipeline snapshot: source `git:b612418`, commit `b612418a41`, files sha256 `15cdf8bde8544918`, reported version `0.2.0`. Only the face channel is enabled (`--channels faces`).

Clips: Kinetics-400 val **400** (action, pose), MOT17 **21** (person detection/count), VoxCeleb2 test **60** probes vs a 118-identity gallery (identification). Arms: original, `reencode` (null arm: same pipeline + encoder, nothing masked = codec-noise floor), `redact_blur` (mosaic+blur), `redact_box` (solid ellipse), `transform` (SD-1.5 surrogate face per track; blur fallback for faces <64 px or score <0.6).

ArcFace-R50 threshold at FAR=1e-3 (calibrated on 1,765,296 impostor pairs from the gallery): **0.217**; SFace uses OpenCV's 0.363.


## 1. Privacy (lower is better)

Per reference face (YuNet on the original frame, >=24 px), pooled as the mean of per-clip rates with 95% clip-bootstrap CIs.

- **Re-ID (same location)**: ArcFace cosine between the original face and the output pixels at the same landmarks >= FAR-1e-3 threshold. The attacker knows exactly where the face is, so this is a worst case.
- **Leak**: a face detector still finds a face there *and* it matches the original identity.
- **Face present**: a face detector still finds *a* face there. For privacy this alone is harmless; it is the utility goal of `transform`.
- **Impostor %**: how often the output face matches a *different* person's original face (from another clip of the same dataset) at the same threshold, i.e. the re-ID rate expected with no identity signal left. Re-ID should be compared with this floor, which is high for small, low-detail faces.
- **Rank-1 of 100**: closed-set identification. Is the true original the best match among itself plus 99 distractor originals? This is threshold-free; chance is 1%.
- **Mask coverage**: the audit log has a face mask containing the face centre (detection recall).

| Arm | Subset | clips / faces | Re-ID ArcFace % | Impostor % | Rank-1 of 100 % | Re-ID SFace % | Leak % | Face present % | Mask coverage % | mean cos |
|---|---|---|---|---|---|---|---|---|---|---|
| original | all faces | 479 / 10943 | 100.0 [100.0, 100.0] | 1.1 [1.0, 1.3] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | – | 1.000 |
| original | faces >=64px | 403 / 5439 | 100.0 [100.0, 100.0] | 0.9 [0.8, 1.1] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | – | 1.000 |
| reencode | all faces | 479 / 10943 | 100.0 [100.0, 100.0] | 1.2 [1.0, 1.3] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.6 [99.4, 99.7] | 99.8 [99.6, 99.9] | 0.0 [0.0, 0.0] | 0.949 |
| reencode | faces >=64px | 403 / 5439 | 100.0 [100.0, 100.0] | 1.0 [0.8, 1.1] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.5 [98.9, 99.9] | 99.9 [99.9, 100.0] | 0.0 [0.0, 0.0] | 0.961 |
| redact_blur | all faces | 479 / 10943 | 15.6 [13.7, 17.4] | 9.1 [8.7, 9.4] | 6.7 [5.6, 7.8] | 6.9 [5.9, 7.8] | 2.5 [1.9, 3.1] | 7.5 [6.4, 8.6] | 98.5 [98.1, 98.9] | 0.127 |
| redact_blur | faces >=64px | 403 / 5439 | 13.6 [11.5, 15.9] | 8.8 [8.4, 9.2] | 7.0 [5.5, 8.7] | 6.7 [5.2, 8.3] | 2.3 [1.4, 3.4] | 6.3 [4.8, 7.9] | 97.8 [96.8, 98.8] | 0.124 |
| redact_box | all faces | 479 / 10943 | 10.8 [9.4, 12.3] | 7.5 [7.1, 7.8] | 7.3 [5.9, 8.7] | 4.2 [3.5, 4.9] | 1.3 [1.0, 1.7] | 1.6 [1.3, 2.0] | 98.5 [98.1, 98.9] | 0.098 |
| redact_box | faces >=64px | 403 / 5439 | 10.8 [8.8, 12.9] | 7.0 [6.7, 7.4] | 8.9 [7.0, 10.8] | 3.7 [2.5, 5.0] | 1.9 [1.1, 3.0] | 2.3 [1.4, 3.4] | 97.8 [96.7, 98.8] | 0.101 |
| transform | all faces | 479 / 10943 | 13.2 [11.5, 15.0] | 5.0 [4.6, 5.4] | 7.5 [6.5, 8.6] | 8.0 [7.0, 9.1] | 3.5 [2.8, 4.3] | 64.2 [60.8, 67.4] | 98.5 [98.1, 98.9] | 0.112 |
| transform | faces >=64px | 403 / 5439 | 11.9 [9.7, 14.2] | 3.3 [3.0, 3.6] | 8.7 [7.1, 10.5] | 8.7 [7.0, 10.5] | 4.4 [3.1, 5.8] | 80.8 [77.9, 83.7] | 97.8 [96.7, 98.8] | 0.108 |

By dataset (all faces): re-ID ArcFace % / impostor % / rank-1 of 100 % / leak % / face present %

| Arm | Kinetics-400 | MOT17 | VoxCeleb2 |
|---|---|---|---|
| original | 100.0 / 1.0 / 100.0 / 100.0 / 100.0 | 100.0 / 5.9 / 100.0 / 100.0 / 100.0 | 100.0 / 0.1 / 100.0 / 100.0 / 100.0 |
| reencode | 100.0 / 1.1 / 100.0 / 99.6 / 99.8 | 100.0 / 6.3 / 100.0 / 98.8 / 99.0 | 100.0 / 0.1 / 100.0 / 100.0 / 100.0 |
| redact_blur | 15.5 / 9.8 / 4.6 / 2.4 / 7.0 | 42.2 / 20.5 / 11.0 / 11.2 / 17.3 | 7.2 / 0.4 / 19.2 / 0.2 / 7.4 |
| redact_box | 10.0 / 8.0 / 3.7 / 1.5 / 1.6 | 28.5 / 20.0 / 3.3 / 0.9 / 0.9 | 11.1 / 0.1 / 32.3 / 0.0 / 1.7 |
| transform | 13.6 / 5.2 / 7.3 / 3.7 / 62.4 | 42.1 / 16.8 / 11.0 / 11.0 / 33.0 | 1.5 / 0.1 / 7.8 / 0.2 / 85.5 |

**Cross-video identification (VoxCeleb2).** Each probe clip is identified against a gallery built from a *different* YouTube video of each of 118 people (chance = 0.85%). Rank-1 % with 95% CI. The clip-level score uses the mean embedding over 16 frames.

| Arm | Attacker | probes | rank-1 per clip % | rank-1 per frame % |
|---|---|---|---|---|
| original | same-location attacker | 60 | 100.0 [100.0, 100.0] | 99.5 [98.8, 100.0] |
| original | detector attacker | 60 | 100.0 [100.0, 100.0] | 99.5 [98.8, 100.0] |
| reencode | same-location attacker | 60 | 100.0 [100.0, 100.0] | 99.6 [99.1, 100.0] |
| reencode | detector attacker | 60 | 100.0 [100.0, 100.0] | 99.5 [98.6, 100.0] |
| redact_blur | same-location attacker | 60 | 10.0 [3.3, 18.3] | 10.5 [5.7, 16.0] |
| redact_blur | detector attacker | 60 | 1.7 [0.0, 5.0] | 1.2 [0.0, 3.6] |
| redact_box | same-location attacker | 60 | 16.7 [8.3, 26.7] | 10.9 [5.8, 16.8] |
| redact_box | detector attacker | 60 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| transform | same-location attacker | 60 | 1.7 [0.0, 5.0] | 1.8 [0.2, 4.3] |
| transform | detector attacker | 60 | 5.0 [0.0, 11.7] | 2.5 [0.3, 5.6] |

## 2. Utility: original vs each arm (paired per clip)

Δ = arm − original (points). 95% paired-bootstrap CI. TOST equivalence margin ±2 points (α=0.05): **EQ** = equivalent; ✗ = not shown equivalent.

| Metric | n | Original | Arm | Value | Δ [95% CI] | TOST p | Equiv. ±2 |
|---|---|---|---|---|---|---|---|
| Action top-1, VideoMAE-B (%) | 400 | 72.8 | reencode | 73.8 | 1.00 [-0.25, 2.50] | 0.078 | ✗ |
| Action top-1, VideoMAE-B (%) | 400 | 72.8 | redact_blur | 61.2 | -11.50 [-15.25, -7.75] | 1.000 | ✗ |
| Action top-1, VideoMAE-B (%) | 400 | 72.8 | redact_box | 63.5 | -9.25 [-13.00, -5.50] | 1.000 | ✗ |
| Action top-1, VideoMAE-B (%) | 400 | 72.8 | transform | 65.0 | -7.75 [-11.00, -4.75] | 1.000 | ✗ |
| Action top-5, VideoMAE-B (%) | 400 | 91.5 | reencode | 91.5 | 0.00 [-1.00, 1.00] | 0.000 | **EQ** |
| Action top-5, VideoMAE-B (%) | 400 | 91.5 | redact_blur | 83.2 | -8.25 [-11.00, -5.50] | 1.000 | ✗ |
| Action top-5, VideoMAE-B (%) | 400 | 91.5 | redact_box | 82.2 | -9.25 [-12.50, -6.25] | 1.000 | ✗ |
| Action top-5, VideoMAE-B (%) | 400 | 91.5 | transform | 82.2 | -9.25 [-12.25, -6.50] | 1.000 | ✗ |
| Action top-1, X-CLIP B/32 (%) | 400 | 75.2 | reencode | 76.0 | 0.75 [-0.25, 2.00] | 0.013 | **EQ** |
| Action top-1, X-CLIP B/32 (%) | 400 | 75.2 | redact_blur | 64.5 | -10.75 [-14.50, -7.25] | 1.000 | ✗ |
| Action top-1, X-CLIP B/32 (%) | 400 | 75.2 | redact_box | 65.2 | -10.00 [-13.50, -6.50] | 1.000 | ✗ |
| Action top-1, X-CLIP B/32 (%) | 400 | 75.2 | transform | 65.0 | -10.25 [-13.75, -7.00] | 1.000 | ✗ |
| Action top-5, X-CLIP B/32 (%) | 400 | 91.8 | reencode | 91.2 | -0.50 [-1.50, 0.50] | 0.001 | **EQ** |
| Action top-5, X-CLIP B/32 (%) | 400 | 91.8 | redact_blur | 83.0 | -8.75 [-11.75, -5.75] | 1.000 | ✗ |
| Action top-5, X-CLIP B/32 (%) | 400 | 91.8 | redact_box | 85.0 | -6.75 [-9.50, -4.25] | 1.000 | ✗ |
| Action top-5, X-CLIP B/32 (%) | 400 | 91.8 | transform | 83.8 | -8.00 [-11.00, -5.00] | 1.000 | ✗ |
| VideoMAE prediction = original's (%) | 400 | 100.0 | reencode | 97.2 | -2.75 [-4.50, -1.25] | 0.820 | ✗ |
| VideoMAE prediction = original's (%) | 400 | 100.0 | redact_blur | 75.5 | -24.50 [-28.75, -20.25] | 1.000 | ✗ |
| VideoMAE prediction = original's (%) | 400 | 100.0 | redact_box | 77.2 | -22.75 [-27.00, -18.75] | 1.000 | ✗ |
| VideoMAE prediction = original's (%) | 400 | 100.0 | transform | 81.2 | -18.75 [-22.50, -15.00] | 1.000 | ✗ |
| Pose OKS vs original, all 17 kpts (x100) | 400 | 100.0 | reencode | 84.3 | -15.70 [-16.59, -14.84] | 1.000 | ✗ |
| Pose OKS vs original, all 17 kpts (x100) | 400 | 100.0 | redact_blur | 59.0 | -41.00 [-42.79, -39.24] | 1.000 | ✗ |
| Pose OKS vs original, all 17 kpts (x100) | 400 | 100.0 | redact_box | 54.4 | -45.58 [-47.34, -43.75] | 1.000 | ✗ |
| Pose OKS vs original, all 17 kpts (x100) | 400 | 100.0 | transform | 65.1 | -34.91 [-36.52, -33.28] | 1.000 | ✗ |
| Pose OKS vs original, 5 head kpts (x100) | 400 | 100.0 | reencode | 85.7 | -14.29 [-15.27, -13.33] | 1.000 | ✗ |
| Pose OKS vs original, 5 head kpts (x100) | 400 | 100.0 | redact_blur | 41.9 | -58.13 [-59.84, -56.39] | 1.000 | ✗ |
| Pose OKS vs original, 5 head kpts (x100) | 400 | 100.0 | redact_box | 35.8 | -64.15 [-65.88, -62.41] | 1.000 | ✗ |
| Pose OKS vs original, 5 head kpts (x100) | 400 | 100.0 | transform | 50.9 | -49.06 [-50.92, -47.17] | 1.000 | ✗ |
| Person recall vs original (%) | 400 | 100.0 | reencode | 98.9 | -1.05 [-1.24, -0.87] | 0.000 | **EQ** |
| Person recall vs original (%) | 400 | 100.0 | redact_blur | 94.0 | -6.01 [-6.86, -5.25] | 1.000 | ✗ |
| Person recall vs original (%) | 400 | 100.0 | redact_box | 93.5 | -6.50 [-7.31, -5.71] | 1.000 | ✗ |
| Person recall vs original (%) | 400 | 100.0 | transform | 94.8 | -5.25 [-5.94, -4.60] | 1.000 | ✗ |
| Person AP50 vs MOT17 GT (x100) | 21 | 80.1 | reencode | 80.0 | -0.12 [-0.43, 0.18] | 0.000 | **EQ** |
| Person AP50 vs MOT17 GT (x100) | 21 | 80.1 | redact_blur | 79.9 | -0.20 [-0.54, 0.11] | 0.000 | **EQ** |
| Person AP50 vs MOT17 GT (x100) | 21 | 80.1 | redact_box | 79.5 | -0.67 [-1.27, -0.11] | 0.000 | **EQ** |
| Person AP50 vs MOT17 GT (x100) | 21 | 80.1 | transform | 79.8 | -0.34 [-0.69, -0.01] | 0.000 | **EQ** |
| Person recall@0.5 vs GT (%) | 21 | 81.8 | reencode | 81.6 | -0.23 [-0.64, 0.15] | 0.000 | **EQ** |
| Person recall@0.5 vs GT (%) | 21 | 81.8 | redact_blur | 81.3 | -0.55 [-0.87, -0.22] | 0.000 | **EQ** |
| Person recall@0.5 vs GT (%) | 21 | 81.8 | redact_box | 80.8 | -1.01 [-1.65, -0.37] | 0.002 | **EQ** |
| Person recall@0.5 vs GT (%) | 21 | 81.8 | transform | 81.2 | -0.63 [-1.00, -0.29] | 0.000 | **EQ** |
| Person count MAE / frame (lower=better) | 21 | 5.4 | reencode | 5.4 | 0.01 [-0.19, 0.24] | – | – |
| Person count MAE / frame (lower=better) | 21 | 5.4 | redact_blur | 5.2 | -0.25 [-0.44, -0.08] | – | – |
| Person count MAE / frame (lower=better) | 21 | 5.4 | redact_box | 4.8 | -0.63 [-1.04, -0.28] | – | – |
| Person count MAE / frame (lower=better) | 21 | 5.4 | transform | 5.2 | -0.19 [-0.40, 0.01] | – | – |
| Action top-1, VideoMAE-B (%) [face>=96px subset] | 161 | 72.0 | reencode | 73.3 | 1.24 [-1.86, 4.35] | 0.309 | ✗ |
| Action top-1, VideoMAE-B (%) [face>=96px subset] | 161 | 72.0 | redact_blur | 50.9 | -21.12 [-28.57, -14.29] | 1.000 | ✗ |
| Action top-1, VideoMAE-B (%) [face>=96px subset] | 161 | 72.0 | redact_box | 57.1 | -14.91 [-21.74, -8.07] | 1.000 | ✗ |
| Action top-1, VideoMAE-B (%) [face>=96px subset] | 161 | 72.0 | transform | 59.0 | -13.04 [-18.63, -7.45] | 1.000 | ✗ |
| Action top-1, X-CLIP B/32 (%) [face>=96px subset] | 161 | 74.5 | reencode | 74.5 | 0.00 [0.00, 0.00] | 0.000 | **EQ** |
| Action top-1, X-CLIP B/32 (%) [face>=96px subset] | 161 | 74.5 | redact_blur | 60.9 | -13.66 [-20.50, -6.83] | 0.999 | ✗ |
| Action top-1, X-CLIP B/32 (%) [face>=96px subset] | 161 | 74.5 | redact_box | 64.0 | -10.56 [-16.77, -4.35] | 0.996 | ✗ |
| Action top-1, X-CLIP B/32 (%) [face>=96px subset] | 161 | 74.5 | transform | 60.2 | -14.29 [-21.12, -8.07] | 1.000 | ✗ |

**Transform − redact_blur** (paired, 95% CI):

| Metric | n | Δ [95% CI] |
|---|---|---|
| Action top-1, VideoMAE-B (%) | 400 | 3.75 [1.00, 6.50] |
| Action top-5, VideoMAE-B (%) | 400 | -1.00 [-3.50, 1.50] |
| Action top-1, X-CLIP B/32 (%) | 400 | 0.50 [-2.25, 3.50] |
| Action top-5, X-CLIP B/32 (%) | 400 | 0.75 [-2.00, 3.25] |
| VideoMAE prediction = original's (%) | 400 | 5.75 [2.50, 9.00] |
| Pose OKS vs original, all 17 kpts (x100) | 400 | 6.09 [5.21, 6.99] |
| Pose OKS vs original, 5 head kpts (x100) | 400 | 9.07 [7.69, 10.52] |
| Person recall vs original (%) | 400 | 0.76 [0.22, 1.35] |
| Person AP50 vs MOT17 GT (x100) | 21 | -0.13 [-0.33, 0.03] |
| Person recall@0.5 vs GT (%) | 21 | -0.09 [-0.24, 0.04] |
| Person count MAE / frame (lower=better) | 21 | 0.07 [-0.04, 0.19] |
| Action top-1, VideoMAE-B (%) [face>=96px subset] | 161 | 8.07 [2.48, 13.66] |
| Action top-1, X-CLIP B/32 (%) [face>=96px subset] | 161 | -0.62 [-6.83, 4.97] |

## 3. Pipeline diagnostics

| Arm | Dataset | face tracks | tracks given a surrogate | tracks blur fallback | median fps (audit) |
|---|---|---|---|---|---|
| reencode | k400 | 0 | 0 | 0 | 4.1 |
| reencode | mot17 | 0 | 0 | 0 | 3.6 |
| reencode | vox2 | 0 | 0 | 0 | 2.4 |
| redact_blur | k400 | 7078 | 0 | 0 | 4.7 |
| redact_blur | mot17 | 730 | 0 | 0 | 4.2 |
| redact_blur | vox2 | 308 | 0 | 0 | 3.5 |
| redact_box | k400 | 7078 | 0 | 0 | 4.8 |
| redact_box | mot17 | 730 | 0 | 0 | 4.3 |
| redact_box | vox2 | 308 | 0 | 0 | 3.6 |
| transform | k400 | 7078 | 892 | 6355 | 1.9 |
| transform | mot17 | 730 | 22 | 720 | 2.5 |
| transform | vox2 | 308 | 64 | 244 | 1.6 |

| Arm | runs | failed | total wall (min) | median wall / clip (s) | frames |
|---|---|---|---|---|---|
| reencode | 481 | 0 | 525.1 | 70.3 | 119174 |
| redact_blur | 481 | 0 | 481.2 | 57.6 | 119174 |
| redact_box | 481 | 0 | 434.8 | 58.6 | 119174 |
| transform | 481 | 0 | 1108.4 | 125.6 | 119174 |

## 4. Change vs `v0.2`

| Headline | v0.2 | v0.2-gen2 |
|---|---|---|
| redact_blur re-ID % | 14.5 | 15.6 |
| redact_blur leak % | 2.4 | 2.5 |
| redact_blur face present % | 6.5 | 7.5 |
| redact_blur Δ Action top-1, VideoMAE-B (%) | -12.00 | -11.50 |
| redact_blur Δ Action top-5, VideoMAE-B (%) | -8.50 | -8.25 |
| redact_blur Δ Action top-1, X-CLIP B/32 (%) | -10.75 | -10.75 |
| redact_blur Δ Action top-5, X-CLIP B/32 (%) | -7.75 | -8.75 |
| redact_blur Δ VideoMAE prediction = original's (%) | -23.75 | -24.50 |
| redact_blur Δ Pose OKS vs original, all 17 kpts (x100) | -40.86 | -41.00 |
| redact_blur Δ Pose OKS vs original, 5 head kpts (x100) | -58.06 | -58.13 |
| redact_blur Δ Person recall vs original (%) | -6.94 | -6.01 |
| redact_blur Δ Person AP50 vs MOT17 GT (x100) | -0.73 | -0.20 |
| redact_blur Δ Person recall@0.5 vs GT (%) | -1.24 | -0.55 |
| redact_blur Δ Person count MAE / frame (lower=better) | -0.40 | -0.25 |
| redact_box re-ID % | 9.6 | 10.8 |
| redact_box leak % | 1.1 | 1.3 |
| redact_box face present % | 1.2 | 1.6 |
| redact_box Δ Action top-1, VideoMAE-B (%) | -9.75 | -9.25 |
| redact_box Δ Action top-5, VideoMAE-B (%) | -8.25 | -9.25 |
| redact_box Δ Action top-1, X-CLIP B/32 (%) | -7.25 | -10.00 |
| redact_box Δ Action top-5, X-CLIP B/32 (%) | -6.50 | -6.75 |
| redact_box Δ VideoMAE prediction = original's (%) | -21.75 | -22.75 |
| redact_box Δ Pose OKS vs original, all 17 kpts (x100) | -45.26 | -45.58 |
| redact_box Δ Pose OKS vs original, 5 head kpts (x100) | -64.21 | -64.15 |
| redact_box Δ Person recall vs original (%) | -6.47 | -6.50 |
| redact_box Δ Person AP50 vs MOT17 GT (x100) | -1.25 | -0.67 |
| redact_box Δ Person recall@0.5 vs GT (%) | -1.86 | -1.01 |
| redact_box Δ Person count MAE / frame (lower=better) | -0.61 | -0.63 |
| transform re-ID % | 12.7 | 13.2 |
| transform leak % | 3.3 | 3.5 |
| transform face present % | 62.2 | 64.2 |
| transform Δ Action top-1, VideoMAE-B (%) | -9.50 | -7.75 |
| transform Δ Action top-5, VideoMAE-B (%) | -7.25 | -9.25 |
| transform Δ Action top-1, X-CLIP B/32 (%) | -11.75 | -10.25 |
| transform Δ Action top-5, X-CLIP B/32 (%) | -9.00 | -8.00 |
| transform Δ VideoMAE prediction = original's (%) | -21.00 | -18.75 |
| transform Δ Pose OKS vs original, all 17 kpts (x100) | -34.66 | -34.91 |
| transform Δ Pose OKS vs original, 5 head kpts (x100) | -48.83 | -49.06 |
| transform Δ Person recall vs original (%) | -5.55 | -5.25 |
| transform Δ Person AP50 vs MOT17 GT (x100) | -0.71 | -0.34 |
| transform Δ Person recall@0.5 vs GT (%) | -1.16 | -0.63 |
| transform Δ Person count MAE / frame (lower=better) | -0.37 | -0.19 |
