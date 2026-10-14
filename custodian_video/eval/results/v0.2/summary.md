# Video Guardian benchmark: results for pipeline tag `v0.2`

Generated 2026-10-02T21:38:45 by `custodian_video/eval/analyze.py` (metric version m2). Pipeline snapshot: source `git:f272950`, commit `f272950266`, files sha256 `9f731ad6c2f8f125`, reported version `0.2.0`. Only the face channel is enabled (`--channels faces`).

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
| redact_blur | all faces | 479 / 10943 | 14.5 [12.6, 16.3] | 9.2 [8.8, 9.6] | 4.9 [4.1, 5.9] | 6.2 [5.4, 7.2] | 2.4 [1.8, 3.0] | 6.5 [5.4, 7.6] | 98.8 [98.4, 99.1] | 0.119 |
| redact_blur | faces >=64px | 403 / 5439 | 12.1 [10.0, 14.3] | 8.9 [8.5, 9.3] | 5.1 [3.8, 6.6] | 6.0 [4.6, 7.5] | 2.6 [1.4, 3.9] | 6.0 [4.4, 7.6] | 97.8 [96.6, 98.8] | 0.116 |
| redact_box | all faces | 479 / 10943 | 9.6 [8.3, 11.0] | 7.6 [7.2, 7.9] | 4.3 [3.4, 5.4] | 3.4 [2.8, 4.0] | 1.1 [0.8, 1.5] | 1.2 [0.9, 1.6] | 98.8 [98.4, 99.1] | 0.089 |
| redact_box | faces >=64px | 403 / 5439 | 9.1 [7.3, 11.0] | 7.1 [6.8, 7.5] | 5.8 [4.3, 7.5] | 3.3 [2.2, 4.6] | 2.1 [1.1, 3.3] | 2.3 [1.3, 3.5] | 97.8 [96.6, 98.8] | 0.092 |
| transform | all faces | 479 / 10943 | 12.7 [11.1, 14.5] | 5.0 [4.6, 5.5] | 7.0 [6.1, 8.1] | 7.4 [6.4, 8.5] | 3.3 [2.7, 4.1] | 62.2 [58.9, 65.5] | 98.8 [98.4, 99.1] | 0.106 |
| transform | faces >=64px | 403 / 5439 | 10.9 [8.9, 13.1] | 3.3 [3.0, 3.7] | 8.9 [7.2, 10.6] | 8.1 [6.5, 9.8] | 3.9 [2.7, 5.3] | 79.6 [76.5, 82.5] | 97.8 [96.6, 98.8] | 0.104 |

By dataset (all faces): re-ID ArcFace % / impostor % / rank-1 of 100 % / leak % / face present %

| Arm | Kinetics-400 | MOT17 | VoxCeleb2 |
|---|---|---|---|
| original | 100.0 / 1.0 / 100.0 / 100.0 / 100.0 | 100.0 / 6.1 / 100.0 / 100.0 / 100.0 | 100.0 / 0.1 / 100.0 / 100.0 / 100.0 |
| reencode | 100.0 / 1.1 / 100.0 / 99.6 / 99.8 | 100.0 / 6.7 / 100.0 / 98.8 / 99.0 | 100.0 / 0.1 / 100.0 / 100.0 / 100.0 |
| redact_blur | 15.0 / 9.9 / 4.4 / 2.3 / 6.2 | 42.3 / 21.4 / 9.7 / 11.6 / 16.8 | 2.1 / 0.4 / 6.8 / 0.0 / 4.9 |
| redact_box | 9.9 / 8.1 / 3.3 / 1.3 / 1.5 | 27.9 / 20.9 / 4.0 / 0.6 / 0.6 | 1.9 / 0.1 / 11.3 / 0.0 / 0.0 |
| transform | 13.2 / 5.2 / 7.0 / 3.4 / 59.5 | 41.1 / 17.2 / 11.0 / 11.2 / 33.0 | 0.8 / 0.1 / 6.1 / 0.2 / 89.6 |

**Cross-video identification (VoxCeleb2).** Each probe clip is identified against a gallery built from a *different* YouTube video of each of 118 people (chance = 0.85%). Rank-1 % with 95% CI. The clip-level score uses the mean embedding over 16 frames.

| Arm | Attacker | probes | rank-1 per clip % | rank-1 per frame % |
|---|---|---|---|---|
| original | same-location attacker | 60 | 100.0 [100.0, 100.0] | 99.5 [98.8, 100.0] |
| original | detector attacker | 60 | 100.0 [100.0, 100.0] | 99.5 [98.8, 100.0] |
| reencode | same-location attacker | 60 | 100.0 [100.0, 100.0] | 99.6 [99.1, 100.0] |
| reencode | detector attacker | 60 | 100.0 [100.0, 100.0] | 99.5 [98.6, 100.0] |
| redact_blur | same-location attacker | 60 | 6.7 [1.7, 13.3] | 5.9 [2.0, 10.9] |
| redact_blur | detector attacker | 60 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| redact_box | same-location attacker | 60 | 5.0 [0.0, 11.7] | 2.5 [0.8, 4.6] |
| redact_box | detector attacker | 60 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| transform | same-location attacker | 60 | 3.3 [0.0, 8.3] | 1.2 [0.0, 2.8] |
| transform | detector attacker | 60 | 3.3 [0.0, 8.3] | 2.2 [0.3, 4.8] |

## 2. Utility: original vs each arm (paired per clip)

Δ = arm − original (points). 95% paired-bootstrap CI. TOST equivalence margin ±2 points (α=0.05): **EQ** = equivalent; ✗ = not shown equivalent.

| Metric | n | Original | Arm | Value | Δ [95% CI] | TOST p | Equiv. ±2 |
|---|---|---|---|---|---|---|---|
| Action top-1, VideoMAE-B (%) | 400 | 72.8 | reencode | 73.8 | 1.00 [-0.25, 2.50] | 0.078 | ✗ |
| Action top-1, VideoMAE-B (%) | 400 | 72.8 | redact_blur | 60.8 | -12.00 [-15.75, -8.25] | 1.000 | ✗ |
| Action top-1, VideoMAE-B (%) | 400 | 72.8 | redact_box | 63.0 | -9.75 [-13.25, -6.25] | 1.000 | ✗ |
| Action top-1, VideoMAE-B (%) | 400 | 72.8 | transform | 63.2 | -9.50 [-13.00, -6.50] | 1.000 | ✗ |
| Action top-5, VideoMAE-B (%) | 400 | 91.5 | reencode | 91.5 | 0.00 [-1.00, 1.00] | 0.000 | **EQ** |
| Action top-5, VideoMAE-B (%) | 400 | 91.5 | redact_blur | 83.0 | -8.50 [-11.50, -5.75] | 1.000 | ✗ |
| Action top-5, VideoMAE-B (%) | 400 | 91.5 | redact_box | 83.2 | -8.25 [-11.25, -5.25] | 1.000 | ✗ |
| Action top-5, VideoMAE-B (%) | 400 | 91.5 | transform | 84.2 | -7.25 [-10.00, -4.50] | 1.000 | ✗ |
| Action top-1, X-CLIP B/32 (%) | 400 | 75.2 | reencode | 76.0 | 0.75 [-0.25, 2.00] | 0.013 | **EQ** |
| Action top-1, X-CLIP B/32 (%) | 400 | 75.2 | redact_blur | 64.5 | -10.75 [-14.25, -7.25] | 1.000 | ✗ |
| Action top-1, X-CLIP B/32 (%) | 400 | 75.2 | redact_box | 68.0 | -7.25 [-10.75, -3.74] | 0.998 | ✗ |
| Action top-1, X-CLIP B/32 (%) | 400 | 75.2 | transform | 63.5 | -11.75 [-15.00, -8.50] | 1.000 | ✗ |
| Action top-5, X-CLIP B/32 (%) | 400 | 91.8 | reencode | 91.2 | -0.50 [-1.50, 0.50] | 0.001 | **EQ** |
| Action top-5, X-CLIP B/32 (%) | 400 | 91.8 | redact_blur | 84.0 | -7.75 [-10.51, -5.00] | 1.000 | ✗ |
| Action top-5, X-CLIP B/32 (%) | 400 | 91.8 | redact_box | 85.2 | -6.50 [-9.25, -4.00] | 0.999 | ✗ |
| Action top-5, X-CLIP B/32 (%) | 400 | 91.8 | transform | 82.8 | -9.00 [-12.25, -6.00] | 1.000 | ✗ |
| VideoMAE prediction = original's (%) | 400 | 100.0 | reencode | 97.2 | -2.75 [-4.50, -1.25] | 0.820 | ✗ |
| VideoMAE prediction = original's (%) | 400 | 100.0 | redact_blur | 76.2 | -23.75 [-28.00, -19.75] | 1.000 | ✗ |
| VideoMAE prediction = original's (%) | 400 | 100.0 | redact_box | 78.2 | -21.75 [-26.00, -17.75] | 1.000 | ✗ |
| VideoMAE prediction = original's (%) | 400 | 100.0 | transform | 79.0 | -21.00 [-25.00, -17.00] | 1.000 | ✗ |
| Pose OKS vs original, all 17 kpts (x100) | 400 | 100.0 | reencode | 84.3 | -15.70 [-16.59, -14.84] | 1.000 | ✗ |
| Pose OKS vs original, all 17 kpts (x100) | 400 | 100.0 | redact_blur | 59.1 | -40.86 [-42.66, -39.11] | 1.000 | ✗ |
| Pose OKS vs original, all 17 kpts (x100) | 400 | 100.0 | redact_box | 54.7 | -45.26 [-47.00, -43.46] | 1.000 | ✗ |
| Pose OKS vs original, all 17 kpts (x100) | 400 | 100.0 | transform | 65.3 | -34.66 [-36.26, -33.05] | 1.000 | ✗ |
| Pose OKS vs original, 5 head kpts (x100) | 400 | 100.0 | reencode | 85.7 | -14.29 [-15.27, -13.33] | 1.000 | ✗ |
| Pose OKS vs original, 5 head kpts (x100) | 400 | 100.0 | redact_blur | 41.9 | -58.06 [-59.75, -56.33] | 1.000 | ✗ |
| Pose OKS vs original, 5 head kpts (x100) | 400 | 100.0 | redact_box | 35.8 | -64.21 [-65.92, -62.46] | 1.000 | ✗ |
| Pose OKS vs original, 5 head kpts (x100) | 400 | 100.0 | transform | 51.2 | -48.83 [-50.68, -46.96] | 1.000 | ✗ |
| Person recall vs original (%) | 400 | 100.0 | reencode | 98.9 | -1.05 [-1.24, -0.87] | 0.000 | **EQ** |
| Person recall vs original (%) | 400 | 100.0 | redact_blur | 93.1 | -6.94 [-8.02, -5.98] | 1.000 | ✗ |
| Person recall vs original (%) | 400 | 100.0 | redact_box | 93.5 | -6.47 [-7.35, -5.60] | 1.000 | ✗ |
| Person recall vs original (%) | 400 | 100.0 | transform | 94.4 | -5.55 [-6.31, -4.84] | 1.000 | ✗ |
| Person AP50 vs MOT17 GT (x100) | 21 | 80.1 | reencode | 80.0 | -0.12 [-0.43, 0.18] | 0.000 | **EQ** |
| Person AP50 vs MOT17 GT (x100) | 21 | 80.1 | redact_blur | 79.4 | -0.73 [-1.43, -0.16] | 0.000 | **EQ** |
| Person AP50 vs MOT17 GT (x100) | 21 | 80.1 | redact_box | 78.9 | -1.25 [-1.99, -0.61] | 0.021 | **EQ** |
| Person AP50 vs MOT17 GT (x100) | 21 | 80.1 | transform | 79.4 | -0.71 [-1.38, -0.16] | 0.000 | **EQ** |
| Person recall@0.5 vs GT (%) | 21 | 81.8 | reencode | 81.6 | -0.23 [-0.64, 0.15] | 0.000 | **EQ** |
| Person recall@0.5 vs GT (%) | 21 | 81.8 | redact_blur | 80.6 | -1.24 [-2.22, -0.44] | 0.055 | ✗ |
| Person recall@0.5 vs GT (%) | 21 | 81.8 | redact_box | 80.0 | -1.86 [-2.99, -0.92] | 0.397 | ✗ |
| Person recall@0.5 vs GT (%) | 21 | 81.8 | transform | 80.7 | -1.16 [-2.12, -0.38] | 0.035 | **EQ** |
| Person count MAE / frame (lower=better) | 21 | 5.4 | reencode | 5.4 | 0.01 [-0.19, 0.24] | – | – |
| Person count MAE / frame (lower=better) | 21 | 5.4 | redact_blur | 5.0 | -0.40 [-0.68, -0.13] | – | – |
| Person count MAE / frame (lower=better) | 21 | 5.4 | redact_box | 4.8 | -0.61 [-1.49, 0.12] | – | – |
| Person count MAE / frame (lower=better) | 21 | 5.4 | transform | 5.0 | -0.37 [-0.67, -0.10] | – | – |
| Action top-1, VideoMAE-B (%) [face>=96px subset] | 161 | 72.0 | reencode | 73.3 | 1.24 [-1.86, 4.35] | 0.309 | ✗ |
| Action top-1, VideoMAE-B (%) [face>=96px subset] | 161 | 72.0 | redact_blur | 51.6 | -20.50 [-27.95, -13.66] | 1.000 | ✗ |
| Action top-1, VideoMAE-B (%) [face>=96px subset] | 161 | 72.0 | redact_box | 55.3 | -16.77 [-22.98, -10.56] | 1.000 | ✗ |
| Action top-1, VideoMAE-B (%) [face>=96px subset] | 161 | 72.0 | transform | 57.1 | -14.91 [-21.12, -8.70] | 1.000 | ✗ |
| Action top-1, X-CLIP B/32 (%) [face>=96px subset] | 161 | 74.5 | reencode | 74.5 | 0.00 [0.00, 0.00] | 0.000 | **EQ** |
| Action top-1, X-CLIP B/32 (%) [face>=96px subset] | 161 | 74.5 | redact_blur | 60.2 | -14.29 [-21.74, -7.45] | 1.000 | ✗ |
| Action top-1, X-CLIP B/32 (%) [face>=96px subset] | 161 | 74.5 | redact_box | 67.1 | -7.45 [-13.66, -1.24] | 0.954 | ✗ |
| Action top-1, X-CLIP B/32 (%) [face>=96px subset] | 161 | 74.5 | transform | 59.0 | -15.53 [-21.74, -9.94] | 1.000 | ✗ |

**Transform − redact_blur** (paired, 95% CI):

| Metric | n | Δ [95% CI] |
|---|---|---|
| Action top-1, VideoMAE-B (%) | 400 | 2.50 [-0.25, 5.50] |
| Action top-5, VideoMAE-B (%) | 400 | 1.25 [-1.00, 3.50] |
| Action top-1, X-CLIP B/32 (%) | 400 | -1.00 [-4.00, 1.75] |
| Action top-5, X-CLIP B/32 (%) | 400 | -1.25 [-4.25, 1.50] |
| VideoMAE prediction = original's (%) | 400 | 2.75 [-0.75, 6.25] |
| Pose OKS vs original, all 17 kpts (x100) | 400 | 6.21 [5.31, 7.13] |
| Pose OKS vs original, 5 head kpts (x100) | 400 | 9.23 [7.84, 10.68] |
| Person recall vs original (%) | 400 | 1.39 [0.73, 2.12] |
| Person AP50 vs MOT17 GT (x100) | 21 | 0.02 [-0.06, 0.13] |
| Person recall@0.5 vs GT (%) | 21 | 0.09 [-0.08, 0.27] |
| Person count MAE / frame (lower=better) | 21 | 0.02 [-0.03, 0.07] |
| Action top-1, VideoMAE-B (%) [face>=96px subset] | 161 | 5.59 [0.00, 11.18] |
| Action top-1, X-CLIP B/32 (%) [face>=96px subset] | 161 | -1.24 [-6.83, 4.35] |

## 3. Pipeline diagnostics

| Arm | Dataset | face tracks | tracks given a surrogate | tracks blur fallback | median fps (audit) |
|---|---|---|---|---|---|
| reencode | k400 | 0 | 0 | 0 | 5.3 |
| reencode | mot17 | 0 | 0 | 0 | 2.6 |
| reencode | vox2 | 0 | 0 | 0 | 6.8 |
| redact_blur | k400 | 6639 | 0 | 0 | 5.0 |
| redact_blur | mot17 | 1127 | 0 | 0 | 2.6 |
| redact_blur | vox2 | 81 | 0 | 0 | 5.6 |
| redact_box | k400 | 6639 | 0 | 0 | 5.1 |
| redact_box | mot17 | 1127 | 0 | 0 | 2.7 |
| redact_box | vox2 | 81 | 0 | 0 | 6.5 |
| transform | k400 | 6639 | 843 | 5956 | 2.7 |
| transform | mot17 | 1127 | 22 | 1117 | 2.5 |
| transform | vox2 | 81 | 61 | 20 | 2.3 |

| Arm | runs | failed | total wall (min) | median wall / clip (s) | frames |
|---|---|---|---|---|---|
| reencode | 481 | 0 | 376.2 | 48.8 | 119174 |
| redact_blur | 481 | 0 | 478.3 | 52.8 | 119174 |
| redact_box | 481 | 0 | 388.8 | 50.7 | 119174 |
| transform | 481 | 0 | 804.9 | 90.2 | 119174 |

## 4. Change vs `v0.1`

| Headline | v0.1 | v0.2 |
|---|---|---|
| redact_blur re-ID % | 15.0 | 14.5 |
| redact_blur leak % | 3.1 | 2.4 |
| redact_blur face present % | 7.4 | 6.5 |
| redact_blur Δ Action top-1, VideoMAE-B (%) | -11.25 | -12.00 |
| redact_blur Δ Action top-5, VideoMAE-B (%) | -7.75 | -8.50 |
| redact_blur Δ Action top-1, X-CLIP B/32 (%) | -9.00 | -10.75 |
| redact_blur Δ Action top-5, X-CLIP B/32 (%) | -7.00 | -7.75 |
| redact_blur Δ VideoMAE prediction = original's (%) | -22.50 | -23.75 |
| redact_blur Δ Pose OKS vs original, all 17 kpts (x100) | -39.52 | -40.86 |
| redact_blur Δ Pose OKS vs original, 5 head kpts (x100) | -56.81 | -58.06 |
| redact_blur Δ Person recall vs original (%) | -5.62 | -6.94 |
| redact_blur Δ Person AP50 vs MOT17 GT (x100) | -0.01 | -0.73 |
| redact_blur Δ Person recall@0.5 vs GT (%) | -0.35 | -1.24 |
| redact_blur Δ Person count MAE / frame (lower=better) | -0.28 | -0.40 |
| redact_box re-ID % | 9.6 | 9.6 |
| redact_box leak % | 1.6 | 1.1 |
| redact_box face present % | 1.8 | 1.2 |
| redact_box Δ Action top-1, VideoMAE-B (%) | -8.25 | -9.75 |
| redact_box Δ Action top-5, VideoMAE-B (%) | -7.75 | -8.25 |
| redact_box Δ Action top-1, X-CLIP B/32 (%) | -7.50 | -7.25 |
| redact_box Δ Action top-5, X-CLIP B/32 (%) | -6.00 | -6.50 |
| redact_box Δ VideoMAE prediction = original's (%) | -20.75 | -21.75 |
| redact_box Δ Pose OKS vs original, all 17 kpts (x100) | -43.92 | -45.26 |
| redact_box Δ Pose OKS vs original, 5 head kpts (x100) | -62.73 | -64.21 |
| redact_box Δ Person recall vs original (%) | -5.79 | -6.47 |
| redact_box Δ Person AP50 vs MOT17 GT (x100) | -0.11 | -1.25 |
| redact_box Δ Person recall@0.5 vs GT (%) | -0.84 | -1.86 |
| redact_box Δ Person count MAE / frame (lower=better) | -0.53 | -0.61 |
| transform re-ID % | 12.8 | 12.7 |
| transform leak % | 4.0 | 3.3 |
| transform face present % | 54.7 | 62.2 |
| transform Δ Action top-1, VideoMAE-B (%) | -9.50 | -9.50 |
| transform Δ Action top-5, VideoMAE-B (%) | -10.75 | -7.25 |
| transform Δ Action top-1, X-CLIP B/32 (%) | -13.00 | -11.75 |
| transform Δ Action top-5, X-CLIP B/32 (%) | -10.75 | -9.00 |
| transform Δ VideoMAE prediction = original's (%) | -19.00 | -21.00 |
| transform Δ Pose OKS vs original, all 17 kpts (x100) | -35.88 | -34.66 |
| transform Δ Pose OKS vs original, 5 head kpts (x100) | -51.38 | -48.83 |
| transform Δ Person recall vs original (%) | -5.18 | -5.55 |
| transform Δ Person AP50 vs MOT17 GT (x100) | -0.01 | -0.71 |
| transform Δ Person recall@0.5 vs GT (%) | -0.40 | -1.16 |
| transform Δ Person count MAE / frame (lower=better) | -0.27 | -0.37 |
