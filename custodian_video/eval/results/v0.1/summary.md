# Video Guardian benchmark: results for pipeline tag `v0.1`

Generated 2026-10-02T14:59:44 by `custodian_video/eval/analyze.py` (metric version m2). Pipeline snapshot: source `git:4dd3004`, commit `4dd3004c7d`, files sha256 `783100026e12ea77`, reported version `0.1.0`. Only the face channel is enabled (`--channels faces`).

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
| reencode | all faces | 479 / 10943 | 100.0 [100.0, 100.0] | 1.2 [1.0, 1.3] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.6 [99.4, 99.8] | 99.8 [99.6, 99.9] | 0.0 [0.0, 0.0] | 0.949 |
| reencode | faces >=64px | 403 / 5439 | 100.0 [100.0, 100.0] | 0.9 [0.8, 1.1] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.5 [98.9, 99.9] | 99.9 [99.9, 100.0] | 0.0 [0.0, 0.0] | 0.961 |
| redact_blur | all faces | 479 / 10943 | 15.0 [13.1, 16.9] | 9.0 [8.6, 9.4] | 5.5 [4.6, 6.4] | 7.4 [6.4, 8.4] | 3.1 [2.5, 3.8] | 7.4 [6.3, 8.6] | 98.2 [97.7, 98.6] | 0.125 |
| redact_blur | faces >=64px | 403 / 5439 | 12.3 [10.2, 14.5] | 8.9 [8.5, 9.3] | 5.7 [4.2, 7.4] | 7.3 [5.7, 9.1] | 3.8 [2.5, 5.4] | 7.5 [5.7, 9.3] | 96.9 [95.5, 98.1] | 0.122 |
| redact_box | all faces | 479 / 10943 | 9.6 [8.2, 11.0] | 7.4 [7.0, 7.7] | 4.6 [3.7, 5.6] | 4.3 [3.6, 5.1] | 1.6 [1.2, 2.1] | 1.8 [1.4, 2.3] | 98.2 [97.7, 98.6] | 0.092 |
| redact_box | faces >=64px | 403 / 5439 | 8.9 [7.1, 10.9] | 7.0 [6.7, 7.4] | 6.2 [4.7, 8.0] | 4.4 [3.1, 5.9] | 3.0 [1.8, 4.4] | 3.1 [1.9, 4.5] | 96.9 [95.4, 98.1] | 0.098 |
| transform | all faces | 479 / 10943 | 12.8 [11.1, 14.6] | 5.6 [5.3, 6.1] | 6.1 [5.1, 7.1] | 8.7 [7.5, 9.9] | 4.0 [3.2, 4.9] | 54.7 [51.2, 58.2] | 98.2 [97.7, 98.6] | 0.102 |
| transform | faces >=64px | 403 / 5439 | 10.4 [8.3, 12.7] | 4.2 [3.8, 4.6] | 6.7 [5.1, 8.5] | 9.0 [7.2, 10.9] | 5.0 [3.5, 6.6] | 69.8 [66.1, 73.6] | 96.9 [95.4, 98.1] | 0.097 |

By dataset (all faces): re-ID ArcFace % / impostor % / rank-1 of 100 % / leak % / face present %

| Arm | Kinetics-400 | MOT17 | VoxCeleb2 |
|---|---|---|---|
| original | 100.0 / 1.0 / 100.0 / 100.0 / 100.0 | 100.0 / 6.1 / 100.0 / 100.0 / 100.0 | 100.0 / 0.1 / 100.0 / 100.0 / 100.0 |
| reencode | 100.0 / 1.1 / 100.0 / 99.6 / 99.8 | 100.0 / 6.6 / 100.0 / 98.8 / 99.0 | 100.0 / 0.1 / 100.0 / 100.0 / 100.0 |
| redact_blur | 15.4 / 9.8 / 4.9 / 3.1 / 7.3 | 45.0 / 20.1 / 15.6 / 13.3 / 17.8 | 2.4 / 0.4 / 5.8 / 0.1 / 4.6 |
| redact_box | 9.9 / 7.9 / 3.6 / 1.8 / 2.0 | 27.4 / 19.7 / 5.5 / 2.2 / 3.6 | 1.7 / 0.1 / 10.7 / 0.0 / 0.1 |
| transform | 13.1 / 5.9 / 5.8 / 4.2 / 53.9 | 43.1 / 17.9 / 15.9 / 13.2 / 34.1 | 0.6 / 0.1 / 4.7 / 0.0 / 66.3 |

**Cross-video identification (VoxCeleb2).** Each probe clip is identified against a gallery built from a *different* YouTube video of each of 118 people (chance = 0.85%). Rank-1 % with 95% CI. The clip-level score uses the mean embedding over 16 frames.

| Arm | Attacker | probes | rank-1 per clip % | rank-1 per frame % |
|---|---|---|---|---|
| original | same-location attacker | 60 | 100.0 [100.0, 100.0] | 99.5 [98.8, 100.0] |
| original | detector attacker | 60 | 100.0 [100.0, 100.0] | 99.5 [98.8, 100.0] |
| reencode | same-location attacker | 60 | 100.0 [100.0, 100.0] | 99.6 [99.1, 100.0] |
| reencode | detector attacker | 60 | 100.0 [100.0, 100.0] | 99.5 [98.6, 100.0] |
| redact_blur | same-location attacker | 60 | 6.7 [1.7, 13.3] | 5.3 [1.6, 10.2] |
| redact_blur | detector attacker | 60 | 1.7 [0.0, 5.0] | 2.0 [0.0, 5.7] |
| redact_box | same-location attacker | 60 | 5.0 [0.0, 11.7] | 3.4 [1.0, 6.2] |
| redact_box | detector attacker | 60 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| transform | same-location attacker | 60 | 1.7 [0.0, 5.0] | 1.6 [0.0, 3.8] |
| transform | detector attacker | 60 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## 2. Utility: original vs each arm (paired per clip)

Δ = arm − original (points). 95% paired-bootstrap CI. TOST equivalence margin ±2 points (α=0.05): **EQ** = equivalent; ✗ = not shown equivalent.

| Metric | n | Original | Arm | Value | Δ [95% CI] | TOST p | Equiv. ±2 |
|---|---|---|---|---|---|---|---|
| Action top-1, VideoMAE-B (%) | 400 | 72.8 | reencode | 73.8 | 1.00 [-0.25, 2.50] | 0.078 | ✗ |
| Action top-1, VideoMAE-B (%) | 400 | 72.8 | redact_blur | 61.5 | -11.25 [-14.75, -7.75] | 1.000 | ✗ |
| Action top-1, VideoMAE-B (%) | 400 | 72.8 | redact_box | 64.5 | -8.25 [-11.75, -4.75] | 1.000 | ✗ |
| Action top-1, VideoMAE-B (%) | 400 | 72.8 | transform | 63.2 | -9.50 [-12.75, -6.25] | 1.000 | ✗ |
| Action top-5, VideoMAE-B (%) | 400 | 91.5 | reencode | 91.5 | 0.00 [-1.00, 1.00] | 0.000 | **EQ** |
| Action top-5, VideoMAE-B (%) | 400 | 91.5 | redact_blur | 83.8 | -7.75 [-10.50, -5.25] | 1.000 | ✗ |
| Action top-5, VideoMAE-B (%) | 400 | 91.5 | redact_box | 83.8 | -7.75 [-10.75, -5.00] | 1.000 | ✗ |
| Action top-5, VideoMAE-B (%) | 400 | 91.5 | transform | 80.8 | -10.75 [-14.00, -7.75] | 1.000 | ✗ |
| Action top-1, X-CLIP B/32 (%) | 400 | 75.2 | reencode | 76.0 | 0.75 [-0.25, 2.00] | 0.013 | **EQ** |
| Action top-1, X-CLIP B/32 (%) | 400 | 75.2 | redact_blur | 66.2 | -9.00 [-12.75, -5.50] | 1.000 | ✗ |
| Action top-1, X-CLIP B/32 (%) | 400 | 75.2 | redact_box | 67.8 | -7.50 [-11.00, -4.00] | 0.999 | ✗ |
| Action top-1, X-CLIP B/32 (%) | 400 | 75.2 | transform | 62.2 | -13.00 [-16.75, -9.25] | 1.000 | ✗ |
| Action top-5, X-CLIP B/32 (%) | 400 | 91.8 | reencode | 91.2 | -0.50 [-1.50, 0.50] | 0.001 | **EQ** |
| Action top-5, X-CLIP B/32 (%) | 400 | 91.8 | redact_blur | 84.8 | -7.00 [-10.00, -4.25] | 1.000 | ✗ |
| Action top-5, X-CLIP B/32 (%) | 400 | 91.8 | redact_box | 85.8 | -6.00 [-8.50, -3.50] | 0.999 | ✗ |
| Action top-5, X-CLIP B/32 (%) | 400 | 91.8 | transform | 81.0 | -10.75 [-14.00, -7.75] | 1.000 | ✗ |
| VideoMAE prediction = original's (%) | 400 | 100.0 | reencode | 97.2 | -2.75 [-4.50, -1.25] | 0.820 | ✗ |
| VideoMAE prediction = original's (%) | 400 | 100.0 | redact_blur | 77.5 | -22.50 [-26.75, -18.50] | 1.000 | ✗ |
| VideoMAE prediction = original's (%) | 400 | 100.0 | redact_box | 79.2 | -20.75 [-24.75, -16.75] | 1.000 | ✗ |
| VideoMAE prediction = original's (%) | 400 | 100.0 | transform | 81.0 | -19.00 [-22.75, -15.25] | 1.000 | ✗ |
| Pose OKS vs original, all 17 kpts (x100) | 400 | 100.0 | reencode | 84.3 | -15.70 [-16.59, -14.84] | 1.000 | ✗ |
| Pose OKS vs original, all 17 kpts (x100) | 400 | 100.0 | redact_blur | 60.5 | -39.52 [-41.32, -37.78] | 1.000 | ✗ |
| Pose OKS vs original, all 17 kpts (x100) | 400 | 100.0 | redact_box | 56.1 | -43.92 [-45.65, -42.10] | 1.000 | ✗ |
| Pose OKS vs original, all 17 kpts (x100) | 400 | 100.0 | transform | 64.1 | -35.88 [-37.51, -34.24] | 1.000 | ✗ |
| Pose OKS vs original, 5 head kpts (x100) | 400 | 100.0 | reencode | 85.7 | -14.29 [-15.27, -13.33] | 1.000 | ✗ |
| Pose OKS vs original, 5 head kpts (x100) | 400 | 100.0 | redact_blur | 43.2 | -56.81 [-58.53, -55.04] | 1.000 | ✗ |
| Pose OKS vs original, 5 head kpts (x100) | 400 | 100.0 | redact_box | 37.3 | -62.73 [-64.48, -60.95] | 1.000 | ✗ |
| Pose OKS vs original, 5 head kpts (x100) | 400 | 100.0 | transform | 48.6 | -51.38 [-53.32, -49.47] | 1.000 | ✗ |
| Person recall vs original (%) | 400 | 100.0 | reencode | 98.9 | -1.05 [-1.24, -0.87] | 0.000 | **EQ** |
| Person recall vs original (%) | 400 | 100.0 | redact_blur | 94.4 | -5.62 [-6.57, -4.77] | 1.000 | ✗ |
| Person recall vs original (%) | 400 | 100.0 | redact_box | 94.2 | -5.79 [-6.65, -4.98] | 1.000 | ✗ |
| Person recall vs original (%) | 400 | 100.0 | transform | 94.8 | -5.18 [-5.93, -4.48] | 1.000 | ✗ |
| Person AP50 vs MOT17 GT (x100) | 21 | 80.1 | reencode | 80.0 | -0.12 [-0.43, 0.18] | 0.000 | **EQ** |
| Person AP50 vs MOT17 GT (x100) | 21 | 80.1 | redact_blur | 80.1 | -0.01 [-0.33, 0.33] | 0.000 | **EQ** |
| Person AP50 vs MOT17 GT (x100) | 21 | 80.1 | redact_box | 80.0 | -0.11 [-0.55, 0.30] | 0.000 | **EQ** |
| Person AP50 vs MOT17 GT (x100) | 21 | 80.1 | transform | 80.1 | -0.01 [-0.29, 0.28] | 0.000 | **EQ** |
| Person recall@0.5 vs GT (%) | 21 | 81.8 | reencode | 81.6 | -0.23 [-0.64, 0.15] | 0.000 | **EQ** |
| Person recall@0.5 vs GT (%) | 21 | 81.8 | redact_blur | 81.5 | -0.35 [-0.76, 0.05] | 0.000 | **EQ** |
| Person recall@0.5 vs GT (%) | 21 | 81.8 | redact_box | 81.0 | -0.84 [-1.51, -0.21] | 0.000 | **EQ** |
| Person recall@0.5 vs GT (%) | 21 | 81.8 | transform | 81.4 | -0.40 [-0.82, 0.01] | 0.000 | **EQ** |
| Person count MAE / frame (lower=better) | 21 | 5.4 | reencode | 5.4 | 0.01 [-0.19, 0.24] | – | – |
| Person count MAE / frame (lower=better) | 21 | 5.4 | redact_blur | 5.1 | -0.28 [-0.48, -0.07] | – | – |
| Person count MAE / frame (lower=better) | 21 | 5.4 | redact_box | 4.9 | -0.53 [-1.04, -0.05] | – | – |
| Person count MAE / frame (lower=better) | 21 | 5.4 | transform | 5.1 | -0.27 [-0.49, -0.05] | – | – |
| Action top-1, VideoMAE-B (%) [face>=96px subset] | 161 | 72.0 | reencode | 73.3 | 1.24 [-1.86, 4.35] | 0.309 | ✗ |
| Action top-1, VideoMAE-B (%) [face>=96px subset] | 161 | 72.0 | redact_blur | 51.6 | -20.50 [-27.33, -13.66] | 1.000 | ✗ |
| Action top-1, VideoMAE-B (%) [face>=96px subset] | 161 | 72.0 | redact_box | 59.0 | -13.04 [-19.25, -6.83] | 1.000 | ✗ |
| Action top-1, VideoMAE-B (%) [face>=96px subset] | 161 | 72.0 | transform | 55.9 | -16.15 [-22.36, -9.94] | 1.000 | ✗ |
| Action top-1, X-CLIP B/32 (%) [face>=96px subset] | 161 | 74.5 | reencode | 74.5 | 0.00 [0.00, 0.00] | 0.000 | **EQ** |
| Action top-1, X-CLIP B/32 (%) [face>=96px subset] | 161 | 74.5 | redact_blur | 62.1 | -12.42 [-19.25, -6.21] | 0.999 | ✗ |
| Action top-1, X-CLIP B/32 (%) [face>=96px subset] | 161 | 74.5 | redact_box | 67.7 | -6.83 [-13.04, -0.62] | 0.942 | ✗ |
| Action top-1, X-CLIP B/32 (%) [face>=96px subset] | 161 | 74.5 | transform | 55.3 | -19.25 [-26.09, -13.04] | 1.000 | ✗ |

**Transform − redact_blur** (paired, 95% CI):

| Metric | n | Δ [95% CI] |
|---|---|---|
| Action top-1, VideoMAE-B (%) | 400 | 1.75 [-0.75, 4.50] |
| Action top-5, VideoMAE-B (%) | 400 | -3.00 [-5.25, -0.75] |
| Action top-1, X-CLIP B/32 (%) | 400 | -4.00 [-7.00, -1.00] |
| Action top-5, X-CLIP B/32 (%) | 400 | -3.75 [-6.76, -0.75] |
| VideoMAE prediction = original's (%) | 400 | 3.50 [0.00, 7.00] |
| Pose OKS vs original, all 17 kpts (x100) | 400 | 3.64 [2.89, 4.42] |
| Pose OKS vs original, 5 head kpts (x100) | 400 | 5.42 [4.00, 6.85] |
| Person recall vs original (%) | 400 | 0.44 [-0.19, 1.11] |
| Person AP50 vs MOT17 GT (x100) | 21 | -0.00 [-0.08, 0.09] |
| Person recall@0.5 vs GT (%) | 21 | -0.05 [-0.15, 0.03] |
| Person count MAE / frame (lower=better) | 21 | 0.01 [-0.02, 0.04] |
| Action top-1, VideoMAE-B (%) [face>=96px subset] | 161 | 4.35 [-0.62, 9.32] |
| Action top-1, X-CLIP B/32 (%) [face>=96px subset] | 161 | -6.83 [-13.04, -1.24] |

## 3. Pipeline diagnostics

| Arm | Dataset | face tracks | tracks given a surrogate | tracks blur fallback | median fps (audit) |
|---|---|---|---|---|---|
| reencode | k400 | 0 | 0 | 0 | 35.5 |
| reencode | mot17 | 0 | 0 | 0 | 18.4 |
| reencode | vox2 | 0 | 0 | 0 | 44.5 |
| redact_blur | k400 | 5076 | 0 | 0 | 23.9 |
| redact_blur | mot17 | 703 | 0 | 0 | 10.4 |
| redact_blur | vox2 | 78 | 0 | 0 | 7.2 |
| redact_box | k400 | 5076 | 0 | 0 | 35.5 |
| redact_box | mot17 | 703 | 0 | 0 | 12.4 |
| redact_box | vox2 | 78 | 0 | 0 | 36.9 |
| transform | k400 | 5076 | 746 | 4400 | 12.5 |
| transform | mot17 | 703 | 20 | 694 | 8.3 |
| transform | vox2 | 78 | 61 | 17 | 10.4 |

| Arm | runs | failed | total wall (min) | median wall / clip (s) | frames |
|---|---|---|---|---|---|
| reencode | 481 | 0 | 55.6 | 5.4 | 119174 |
| redact_blur | 481 | 0 | 216.0 | 14.7 | 119174 |
| redact_box | 481 | 0 | 62.3 | 5.8 | 119174 |
| transform | 481 | 0 | 187.9 | 20.0 | 119174 |
