# Video Guardian benchmark (`custodian_video/eval/`)

This is the video counterpart of the text paper's experiment. It asks whether replacing a face with a consistent synthetic surrogate (`transform`) removes identity while keeping downstream video models working, and whether redaction (blur or box) destroys that utility.

It implements the evaluation protocol proposed in `docs/video_masking_survey.md` §3 for the face channel. Everything is scripted. Each run is keyed by a **version tag** so that a new pipeline version can be benchmarked with one command and compared with an earlier one.

## One command

```bash
# v0.1 = the committed prototype (git 4dd3004, custodian_video 0.1.0)
CUDA_VISIBLE_DEVICES=7 WORKERS=5 custodian_video/eval/run_all.sh v0.1 git:4dd3004

# a later version: snapshot the current working tree, and compare with v0.1
CUDA_VISIBLE_DEVICES=7 custodian_video/eval/run_all.sh v0.2 worktree v0.1
# ...or pin a commit:  run_all.sh v0.2 git:<sha> v0.1
```

The result is written to `results/<tag>/summary.md`, with machine-readable copies in `summary.json` and `per_clip/<arm>/<clip>.json`. Face embeddings go to `data/video_eval/embeddings/<tag>/`, outside git.

Behaviour of the runner:
- **Resumable.** Every stage is idempotent. Re-running a tag only fills in missing outputs and measurements.
- **GPUs.** `CUDA_VISIBLE_DEVICES` is required, and GPU 2 is refused.
- **Pinned code.** The pipeline package is snapshotted once per tag into `data/video_eval/pipeline_snapshots/<tag>/`, from a git revision or the working tree. It is run from that snapshot through its public CLI (`python -m custodian_video …`). Concurrent edits to `custodian_video/custodian_video/` therefore cannot leak into a tag halfway through a run. `results/<tag>/run_info.json` records the snapshot's source, commit, file hash and reported `__version__`.
- **Original-video cache.** Features of the original videos do not depend on the pipeline version. They are cached in `data/video_eval/cache/`, so a new tag only re-measures the processed arms.

### Environments

| Env | Used for |
|---|---|
| `/data/qbao775/miniconda3/envs/custodian-video` | The pipeline under test. It is untouched. |
| `/data/qbao775/miniconda3/envs/custodian-video-eval` | Evaluation: torch 2.5.1/cu121, torchvision 0.20.1, transformers 4.46.3, onnxruntime-gpu 1.22, opencv 5. |

To recreate the evaluation env:

```bash
conda create -p /data/qbao775/miniconda3/envs/custodian-video-eval python=3.11 -y
E=/data/qbao775/miniconda3/envs/custodian-video-eval/bin/python
$E -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
$E -m pip install transformers==4.46.3 onnxruntime-gpu==1.22.0 opencv-python-headless av scipy pandas \
    "huggingface_hub<1" pillow "numpy<2.3" tabulate imageio-ffmpeg
```

### Stages

| Script | What it does |
|---|---|
| `download.py` | Fetches raw data and evaluation weights into `data/video_eval/` (git-ignored). |
| `prepare.py` | Deterministic clip selection (seed 0). Writes the mp4 clips and the committed clip list `manifest.json`. |
| `run_arms.py` | Runs every clip × arm through the pipeline CLI, using `--channels faces`. |
| `metrics.py` | Per-clip privacy and utility measurements. |
| `analyze.py` | Paired bootstrap, TOST, and `summary.md`. |

## Runtime (v0.1, one A100 shared with other jobs, 5 workers pinned to 8 cores each)

| Stage | Time |
|---|---|
| Pipeline runs | 724 clip × arm runs, about 4 worker-hours, about 1 h wall. Transform has a median of 18 s per clip. |
| Metrics | About 30 min for the originals (cached once), plus about 25 min per tag for all arms |
| Analysis | Under 1 min |

The data download and preparation, done once, took about 40 min. Most of that was MOT17 range reads.

## Data (181 evaluated clips, plus 118 gallery clips)

| Set | Clips | Use | Source / licence |
|---|---|---|---|
| **Kinetics-400, validation** | 100 clips, 100 distinct classes, ≤10 s, ≤720p | Action recognition, pose | [DeepMind Kinetics](https://github.com/google-deepmind/kinetics-i3d) val shard 0 (1,000 clips) from the CVDF tarball mirror `kiyoonkim/kinetics-400-targz` on HF. Annotations are **CC BY 4.0**. The videos are YouTube content owned by their uploaders (only 1/100 is flagged CC) and are used for **non-commercial research evaluation only**. Selection: YuNet finds a face (score ≥0.8, ≥40 px) in ≥50% of 8 sampled frames (246/1,000 pass). One clip per class is taken first, then the list is filled up to 100. |
| **MOT17, train** | 21 windows of 150 frames (3 per sequence × 7 sequences), mostly 1080p | Person detection AP50, recall, count | [MOTChallenge](https://motchallenge.net/data/MOT17/), **CC BY-NC-SA 3.0**. Only the needed frames are read from the official zip with HTTP range requests. The ground truth is the MOT17 `gt.txt`. |
| **VoxCeleb2, test** | 60 probe clips (6 s, 224×224), each from one identity. A gallery of 118 identities uses a **different YouTube video** of each person. | Cross-video face identification (privacy). Also face-present utility. | [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) via the `ProgramComputer/voxceleb` HF mirror. The dataset is **CC BY 4.0**, and copyright of the videos stays with the owners. **Research use.** |

Clip identities and sources are listed in `manifest.json`. Raw data and outputs live under `data/video_eval/`, which is git-ignored.

## Arms

| Arm | CLI args (always with `--channels faces`) |
|---|---|
| `original` | Not processed |
| `reencode` | `--mode redact --face-threshold 1.01`. This is the **null arm**: the same decode → pipeline → libx264 path with nothing masked. It measures the codec-noise floor of every utility metric. |
| `redact_blur` | `--mode redact --face-style blur` (elliptical mosaic + heavy blur, the default redaction) |
| `redact_box` | `--mode redact --face-style box` (the "black box" baseline) |
| `transform` | `--mode transform`. v0.1 uses one SD-1.5 img2img surrogate per track. Faces under 64 px or with score under 0.6 fall back to blur. |

Only the face channel is evaluated. The plates, on-screen text and audio channels are switched off so that the measurement isolates face de-identification.

## Metrics

### Privacy (lower is better)

Privacy is measured per **reference face**: YuNet (score ≥0.7, min side ≥24 px) run on the *original* frame. 16 uniformly spaced frames are used per clip, or every 10th frame for MOT17.

| Metric | Definition |
|---|---|
| **Re-ID (same location)** | ArcFace-R50 cosine between the original face and the *output* pixels aligned with the original's landmarks, counted against a threshold at **FAR = 1e-3**. The threshold is calibrated on about 1.8 M impostor pairs from the VoxCeleb2 gallery. This is a worst-case attacker who knows exactly where the face was. The same metric is also computed with SFace at OpenCV's 0.363 threshold. |
| **Leak** | YuNet still detects a face overlapping the original (IoU ≥0.3) **and** its ArcFace embedding matches the original identity. This is the "unmasked original face" rate. |
| **Mask coverage** | The audit log contains a face mask box containing the reference face's centre. This is detection recall, the HIPAA-critical number. |
| **Cross-video identification (VoxCeleb2)** | Rank-1 identification of each processed probe clip against the 118-identity gallery built from *other* videos (chance 0.85%). Two attackers are used: (a) same-location and (b) detector, where YuNet must find the face itself. Reported per clip (16-frame mean embedding) and per frame. |

The face recognisers are ArcFace R50 and SFace:
- **ArcFace R50** is InsightFace `buffalo_l` `w600k_r50.onnx`, trained on WebFace600K. Its weights are **non-commercial research only**, and it is used here only as an evaluation attacker.
- **SFace** comes from OpenCV-zoo and is Apache-2.0.

The detector is YuNet from OpenCV-zoo, which is MIT-licensed.

### Utility (higher is better)

Each metric is computed on the original and on every arm with the *same* model. Utility is the paired difference.

| Data | Metric | Model (licence) |
|---|---|---|
| Kinetics-400 | Action top-1 / top-5 vs label, 16 frames | VideoMAE-B fine-tuned on K400, `MCG-NJU/videomae-base-finetuned-kinetics` (CC BY-NC 4.0) |
| Kinetics-400 | Action top-1 / top-5, 8 frames | X-CLIP B/32, `microsoft/xclip-base-patch32` (MIT) |
| Kinetics-400 | Pose agreement: OKS of output keypoints vs the original's, persons matched by IoU ≥0.5 (unmatched = 0). Computed for all 17 keypoints and for the 5 head keypoints. | torchvision Keypoint R-CNN R50-FPN, COCO (BSD-3 code, COCO-trained weights) |
| MOT17 | Person AP50 vs GT (class 1, consider = 1, visibility ≥0.25; other boxes are ignore regions), recall@0.5 and count MAE per frame, at score ≥0.5, every 5th frame | torchvision Faster R-CNN R50-FPN v2, COCO |
| All | **Face present**: a face detector still finds a face where the original face was | YuNet |

### Statistics (mirroring the text paper)

- Paired per-clip deltas (arm − original), in points.
- 95% paired-bootstrap CI, with 10,000 resamples over clips.
- **TOST equivalence** at a ±2-point margin: two one-sided z-tests on the mean paired difference. The arm is equivalent when p < 0.05, which coincides with the 90% CI lying inside ±2.
- A transform − redact_blur paired contrast.
- Privacy rates get 95% clip-bootstrap CIs.

## v0.1 headline (full table in `results/v0.1/summary.md`)

| | redact_blur | redact_box | transform |
|---|---|---|---|
| Same-location re-ID (ArcFace, FAR 1e-3) / impostor floor | 14.5 / 8.2 % | 9.9 / 6.9 % | 12.2 / 5.4 % |
| Closed-set rank-1 of 100 (chance 1%) | 6.9 % | 6.6 % | 6.8 % |
| VoxCeleb2 cross-video rank-1 (118 ids) | 6.7 % | 5.0 % | 1.7 % |
| Face still present | 6.5 % | 1.7 % | 55.1 % |
| Δ action top-1, VideoMAE / X-CLIP (null arm: 0 / +1) | −12 / −7 | −6 / −3 | −7 / −11 |
| Δ pose OKS ×100, all keypoints / head keypoints (null arm: −16.7 / −15.1) | −41.1 / −59.2 | −44.4 / −63.8 | −37.4 / −53.3 |
| Δ MOT17 person AP50 | −0.0 (EQ) | −0.1 (EQ) | −0.0 (EQ) |

Two features of the v0.1 run limit what it can show:
- **Most faces never received a surrogate.** In v0.1, `transform` gave a surrogate to only 203/910 Kinetics face tracks and 20/703 MOT17 tracks. The rest fell back to blur, because of the <64 px / score <0.6 gate. This run is therefore mostly a test of the gate, not of surrogate quality.
- **The ±2-point TOST is underpowered for binary outcomes at this sample size.** With n = 100 clips, even the null arm is not shown equivalent on action top-1. To claim equivalence at ±2 points, the Kinetics set has to grow, to about 400 or more clips.

## Caveats

These limitations are inherent to the design:
- **Faces only.** The plates, text and audio channels are not exercised.
- **The same-location attacker is deliberately strong.** It upper-bounds what an attacker can recover from the pixels where the face was.
- **Kinetics labels mostly do not depend on identity,** so a face-centric subset (eating, laughing, applying make-up and similar classes) is where masking could matter most. Per-class breakdowns are in `per_clip/`.
- **MOT17 faces are mostly small.** Below 64 px, the v0.1 transform falls back to blur, so on MOT17 `transform` ≈ `redact_blur` by construction. The fallback counts per dataset are in §3 of the summary.
- **The VideoMAE checkpoint is CC BY-NC**, and the ArcFace weights are non-commercial. Both are used for evaluation only and never ship in the product.
