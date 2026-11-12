# Face-channel experiments (`face_d` – `face_i`)

These experiments develop and evaluate the surrogate-face options of `custodian_video` on the six openly licensed demo clips described in the main README. Each folder holds the scripts used, the result tables (`summary.json` / `summary.md`, `out/*.json`, `out/*.md`) and short notes.

| Folder | Question |
|---|---|
| `face_d` | Static vs re-enacted surrogate faces: motion fidelity (mouth, blink, yaw), identity leakage, seams and speed. |
| `face_e` | Surrogate realism v1 vs v2 (SD keyframe surrogate). Also checks coverage thresholds, the small-face floor and boundary seams, and writes a privacy report. |
| `face_f` | v2 vs v3 parsing-aware paste (face/hair segmentation), including a VLM preference check. |
| `face_g` | v2 vs the research-only synthetic-identity face swap (`--face-backend research-swap`): re-identification, motion, page stability and impostor floors. |
| `face_h` | False-positive face filtering and identity matching for the research swap; final privacy tables (`out/face_privacy_*.json`). |
| `face_i` | Why the research swap fell back to blur on the low-resolution meeting clip; `restore_small_faces.patch` is an experimental change that is **not** merged. |

## Metrics

- **Re-identification (SFace, ArcFace)**: whether a masked face can still be matched to the person's reference faces. ArcFace is used for evaluation only.
- **Coverage**: the share of reference faces that are masked.
- **Impostor floor**: the match rate between different people, which sets the baseline for re-identification.
- **Motion fidelity**: correlation of mouth opening, blinking and head yaw between the original and the surrogate.
- **Seams**: colour difference (ΔE) along the paste boundary.
- **Flicker**: frame-to-frame instability relative to the original.

## Not included

These files are left out of the repository:

- face crops, contact sheets and rendered clips (`media/`, `*.jpg`, `*.mp4`);
- per-track face data (`out/*_tracks.pkl`);
- run logs.

The crops show the people in the demo footage, and the track files contain face landmarks, so they are kept out of a public repository.

Everything can be regenerated from the demo inputs with the scripts in each folder. The MediaPipe segmentation model used by `face_f` (`selfie_multiclass_256x256.tflite`) is downloaded separately from Google's MediaPipe model hub.

## Running

The shell scripts read their interpreters and paths from environment variables:

| Variable | Used for |
|---|---|
| `PY` | Python with `custodian_video` installed |
| `CUSTODIAN_VLM_PYTHON` | Python with the VLM dependencies |
| `CUSTODIAN_MP_PYTHON` | Python with MediaPipe |
| `SCRATCH` | Temporary output directory |
| `VIDEO_DEMO_DIR` | The demo inputs |

Run the scripts from `custodian_video/`.

The research swap uses models with research-only or non-commercial licences; see the main README.
