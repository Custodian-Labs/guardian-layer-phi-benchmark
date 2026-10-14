# Custodian Video Guardian: survey, architecture and prototype results

*Local, private, LLM/VLM-assisted de-identification of video. Status as of 2026-09-26.*

**Scope and method.**
- **Survey (§1).** Licences, maintenance status and metrics were checked against primary sources: repo LICENSE files and the GitHub API, Hugging Face model cards, and the papers. Anything not confirmed from a primary source is marked **UNVERIFIED**. Star counts are approximate.
- **Prototype (§4–§5).** `custodian_video/` in this repo, run on three Creative Commons clips.
- **Note on the text paper's thesis.** The text paper argues that surrogates keep data useful where redaction does not. §3 shows how that maps to video, and §5 gives a first small-scale confirmation.

---

## 0. Summary

1. **No open project covers face, on-screen text and audio together, and no local product ships realistic surrogates.**
   - The open tools (deface, EgoBlur, DashcamCleaner) mask faces and/or plates with blur.
   - The commercial on-prem tools (Sighthound, CaseGuard, Gallio, Celantur, Facit) blur, and some also bleep audio.
   - Only **brighter AI** sells surrogate ("Deep Natural Anonymization") faces and plates, and it does not advertise audio or on-screen text.
   - A local, multi-channel, *transform-by-default* guardian is therefore an open position.
2. **A commercially usable local stack exists for detection, OCR, ASR and PII:**
   - EgoBlur Gen2 (Apache-2.0 weights) for faces and plates
   - supervision/ByteTrack (MIT) and SAM 2.1 (Apache-2.0) for tracking and masks
   - PaddleOCR / RapidOCR (Apache-2.0) for OCR
   - faster-whisper / WhisperX (MIT / BSD-2) for ASR
   - Qwen3-VL / Qwen2.5-VL-7B / Gemma 4 (Apache-2.0) as the scene VLM
   - Presidio (MIT) with GLiNER-PII (Apache-2.0) or a local LLM for PII
3. **The weak link is the surrogate face.** Every well-known realistic anonymiser has non-commercial weights or copyleft code:
   - DeepPrivacy2 checkpoints are FDF/FDH, which is CC BY-NC(-SA).
   - InsightFace's swapper and detectors are non-commercial.
   - SimSwap is CC BY-NC.
   - FAMS and NullFace are AGPL.
   - StyleGAN-based methods carry the NVIDIA non-commercial licence.

   This is where the proprietary moat is: a surrogate generator trained on licensed or consented faces.
4. **Prototype.** It runs fully locally, handles all four channels, and has both `redact` and `transform` modes plus a JSON audit log.
   - **Clip1, synthetic PII over real interview footage:**
     - On-screen PII: 3/3 values caught and 0/3 still readable afterwards; benign lines 100% intact.
     - Spoken PII: all 3 values caught; 0/7 ground-truth tokens remain in a re-transcript.
     - Faces: 0% re-identification of masked faces in either mode.
   - **The text-paper effect reproduces on video:**
     - Surrogate faces are still detected as faces **97.9%** of the time, versus **3.1%** after blur.
     - Surrogate on-screen text and speech are re-detected as the same PII type in 3/3 and 3/3 cases (text: NAME, PHONE, ID).
     - Redaction destroys both.
   - **Throughput:** 3–8 fps end-to-end at 1080p on a heavily shared A100. OCR and LLM calls dominate. Faces-only runs at about 18–26 fps.

---

## 1. Survey of open-source components (2023–2026)

**Legend:**
- **Commercial?** Yes = permissive code and weights. Cond. = allowed under use or scale conditions. NC = non-commercial. Copyleft = GPL/AGPL obligations.
- **Local** = runs offline once the weights are downloaded.

### 1.1 Face / head and licence-plate detection

| Name | Code / weights licence | Commercial? | What it masks | Speed / quality (reported) | Maintenance | Local deploy |
|---|---|---|---|---|---|---|
| **Meta EgoBlur Gen1 + Gen2** — [repo](https://github.com/facebookresearch/EgoBlur), [site](https://www.projectaria.com/tools/egoblur/), [paper](https://arxiv.org/abs/2308.13093) | Apache-2.0 / **Apache-2.0 weights** ("research and commercial") | **Yes**. Faces use RetinaFace-teacher pseudo-labels; plates use 200k in-house labels. | Faces and licence plates (Faster R-CNN X-101-FPN, about 104M params) | Faces on CCV2: AR 0.998 / AP 0.99. Plates: AP 0.929 / AR 0.992. Consistent across demographic groups. No published fps; heavy. | About 285★; v2.0.1 (2026-04) | `pip install egoblur`, TorchScript, GPU recommended |
| **deface** (ORB-HD) — [repo](https://github.com/ORB-HD/deface) | MIT / CenterFace ONNX (MIT) | Code yes. **Data caveat:** CenterFace is benchmarked on WIDER FACE (CC BY-NC-ND); training set **UNVERIFIED**. | Faces: blur, box or mosaic | CenterFace: 8.7 ms at 1080p on a 2080Ti; WIDER easy/medium/hard 93.5 / 92.4 / 87.5 | About 1.6k★; last release 2023-10 (low activity) | pip; ONNX Runtime CPU/CUDA/OpenVINO |
| OpenCV **YuNet** — [zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT / MIT | **Data caveat:** trained on WIDER FACE (non-commercial) | Faces and 5 landmarks | Millisecond-level; WIDER AP 0.834 / 0.824 / 0.708 | Active | Built into `cv2.FaceDetectorYN` |
| MediaPipe Face Detector (BlazeFace) — [docs](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector) | Apache-2.0 / weights licence **UNVERIFIED** | Probably | Faces; weak on small or distant faces | 2.9 ms CPU on a Pixel 6; 0.39 AR on EgoBlur's grayscale Aria footage | Very active | pip, TFLite |
| RetinaFace (serengil) — [repo](https://github.com/serengil/retinaface) | MIT / InsightFace weights | **NC** (weights) | Faces | WIDER hard about 64 (single scale) | Active | pip |
| **SCRFD / InsightFace** — [repo](https://github.com/deepinsight/insightface) | MIT / **"non-commercial research only"** | **NC** | Faces and 5 keypoints | 10GF: 4.9 ms, WIDER hard 83.05 | About 30k★ | ONNX |
| YOLOv5-face / yolov8-face / yolo-face — [1](https://github.com/deepcam-cn/yolov5-face) [2](https://github.com/derronqi/yolov8-face) [3](https://github.com/akanametov/yolo-face) | **GPL-3.0** (and Ultralytics **AGPL-3.0**) / WIDER-trained | **Copyleft and NC data** | Faces | YOLOv5m-face 95.3 / 93.8 / 85.3 | 2024–2026 | ONNX |
| Ultralytics YOLO — [repo](https://github.com/ultralytics/ultralytics) | **AGPL-3.0** (enterprise licence sold) | **Copyleft** | Generic detector | — | About 62k★ | pip |
| **open-image-models** (plates) — [repo](https://github.com/ankandrew/open-image-models) | MIT / MIT; training data **UNVERIFIED** | Yes, as far as stated | Plates (YOLOv9 ONNX) plus RF-DETR COCO | yolo-v9-s-608: P 0.957, R 0.917, mAP50 0.966 | Pushed 2026-07 | pip, ONNX |
| fast-alpr — [repo](https://github.com/ankandrew/fast-alpr) | MIT / MIT | Yes | Plate detection and OCR | — | v0.4.0 (2026-03) | pip |
| understand-ai/anonymizer — [repo](https://github.com/understand-ai/anonymizer) | Apache-2.0 / in-house weights | Yes | Faces and plates | — | **Archived** (TF1) | — |
| RF-DETR, D-FINE, YOLOX, YOLOv9-MIT (base models to fine-tune) — [RF-DETR](https://github.com/roboflow/rf-detr), [D-FINE](https://github.com/Peterande/D-FINE), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [YOLO-MIT](https://github.com/MultimediaTechLab/YOLO) | Apache / Apache / Apache / MIT | Yes (COCO pretraining) | Train on your own face/plate labels | — | Active | ONNX |

### 1.2 Realistic surrogates ("transform"): full body, face, face swap

| Name | Code / weights | Commercial? | Keeps pose / expression? | Speed / quality | Maintenance |
|---|---|---|---|---|---|
| **DeepPrivacy2** — [repo](https://github.com/hukkelas/deep_privacy2), [paper](https://arxiv.org/abs/2211.09454) | Apache-2.0, **but** `sg3_torch_utils` is under the NVIDIA non-commercial licence. Checkpoints "follow the license of the datasets": [FDF](https://github.com/hukkelas/FDF) and [FDH](https://github.com/hukkelas/FDH), mostly CC BY-NC(-SA). | **NC** | Face (FDF) and full body (DensePose-guided); keeps pose and silhouette | Face FID 0.56 (FDF128); body FID 5.6. About 8–12 fps generator on a GTX 1080; about 2.8 s for an image with 12 people. | Stale since 2024-01 |
| DeepPrivacy v1 — [repo](https://github.com/hukkelas/DeepPrivacy) | MIT / FDF | **NC** | 7-keypoint face inpainting | — | Stale |
| CIAGAN — [repo](https://github.com/dvl-tum/ciagan) | MIT / weights **UNVERIFIED** | ? | Landmark-conditioned identity swap | — | Stale since 2021 |
| FALCO (CVPR'23) — [repo](https://github.com/chi0tzp/FALCO) | Apache / needs StyleGAN2 (NVIDIA NC) | **NC** | Attribute-preserving latent optimisation (slow) | — | Stale |
| RiDDLE (CVPR'23) — [repo](https://github.com/DongzeLi-CASIA/RiDDLE) | **No licence** / StyleGAN2 | **No** | Reversible, keyed | — | — |
| **FAMS**, Face Anonymization Made Simple (WACV'25) — [repo](https://github.com/hanweikung/face_anon_simple), [paper](https://arxiv.org/abs/2411.00762) | **AGPL-3.0** / HF weights have **no licence**; built on SD-2.1 | **Copyleft / unclear** | Keeps expression, pose and gaze | State of the art on CelebA-HQ, **but** in a video study its identity similarity stayed at 0.598 (weak anonymisation) | Active |
| NullFace (FG'26) — [repo](https://github.com/hanweikung/nullface) | **AGPL-3.0** | Copyleft | Training-free, localised, keeps attributes | Slow (diffusion inversion) | New |
| AnonNET (ICCVW'25) — [repo](https://github.com/anilegin/AnonNET), [paper](https://arxiv.org/abs/2601.11635) | MIT / uses LivePortrait, whose InsightFace parts are **NC** | Needs component swaps | **Talking-head video** anonymisation with motion re-synthesis; keeps age, gender, race, pose and expression | VoxCeleb2, CelebV-HQ, HDTF | Small |
| InsightFace inswapper — [repo](https://github.com/deepinsight/insightface) | MIT / **NC** | **NC** | Face swap | — | Active |
| SimSwap — [repo](https://github.com/neuralchen/SimSwap) | **CC BY-NC 4.0** | **NC** | Swap with the best temporal consistency in a 2025 video study (SSIM 0.920) | — | Stale |
| FaceFusion / roop / ReActor — [FaceFusion](https://github.com/facefusion/facefusion), [roop](https://github.com/s0md3v/roop), [ReActor](https://github.com/Gourieff/ComfyUI-ReActor) | OpenRAIL-AS / AGPL / GPL, all bundling inswapper (NC) | **No** | Swap | — | roop archived |

### 1.3 Segmentation and tracking (temporal consistency)

| Name | Licence | Notes |
|---|---|---|
| **SAM 2 / 2.1** — [repo](https://github.com/facebookresearch/sam2) | Apache-2.0 (code and weights) | Promptable video masks. A100: tiny 91 fps, large 39.5 fps. SA-V J&F 76.5–79.5. |
| SAM 3 / 3.1 — [repo](https://github.com/facebookresearch/sam3) | "SAM License": royalty-free, but bans ITAR/military uses and terminates if you bring IP litigation; gated | Needs legal review; text-prompted concept tracking |
| EfficientTAM — [repo](https://github.com/yformer/EfficientTAM) | Apache-2.0 | Lightweight SAM 2 (over 10 fps on an iPhone 15) |
| ByteTrack / OC-SORT / BoT-SORT | MIT | Use your own detector. The bundled YOLOX-on-CrowdHuman weights may be NC (**UNVERIFIED**). |
| **Roboflow supervision** — [repo](https://github.com/roboflow/supervision) | MIT | ByteTrack, blur and pixelate annotators, video I/O |
| Norfair — [repo](https://github.com/tryolabs/norfair) | BSD-3 | Lightweight tracker |
| BoxMOT — [repo](https://github.com/mikel-brostrom/boxmot) | **AGPL-3.0** | Trap |

### 1.4 On-screen text: OCR and PII detection

| Name | Licence | Notes |
|---|---|---|
| **PaddleOCR 3.x** (PP-OCRv5; PP-OCRv6 released 2026-06) — [repo](https://github.com/PaddlePaddle/PaddleOCR) | Apache-2.0 code and weights | v6 tiny/small/medium; README claims +4.6 det / +5.1 rec over v5 |
| **RapidOCR** — [repo](https://github.com/RapidAI/RapidOCR) | Apache-2.0 | PaddleOCR models as ONNX; pip; word boxes. **Used in the prototype.** |
| EasyOCR, docTR, Tesseract | Apache-2.0 | Alternatives |
| **Presidio** (analyzer, anonymizer, image-redactor) — [repo](https://github.com/data-privacy-stack/presidio) | MIT | Moved from Microsoft to the "Data Privacy Stack" org in 2026. Its image redactor handles OCR-based redaction of images and DICOM. |
| GLiNER / GLiNER-PII — [repo](https://github.com/urchade/GLiNER) | Apache-2.0 (urchade, knowledgator). nvidia/gliner-PII is under the NVIDIA Open Model License. | Zero-shot NER; ONNX |
| OpenAI privacy-filter — [HF](https://huggingface.co/openai/privacy-filter) | Apache-2.0 (per model card) | 1.5B-total / 50M-active token classifier for on-prem use |
| Piiranha — [HF](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information) | **CC BY-NC-ND** | Trap |
| Local LLMs (Qwen2.5/3, Llama 3.x, Gemma 4) | Apache-2.0 (Qwen, Gemma 4) or Llama licence | Same approach as the text benchmark. **Used in the prototype:** qwen2.5:7b via Ollama. |

### 1.5 Audio

| Name | Licence | Notes |
|---|---|---|
| Whisper / **faster-whisper** — [repo](https://github.com/SYSTRAN/faster-whisper) | MIT (code and weights) | Word timestamps. **Used in the prototype.** |
| WhisperX — [repo](https://github.com/m-bain/whisperX) | BSD-2 | Forced-alignment word timing (60–70× real time). Per-language wav2vec2 alignment licences **UNVERIFIED**. |
| pyannote.audio — [repo](https://github.com/pyannote/pyannote-audio) | MIT code. 3.1 models are MIT but gated; community-1 is CC-BY-4.0, gated, and works offline. | Diarization, needed for per-speaker policy |
| VoicePrivacy 2024 baselines — [repo](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024), [plan](https://arxiv.org/abs/2404.02677) | **GPL-3.0** | EER / WER / UAR protocol. B5 (ASR-BN+VQ) scores EER 34.3 / WER 4.4. Use the protocol, not the code. |
| **kNN-VC** — [repo](https://github.com/bshall/knn-vc) | MIT (with a jokey clause; GitHub reports NOASSERTION); WavLM encoder terms **UNVERIFIED** | Simple any-to-any voice conversion to a fixed or pooled synthetic target voice |
| **OpenVoice V1/V2** — [repo](https://github.com/myshell-ai/OpenVoice) | MIT, "free for commercial use" since 2024-04 | Tone-colour conversion |
| Seed-VC | GPL-3.0, archived | Trap |
| StreamVoiceAnon | Apache code, but fine-tuned from fish-speech (**CC BY-NC-SA**) | Trap |
| Piper TTS (`piper-tts` ≥1.3, [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)) | **GPL-3.0** (confirmed from the installed package metadata); voices vary | Used only as an *optional* plug-in in the prototype. Replace it for a product. |

### 1.6 Open VLMs and open-vocabulary detectors (scene-level sensitive objects: screens, documents, badges, whiteboards, tattoos)

| Model | Licence | Commercial? |
|---|---|---|
| Qwen2.5-VL-7B / 32B | Apache-2.0 | Yes |
| Qwen2.5-VL-**3B** | Qwen Research License (NC) | **No** |
| Qwen2.5-VL-72B | Qwen licence (100M MAU cap) | Cond. |
| **Qwen3-VL** 2B / 8B / 30B-A3B / 235B; Qwen3.5 (native image-text) | Apache-2.0 | Yes |
| Gemma 3 / 3n, PaliGemma 2 | Gemma Terms (prohibited-use policy) | Cond. |
| **Gemma 4** (E2B–31B) | Apache-2.0 | Yes |
| **Florence-2** | MIT | Yes |
| **OWLv2**; Grounding DINO (original); Grounded-SAM-2 | Apache-2.0 | Yes |
| Grounding DINO 1.5 / DINO-X | API only | Not local |
| YOLO-World / YOLOE | GPL-3.0 / AGPL-3.0 | Copyleft |
| Moondream2 / Moondream3-preview | Apache-2.0 / BSL 1.1 | Yes / restricted |
| SmolVLM / SmolVLM2 | Apache-2.0 | Yes |

### 1.7 End-to-end open projects

- **deface** (MIT) and **EgoBlur CLI** (Apache): faces, and faces+plates respectively.
- **DashcamCleaner** (AGPL, Ultralytics): faces and plates at about 10 fps at 1080p on a GTX 1060.
- **understand-ai anonymizer**: archived.
- **Presidio image-redactor**: text in images only.
- AWS Rekognition samples: cloud only.

**No open project combines face, on-screen text and audio.** One close research analogue is **CAIAMAR** (CVPRW'26, [arXiv 2603.27817](https://arxiv.org/abs/2603.27817)). It is an on-prem agentic VLM plus diffusion anonymiser that produces GDPR audit trails. On CUHK03-NP it reduced person re-ID R1 from 62.4% to 16.9%. Its code link is **UNVERIFIED**.

### 1.8 Commercial comparators (positioning)

| Vendor | Deployment | Surrogate or blur | Audio | On-screen text | Public price |
|---|---|---|---|---|---|
| **brighter AI**, Deep Natural Anonymization ([site](https://brighter.ai/product/)) | SaaS or **on-prem container** | **Surrogate** faces and plates | Not stated | Not stated | — |
| Celantur ([site](https://www.celantur.com/container/)) | Cloud, **on-prem, edge** | Blur (faces, persons, plates, vehicles) | No | No | Per image / per video-hour |
| Secure Redact / Pimloc ([site](https://www.secureredact.ai/pricing)) | SaaS, API, **on-prem** | Blur | **Yes** | — | Credits (1 credit = 1 video-minute) |
| Veritone Redact ([site](https://www.veritone.com/applications/redact/)) | Cloud | Blur | Yes | — | About $600/month (**UNVERIFIED**) |
| Axon Redaction Assistant ([site](https://www.axon.com/products/redaction-assistant)) | Axon cloud | Masks heads, plates, **screens** | — | Screens | Sales |
| Sighthound Redactor ([site](https://www.sighthound.com/products/redactor)) | **Desktop / on-prem** | Blur | Mute, beep or scramble | — | Sales |
| CaseGuard Studio ([pricing](https://caseguard.com/pricing/)) | Cloud, **on-prem, offline** | Blur (faces, plates, screens) | **Yes** (50+ languages) | Screens | $299–379 per user per month |
| Gallio PRO ([site](https://gallio.pro/)) | **100% on-prem** | Blur (faces, plates) | No | No | €89/month |
| Facit Identity Cloak ([site](https://facit.ai/)) | **On-prem** | Blur, including body and inverse blur, plus name tags and signage | Edits audio | Name tags | — |
| Azure Video Indexer / Google person-blur / AWS Rekognition | Cloud | Blur, faces only | No | No | Per job |

**Positioning.** Two things are both missing from the market:
- no **on-prem** vendor offers **surrogate** (not blur) masking across **faces, on-screen text and speech**;
- no vendor offers a per-detection audit log together with a measured "utility-preserved" claim.

That combination is Custodian's text thesis applied to video.

---

## 2. Recommended architecture: "Custodian Video Guardian"

```
                    ┌──────────── local box (GPU optional) ─────────────┐
 video ─► decode ─► │ DETECT (per frame / stride)                        │
                    │  faces+heads : EgoBlur Gen2 (Apache)  [fast tier:  │
                    │                RF-DETR/D-FINE fine-tuned in-house] │
                    │  plates      : EgoBlur-LP + open-image-models (MIT)│
                    │  text        : PP-OCRv5/v6 via RapidOCR (Apache)   │
                    │  scene VLM   : Qwen3-VL-8B / Gemma-4 (Apache)      │
                    │    - "is this a screen/document/badge/whiteboard?" │
                    │    - long-tail objects (tattoos, ID cards, mail)   │
                    │    + OWLv2/Florence-2 grounding for boxes          │
                    │ TRACK : ByteTrack via supervision (MIT) + offline  │
                    │         gap-fill/pad/smooth; SAM 2.1 (Apache) masks│
                    │ PII   : local LLM (Qwen) + GLiNER-PII/Presidio+regex│
                    │ AUDIO : faster-whisper/WhisperX (MIT/BSD) words →  │
                    │         PII spans; pyannote community-1 speakers   │
                    │ POLICY: per-channel redact | transform | keep       │
                    │ TRANSFORM:                                          │
                    │   faces : in-house surrogate generator (see §6),   │
                    │           one identity per track, pose/expression- │
                    │           conditioned                               │
                    │   text  : same-type surrogate, re-typeset in place │
                    │   speech: surrogate words via licensed TTS;        │
                    │           voice conversion (kNN-VC/OpenVoice, MIT) │
                    │ AUDIT : JSON per detection (hash, never raw PII),  │
                    │         model versions, input/output SHA-256       │
                    └────────────────────────────────────────────────────┘
```

Why each component was chosen:
- **Offline two-pass, not streaming, for the file product.** Privacy failures are *misses*, and a single unmasked frame leaks. Knowing the future lets us fill detector gaps, pad tracks and smooth boxes. The prototype shows this is cheap: tracking costs under 10 ms per clip. A streaming / edge variant can use a causal tracker with a lookahead buffer.
- **EgoBlur Gen2 as the default detector.** It is the only high-recall face and plate detector whose *weights* are Apache-2.0, and it was not trained on WIDER FACE. It is heavy, so a fast tier should be fine-tuned from Apache/MIT detectors on licensed or EgoBlur-pseudo-labelled data. The prototype uses CenterFace for speed and must switch before any commercial release.
- **Local LLM for text PII.** This reuses the text paper's span prompt, so detection quality is directly comparable with the paper. qwen2.5:7b got 3/3 correct in the prototype, running in about 1.7 s per new block of text with results cached per line. GLiNER-PII or Presidio should run alongside as a cheap first pass to reduce LLM calls.
- **VLM as a context gate.** The biggest precision problem in the prototype is street signage being flagged as names. A VLM question such as "is this text on a document, screen, badge or whiteboard, or is it public signage?" solves this, and the same VLM covers long-tail objects that detectors don't have classes for.

**Licensing traps to avoid in the product:**

| Trap | Where it appears |
|---|---|
| AGPL / GPL | Ultralytics (YOLOv8/11), YOLOE, BoxMOT, DashcamCleaner, yolo-face variants, YOLO-World, FAMS, NullFace, ReActor, Seed-VC, VoicePrivacy baseline code, **piper-tts ≥1.3** |
| Non-commercial weights | InsightFace (SCRFD, RetinaFace, ArcFace, inswapper) and everything that bundles it (LivePortrait, FaceFusion, roop); DeepPrivacy2 checkpoints (FDF/FDH); StyleGAN2/3 (NVIDIA); SimSwap; Piiranha; Qwen2.5-VL-3B; fish-speech derivatives |
| Non-commercial training data | WIDER FACE (CC BY-NC-ND) sits behind CenterFace, YuNet, RetinaFace, SCRFD and YOLO-face weights |
| Other conditions | SAM 3 licence (use bans, litigation termination); Gemma 3 terms; Qwen-72B MAU cap; Moondream3 (BSL); unlicensed HF weights (all rights reserved) |

---

## 3. Mapping the surrogate-vs-redact thesis to video

**The text paper's result.** Replacing each PHI span with a realistic same-type surrogate keeps documents **detectable and useful**: recall moved 76.1% → 74.9%, which is TOST-equivalent within ±2 points. `*****` redaction destroys that.

**The video analogue.**

| | Text (paper) | Video |
|---|---|---|
| Unit | PHI span | Face track, on-screen PII word span, spoken PII word span, plate track |
| Redact | `*****` | Blur / pixelate / box; bleep |
| Transform | Same-type surrogate, consistent per entity | Same-type **synthetic face per track**; fake value **re-typeset** in the same field; fake value **spoken** in the same slot. Consistent across frames *and across modalities*: the same name maps to the same surrogate on screen and in speech. |
| Privacy check | Original PHI no longer present | Face **re-identification rate** (ArcFace/AdaFace/SFace attacker; attacker knows face location), OCR and ASR leakage of the ground-truth PII, speaker EER for voice |
| Utility check | Downstream PHI detectors keep recall | Downstream **vision/audio models keep working**: face detection, person counting, pose/keypoints, action recognition, gaze/expression, MOT; ASR WER; PII detectors re-detect the surrogate |

### Proposed evaluation protocol (the "video Guardian benchmark")

1. **Data.**
   - Public, licence-clean sets with labels:
     - Kinetics / AVA / Charades for action
     - COCO-keypoints / PoseTrack for pose
     - MOT17/20 or CrowdHuman-like data for counting and tracking (check licences)
     - VoxCeleb2 / CelebV-HQ for talking heads
     - LibriSpeech / VoicePrivacy sets for audio
   - A **synthetic PII overlay set** built like `demo/make_demo_inputs.py`: screens, badges and whiteboards with known fake PII, plus TTS speech with known PII. This gives exact span-level ground truth.
2. **Arms.** Original · blur · pixelate · box · transform (surrogate) · optionally transform+voice-conversion.
3. **Privacy metrics** (lower is better):
   - Face re-ID. Metrics: (a) same-location verification rate at the FAR=1e-3 threshold; (b) rank-1 identification against a gallery built from the *original* video. Attackers: ArcFace/AdaFace, plus a VLM attacker.
   - Residual PII: OCR and ASR leakage of ground-truth strings.
   - Speaker-verification EER, using the VoicePrivacy protocol with a semi-informed attacker.
   - **Plus recall, the HIPAA-critical number:** the fraction of ground-truth face / PII frames that were masked.
4. **Utility metrics** (higher is better), evaluated as the paper does ("same detector, original vs de-identified"):
   - Face detection AP and person count MAE
   - Keypoint AP (COCO / PoseTrack)
   - Action-recognition top-1 (VideoMAE / X-CLIP)
   - MOT HOTA / IDF1
   - Expression / gaze agreement
   - ASR WER outside the PII spans
   - **PII re-detection recall of surrogates** (this is the direct analogue of the paper's headline)
   - Also train-on-anonymised / test-on-original, following [Hukkelås & Lindseth, CVPRW'23](https://arxiv.org/abs/2306.05135): on COCO keypoints, face blur cost −11.7 keypoint AP versus −4.6 for realistic faces.
5. **Temporal metrics:** identity consistency of the surrogate within a track (embedding cosine across frames), flicker (inter-frame SSIM inside the mask), and FVD. These follow [Assessing face swapping as anonymizers in video, 2025](https://arxiv.org/abs/2505.20985) and JaGAN's Identity-Invariance score.
6. **Statistics:** paired per-clip deltas with **TOST equivalence** margins (e.g. ±2 AP / ±2 pts accuracy), exactly as in the paper. Pair these with non-inferiority on the privacy side (re-ID ≤ impostor rate).

---

## 4. Prototype: `custodian_video/`

**What it contains:**
- A Python package with a CLI: `custodian-video in.mp4 -o out.mp4 --mode redact|transform`.
- A conda env at `/data/qbao775/miniconda3/envs/custodian-video`.
- Full details in `custodian_video/README.md`.

**Channels:**

| Channel | Components | redact | transform |
|---|---|---|---|
| Faces | CenterFace ONNX every frame → offline IoU/centre tracker → gap interpolation (≤0.6 s), ±3-frame padding, smoothing, 1.25× dilation | Mosaic + heavy blur (elliptical) | SD-1.5 img2img surrogate per track (strength 0.75, per-track seed). The surrogate is aligned with 5 landmarks, LAB colour-matched and feather-blended every frame. Faces under 64 px or with score under 0.6 get blur instead. |
| Plates | open-image-models YOLOv9 every 2nd frame, then tracking | Blur | Blur (no surrogate yet) |
| On-screen text | RapidOCR PP-OCRv5 every 0.5 s → qwen2.5:7b span prompt + regex → **word-level quads of only the PII characters**; text tracks interpolated, padded ±1 stride | Black box | Inpaint the background and **re-typeset a same-type surrogate** in the estimated ink colour, perspective-warped into the quad |
| Audio | faster-whisper small.en word timestamps → the same LLM + regex → word time ranges (±60 ms) | 1 kHz bleep | Spoken surrogate via local TTS, time-fitted to the slot and RMS-matched (Piper plug-in; bleep if no TTS) |
| Audit | JSON: per track/event id, channel, type, frame and time range, per-frame boxes tagged `det`/`interp`/`pad`, score, action, surrogate, PII **hash + masked preview**, model names and licences, input/output SHA-256, stage timings | ✓ | ✓ |

**Robustness on a shared host.**
- **CUDA out-of-memory.** Whisper and Stable Diffusion fall back to CPU, and detector sessions are freed before generation starts.
- **Ollama errors.** On a 5xx the request is retried, then falls back to regex-only detection, and the event is logged.
- **Face-surrogate backend.** DeepPrivacy2 is a documented hook (`DeepPrivacy2Backend`) and is not bundled, because of its NC checkpoints.

---

## 5. Prototype results

Setup: 1080p clips run on a **heavily shared** A100 (other jobs held the GPU at 99–100% utilisation, which adds noise to the timings). OCR ran on CPU.

- **Clip1** is 18 s from a CC BY 3.0 Royal Society interview. It contains two people and a camera cut. We added a synthetic "patient intake" screen with a fake name, phone number and account number, and replaced the audio with TTS reading those fake values.
- **Clip2** is a 9 s CC BY-SA Tokyo street scene with dozens of small faces.
- **Clip3** is a 12 s CC BY Ljubljana street with parked vans, their plates, and a few pedestrians.

Sources are listed in `custodian_video/demo/inputs/SOURCES.md`. Metrics are from `custodian_video/demo/evaluate.py`, and the JSON files are in `custodian_video/demo/outputs/*_eval.json`.

### 5.1 Clip1: every channel against ground truth

| Check | Original | redact | transform |
|---|---|---|---|
| On-screen GT PII strings still OCR-readable | 3/3 | **0/3** | **0/3** |
| Benign on-screen lines still readable | 100% | **100%** | **100%** |
| Surrogate on-screen values re-detected as PII by the same local LLM | — | — | **3/3** (NAME, PHONE, ID) |
| Spoken GT PII tokens in a Whisper re-transcript | 7/7 | **0/7** | **0/7** |
| Spoken surrogates re-detected as PII | — | — (bleeped) | **3/3** (NAME, PHONE, ID) |
| Face re-ID of masked faces (SFace cos > 0.363, same-location attacker; 91 faces ≥64 px) | — | **0.0%** (mean cos 0.157) | **0.0%** (mean cos 0.070) |
| **A face is still detected** by an independent detector (YuNet) | 100% | **3.1%** | **97.9%** |
| Output identity consistency within a track (cos across frames) | — | 0.86 (blob) | **0.92** |

The re-transcript of the transform output reads:

> "Hello, my name is **Samuel Tanaka**. I work in the flood research team. You can reach me on **890-894-7327**. My patient account number is **514-382-77**."

The on-screen field shows the same `Name: Samuel Tanaka`, because the surrogate is consistent across modalities. The local Qwen2.5-VL judge transcribed the transform panel as "Name: Samuel Tanaka / Phone: (890) 894-7327 / Acct #: 51438277". The benign lines were unchanged.

**This is the text paper's result reproduced on video, at small scale.**
- Both modes remove identity: 0% re-ID and 0/3 readable PII.
- Only `transform` keeps the content *detectable and useful*: faces are still found 97.9% of the time versus 3.1% after blur, and the surrogates are re-detected as PII 3/3 on screen and 3/3 in speech.

**SD img2img strength sweep** (clip1, faces only; from `demo/outputs/sweep/`). The re-ID column counts all 96 reference faces, including 5 that were never masked (see §5.2), which is why 5.2% is the floor.

| Strength | Mean cos to original (all refs) | Re-ID (all refs) | Still detected | Identity consistency |
|---|---|---|---|---|
| 0.45 | 0.384 | 38.5% (identity leaks) | 97.9% | 0.95 |
| 0.62 | 0.262 | 5.2% (= 0% of masked) | 97.9% | 0.94 |
| **0.75 (default)** | **0.117** | 5.2% (= 0% of masked) | 97.9% | 0.92 |
| 0.90 | 0.159 | 5.2% | 99.0% | 0.93 |

### 5.2 Face recall and precision (clips 1–3)

No human annotator was available, and the image viewer in this environment was down. Recall is therefore measured against an **independent reference detector** (YuNet, threshold 0.6, faces ≥16 px, every 5th frame). The misses were then adjudicated by a **local VLM judge** (Qwen2.5-VL-7B, forced multiple choice).

| Clip | YuNet reference faces | Covered by our masks | Misses as judged by the VLM |
|---|---|---|---|
| clip1 | 96 | **94.8%** (5 misses) | In the step-10 spot check (3 misses): 1 was torso/hands (a YuNet false positive), and 2 were a defocused region top-left in shot 2, which the VLM called a "blurred face" and CenterFace scores below 0.1. Ambiguous; flagged for human review. |
| clip2 | 319 | **91.2%** | Of 10 sampled misses: 0 clearly visible faces; 4 back-of-head, 2 dogs, 1 hand, 3 too blurry |
| clip3 | 9 | 44.4% (small n) | 3 trees/foliage, 2 too blurry |

**Precision** was checked on a VLM-judged sample of our detections:
- Clip2, 14 face tracks: 1 visible face, 9 back-of-head, 3 too blurry, 1 dog.
- Clip3, 13 face tracks: 10 back-of-head, 2 too blurry, 1 building or sign.
- Plates: 5 of 6 were plates, and 1 was too blurry to tell.

CenterFace also fires on heads seen from behind. That is acceptable for privacy, and it is why surrogates are gated on detection score and face size.

**Small faces.**
- **SFace is unreliable below 32 px.** The impostor baseline (original face versus a *different* person's original face) already reaches cos 0.18 with a 17% false-match rate.
- **The v1 blur (6× downscale + Gaussian) left small faces "re-identifiable"** under this metric: 68% for faces under 32 px, against 6% for a solid box.
- **The fix:** we switched to a coarse (≤4-cell) mosaic plus heavy blur. That brought faces under 32 px down to **7.2%**, which is at impostor level, and faces of 32–64 px to 13.5%.
- **Lesson:** light blur is not anonymisation, and the evaluation harness caught it.

### 5.3 On-screen text on real street footage (precision)

Clip3 produced 11 on-screen "PII" tracks and clip2 produced 2. All of them are **false positives**: shop signs read as NAME, and stray numbers read as ID or AGE. These were short (1–2 OCR samples) and conservative, but in `transform` they would re-letter a shop sign. The prompt, and an OCR rule that drops word-only "OTHER" spans, removed the worst cases. The real fix is the **VLM context gate** in §2.

### 5.4 Throughput

1080p. The GPU was shared at about 100% utilisation by other jobs, and OCR ran on CPU.

| Run | End-to-end fps | Pass 1 (detect) fps | Notes |
|---|---|---|---|
| clip1 redact / transform | 1.9 / 3.8 | 6.5 / 9.7 | OCR+LLM 32–55 s of about 47–69 s in pass 1. SD surrogates took 6.8 s for 2 tracks. |
| clip2 redact / transform | 8.4 / 5.6 | 12.5 / 8.0 | 45 face tracks |
| clip3 redact / transform | 7.6 / 4.6 | 9.9 / 6.0 | 6 plate tracks |
| Faces only (clip2, blur / box / pixelate) | **18–26** | — | CenterFace alone takes about 20 ms per 1080p frame |
| LLM | — | — | qwen2.5:7b: 5 calls, 1.6 s in total, for clip1 (per-line cache) |

**Where the time goes:** OCR (CPU) plus the LLM, then rendering (1080p rawvideo pipe → libx264), then detection. The obvious speed-ups:
- OCR on GPU / TensorRT
- text-region gating (only OCR regions a VLM or detector says are screens or documents)
- NVDEC/NVENC for decoding and encoding
- batching the detectors

---

## 6. Limitations (honest list)

1. **The surrogate face is a static texture per track.** Mouth and eye motion are not reproduced, and large yaw distorts the warp. The Qwen-VL judge rated the talking-head surrogates as natural and artifact-free. On small street faces it saw misaligned eyes and mouths, and that motivated the ≥64 px / score ≥0.6 gate. Stable Diffusion 1.5 (CreativeML OpenRAIL-M) is a stop-gap.
2. **The evaluation is small and partly automated.** It covers 3 clips. The "hand-check" was done with an independent detector plus a VLM judge because no human rater was available. SFace is weak on tiny faces. A real study needs human-labelled face boxes, ArcFace/AdaFace attackers, and a human re-identification test.
3. **Detectors.** CenterFace's training data is likely WIDER-derived, so it is not for commercial use; switch to EgoBlur Gen2. There is no full-body, gait or clothing anonymisation (identity leaks through body and context). Plates are detected every 2nd frame.
4. **On-screen text.** OCR runs at 0.5 s intervals, mitigated by padding one stride backwards and forwards. Text precision is low on signage. There is no handwriting model. Rotated or curved text depends on OCR quality.
5. **Audio.** Only the PII words are bleeped or replaced; **the speaker's voice itself is not anonymised**, and voice conversion is on the roadmap. Piper is GPL-3.0. The surrogate TTS voice does not match the speaker, so it is audible as an insert.
6. **Scale.** Short clips only. Pass 1 keeps all detections in memory, which is fine for minutes of video but not hours. The next step is chunked streaming with overlap.

## 7. Roadmap

| Phase | Deliverable |
|---|---|
| **0–1 month** | Swap in EgoBlur Gen2 (Apache) plus supervision/ByteTrack. Add a GLiNER-PII/Presidio pre-filter and a VLM context gate (Qwen3-VL-8B or Gemma-4 E4B) for text regions. Move OCR to GPU. Chunked processing. Add a replacement for the GPL TTS. Package as a container with an offline model bundle. |
| **1–3 months** | Build the **video Guardian benchmark** from §3 (synthetic PII overlays + public pose/action/MOT sets, TOST analysis) and publish results as the video counterpart of the text paper. Add voice anonymisation (kNN-VC / OpenVoice, MIT) evaluated with the VoicePrivacy protocol. Add SAM 2.1 masks for full-head, hair and body. Add a human-review UI driven by the audit log. |
| **3–6 months** | **A proprietary surrogate generator, which is the moat.** It needs: a landmark- and expression-conditioned face generator trained on licensed or consented data (DeepPrivacy2-style architecture, re-implemented without NVIDIA or InsightFace code); one identity latent per track; temporal-consistency loss; distillation to real-time. Then extend to full-body and plate surrogates. |
| **6+ months** | Edge / streaming mode (causal tracker with lookahead buffer). A policy engine (per-person consent: keep, redact or transform). Reversible, keyed pseudonymisation for authorised re-identification. Certification-oriented audit (HIPAA Safe Harbor and GDPR Art. 25 mapping). |

---

## 8. v0.2 changes

v0.2 makes three changes. **Detection** moves to commercially licensed models: EgoBlur faces and plates replace CenterFace and YOLOv9, both of which stay available as flags. A **local VLM scene-context gate** now decides which flagged on-screen text gets masked. An **optional Custodian Guardian Layer backend** serves both text channels. Everything is still local by default (`--pii-backend local`).

Numbers come from `custodian_video/demo/evaluate.py` and `demo/compare_pii_backends.py`:
- v0.1: `demo/outputs/*_eval.json`
- v0.2: `demo/outputs/v02/`
- face-detector sweep, run under one machine load: `demo/outputs/v02_face_sweep/`
- PII backends: `demo/outputs/pii_backend_comparison.json`

The host was heavily shared, with a load average of 170–550 on 256 cores, so wall-clock fps is noisy. Per-detector milliseconds per frame are the fairer speed comparison.

### 8.1 Faces and plates: EgoBlur (Apache-2.0) as the default

**Weights and licence.**
- The weights are EgoBlur **Gen1** TorchScript (`ego_blur_face.jit`, `ego_blur_lp.jit`) from Meta's official repository [huggingface.co/projectaria/EgoBlur](https://huggingface.co/projectaria/EgoBlur), revision `9c0b319`.
- Licence **Apache-2.0 for code and weights**, re-verified on 2026-10-02. The repository is tagged `apache-2.0`, and its README says "open-source license (Apache 2.0) for both commercial and non-commercial use". The zip SHA-256 hashes match the repository's LFS pointers.
- **Gen2** weights are also Apache-2.0 per the projectaria.com FAQ. They are only distributed after an e-mail form that also signs the user up for Meta e-mails, so they were not downloaded on the user's behalf. The loader prefers `*_gen2.jit` if it is placed in `EGOBLUR_DIR`.

**Landmarks.** EgoBlur outputs boxes only. The 5 landmarks that the surrogate-face backend needs come from YuNet run on the face crop, with an in-box template as the fallback. On clip1, 467 of 554 detections got YuNet landmarks; most of the template fallbacks are the low-score non-face boxes.

**Threshold.** The face threshold is 0.1, which is EgoBlur's own default. At 0.2, clip2 coverage was 86.5%.

| Clip | Metric | v0.1 CenterFace | v0.2 EgoBlur (face 0.1) |
|---|---|---|---|
| clip1 | Coverage of YuNet reference faces | 94.8% | **94.8%** |
|  | Re-ID, all reference faces (redact) / masked faces | 5.2% / 0.0% | 5.2% / 0.0% |
|  | Face tracks | 2 | 9 (plant, hand and duplicate boxes at scores 0.10–0.37; blur only) |
| clip2 | Coverage of YuNet reference faces | 91.2% | **89.7%** |
|  | Re-ID, all reference faces / masked faces | 17.2% / 9.3% | 17.6% / **8.0%** |
|  | Face tracks | 45 | 69 |
| clip3 | Coverage of YuNet reference faces (n = 9) | 44.4% | **55.6%** |
|  | Re-ID, all reference faces / masked faces | 66.7% / 25% | **55.6% / 20%** |
|  | Face tracks | 13 | 82 (van fronts, sculptures) |
| all | Face detector cost per 1080p frame (same load, faces only) | 34–43 ms | 120–149 ms |

**What changed in the misses.** CenterFace caught some defocused small faces that EgoBlur misses, and it also masked back-of-head views; EgoBlur ignores those. The YuNet "misses" on clip2 that we inspected were partly back-of-head false positives of YuNet itself.

**False positives.** EgoBlur's face model fires on van fronts, sculptures and plants. In `redact` these are only extra blur. In `transform` the surrogate gate (score ≥ 0.6, at least 64 px) keeps surrogate faces off them.

**`transform` utility on clip1.** The face is still detected 95.8% of the time (v0.1: 97.9%). Identity consistency is 0.75 (v0.1: 0.92). The surrogate-face backend (`surrogate_face.py`) was being rewritten in parallel, so these two numbers reflect that work as well as the detector change.

**Plates** (EgoBlur LP vs YOLOv9, every 6th frame):
- At threshold 0.8, EgoBlur reproduces 40 of 41 YOLOv9 plate detections on clip3. It also finds a clearly visible van plate that YOLOv9 missed (score 0.99), plus some dark-grille false positives.
- At 0.3 it fired on clip1's on-screen form (73 boxes), and on bags and shoes in clip2 (22). At 0.8 neither clip has any false plates.
- Plate tracks on clip3: 6 → 12. EgoBlur LP is the default because its licence is clean, while YOLOv9's training-data provenance is unstated. `--plate-detector yolov9` stays available.

### 8.2 On-screen text: VLM scene-context gate

The gate shows Qwen3-VL-8B-Instruct (Apache-2.0, local) the frame with the OCR line outlined, plus a close-up. The VLM picks the surface the text is on:
- document, screen or overlay, badge, whiteboard, or licence plate: **masked**;
- public signage, or not real text: **left visible**;
- anything else, or a VLM failure: masked.

There is one decision per text track, and every decision is written to the audit log.

| Clip | v0.1 text "PII" tracks masked (all false positives) | v0.2 candidate tracks | Left visible by the gate | Still masked |
|---|---|---|---|---|
| clip1 (3 planted PII) | 3 (all planted) | 3 | 0 | **3/3 planted** (filed as screen/overlay); benign lines 100% readable |
| clip2 | 2 | 2 | 1 (OCR noise) | 1 (a dog's head OCR'd as "1X", mis-filed as "screen") |
| clip3 | 11 | 10 | 9 (7 shop signs, 2 clock/window patterns) | 1 (a "2x" sticker in a shop window) |

Signage false positives on the street clips went from **13 to 2**, and no planted PII was lost on clip1. A first prompt version without the "not real text" option left 3 such tracks masked on clip3. The gate costs 11–21 s per clip, mostly model loading.

### 8.3 Guardian Layer backend for the text channels (`--pii-backend local|guardian|both`)

**How it works.**
- Guardian Layer returns rewritten text. Edited spans are recovered by token-level difflib alignment and mapped back to OCR word boxes and Whisper word times.
- Masking always uses the union with the local detector: regex in `guardian` mode, LLM plus regex in `both`.
- A Guardian edit is used as a surrogate only if it overlaps a span flagged by the local detector or by Guardian's own `sensitive_words`.
- A Guardian surrogate that repeats a token of the original falls back to the local surrogate.
- One surrogate registry per run keeps screen and speech consistent.

The comparison ran on the text and audio channels of all three clips, in both modes.

| | local | guardian | both |
|---|---|---|---|
| clip1 planted on-screen PII masked (and not OCR-readable afterwards) | 3/3 (0/3 readable) | 3/3 (0/3) | 3/3 (0/3) |
| clip1 planted spoken PII removed (GT tokens left in the Whisper re-transcript) | 3/3 (0/7) | 3/3 (0/7) | 3/3 (0/7) |
| clip1 false-positive values (screen / speech) | 0 / 0 | 0 / 0 | 0 / 0 |
| clip1 benign on-screen lines readable | 100% | 100% | 100% |
| Screen ↔ speech surrogate consistency (same value, same surrogate) | 3/3 | 3/3 | 3/3 |
| Surrogates re-detected as PII (`transform`) | 3/3 | 3/3 | 3/3 |
| Who supplied the surrogates | local | name: Guardian; phone and account: local fallback | same as `guardian` |
| Street-clip text tracks masked after the gate (clip2 / clip3) | 1 / 1 | 0 / 1 | 1 / 2 |
| Guardian Layer HTTP calls (6 runs) | 0 | 26 (36 cache hits) | 0 (62 cache hits) |
| Guardian Layer latency | — | 14.8 s total, about 0.57 s per call | (cached) |

**What Guardian Layer did on the clip1 panel and transcript.** These were read from the result cache, which holds synthetic data only.
- **Names.** `Margaret Holloway` → `Elizabeth Loleatta`, consistent between screen and speech.
- **Phone, partial leak.** On screen the rewrite produced `(414) 555-0199`: the area code moved by 1 and the rest stayed verbatim. In speech it produced `526-666-1200`. The leak check rejected the screen surrogate, and the regex net masked the value.
- **Account number, never flagged.** Guardian never flagged `88427139` or `884-271-39` in either channel, so it was left verbatim; the regex net caught it.
- **Non-PII flagged.** It flagged non-PII as sensitive: `PATIENT INTAKE`, `Flood Clinic`, `Phone`, `river`, `Room`, `next week`, and `a great day`. It "transformed" some of these into inflections (`Floods`, `Phones`, `rivers`, `Rooms`).
- **Without the plausibility filter.** A Guardian-only span must contain a digit or `@`, or have been really replaced. Without that filter the union masked **9** text tracks on clip1 instead of 3, plus a 4th audio event (`demo/outputs/v02_guardian_nofilter_clip1/`).

The diagnostics below are summed over sampled OCR frames, so repeated lines count once per frame:
- 345 raw Guardian flags. 105 of them were left unchanged by Guardian's own rewrite, and 154 were inflection-only.
- 193 of 320 Guardian edits were ignored as non-PII.
- In `both` mode, 48 local-detector spans were missed by Guardian Layer.

**Conclusion.** On this demo, Guardian Layer adds realistic name surrogates. It finds no PII that the local LLM plus regex missed. Used alone, it would have leaked the account number and part of the phone number, and it would have masked field labels. The union-plus-filter design keeps the outcome identical to `local` for privacy and utility. `both` is the recommended setting when Guardian Layer surrogates are wanted. The demo is small (3 planted values), so this is a qualitative check, not a benchmark.

**API budget.** This task used **33 Guardian Layer calls** of the 2,000 cap: 2 for smoke tests, 5 for the run before the filter was added, and 26 for the comparison. There were no 429s.

**Deployment.** Guardian Layer can be deployed on-prem (confirmed by the vendor), so a product can keep this backend local. The hosted API was used here only for research on synthetic or public demo data.

**Follow-up: `guardian-replace` (2026-10-06; web demo config "Blur + Custodian Guardian Layer").** In this mode Guardian Layer supplies the replacement values, but it does not decide which words change. Faces and plates are blurred as in Blur, with `--mode transform --face-mode redact`.
- **Spans.** The local LLM and regex find the spans. Guardian `sensitive_words` add a span only if it contains a digit or `@`, or if it overlaps a typed local span. In the second case it may extend that span only over adjacent word characters.
- **Values.** For each span, Guardian's value is used only if `complete_replacement()` accepts it. The checks are:
  - same type and format;
  - digit values change more than half of their digits and keep no run of 3 unchanged digits;
  - names and other words: every word is new;
  - placeholders are rejected.
  If the check fails, the local same-type surrogate is used.
- **Audit.** The log records `surrogate_source` and `surrogate_note` for each span.

Results on the six demo clips. All Guardian results came from the existing result cache except one call on clip 6, so this task made 2 new HTTP calls.
- **Planted PII.** On screen 9/9 caught, in speech 3/3, and benign lines readable 4/4.
- **Spans.** The masked and replaced spans are the same as in "Blur + fake text/speech", except for one digit-bearing Guardian-only OCR fragment on clip 3. The scene gate left that fragment visible, so no extra pixels changed.
- **Who supplied the values.** 6 of the 9 planted values used Guardian Layer's value: all 3 names, both phone numbers and the e-mail address. The phone surrogates come from Guardian's full-value `top1_replacements`, such as `(526) 666-1200`. Its in-text rewrite changed only the area code, and that partial edit is rejected. Guardian left the 3 ID numbers (account, member and badge) unchanged, so they used the local fallback. The 5 non-planted text values that the local detector also replaces (banner and sign fragments) all used the local fallback.
- **Face metrics.** Identical to Blur: same masks, and the precision sample is shared when the face masks are identical.


### 8.4 Throughput (v0.2, shared host)

End-to-end throughput with all channels, redact / transform:

| Clip | v0.1 | v0.2 |
|---|---|---|
| clip1 | 1.9 / 3.8 fps | 0.9 / 1.7 fps |
| clip2 | 8.4 / 5.6 fps | 2.3 / 2.0 fps |
| clip3 | 7.6 / 4.6 fps | 2.2 / 2.3 fps |

- Detection now dominates pass 1. EgoBlur face and plate models take about 105–145 ms per 1080p frame each, versus about 20 ms for CenterFace in v0.1. Plates run on every 2nd frame.
- Part of the drop is host load: v0.2 ran with a load average of 170–550 on 256 cores.
- The next steps are batched fp16 inference and a fast tier distilled from EgoBlur on licensed data.

### 8.5 Open issues

- EgoBlur Gen2 weights are not tested (they need the e-mail form).
- YuNet supplies the landmarks. It is MIT-licensed but trained on WIDER FACE; `--face-landmarks template` avoids it.
- The scene gate still mis-files some non-text blobs as "screen". That direction is safe, because they stay masked.
- The Guardian Layer diagnostics count per OCR-frame occurrence, not per unique line.

## 9. Formal evaluation (481 clips): does the surrogate keep more utility than redaction?

Benchmark: `custodian_video/eval/`, face channel only. The clips are:

- **Kinetics-400 val, 400 clips:** action recognition and pose.
- **MOT17, 21 windows:** person detection and count.
- **VoxCeleb2, 60 probes:** identification against a 118-identity gallery.

Each tag pins one pipeline commit. Statistics are paired per clip, with 95% bootstrap CIs and a ±2-point TOST. The `reencode` null arm runs the same pipeline with nothing masked, which isolates codec noise. Its action top-1 changes by +1.0 [−0.25, +2.5], so the set is powered to detect differences of about 2 points.

### 9.1 Transform minus blur (Kinetics, n = 400; positive = surrogate keeps more)

| | v0.1 static surrogate | v0.2 animated surrogate | v0.2-gate32 surrogates down to 32 px | v0.2-gen2 EgoBlur Gen2 @0.10 |
|---|--:|--:|--:|--:|
| Pose agreement, 5 head keypoints (OKS ×100) | +5.4 [4.0, 6.8] | +9.2 [7.8, 10.7] | **+13.1** [11.6, 14.7] | +9.1 |
| Pose agreement, all 17 keypoints | +3.6 [2.9, 4.4] | +6.2 [5.3, 7.1] | **+8.3** [7.3, 9.3] | – |
| Action top-1, VideoMAE-B | +1.8 [−0.8, 4.5] | +2.5 [−0.3, 5.5] | **+3.5** [0.5, 6.5] | +3.8 |
| Action top-1, X-CLIP B/32 | −4.0 [−7.0, −1.0] | −1.0 [−4.0, 1.8] | −1.5 [−4.5, 1.5] | +0.5 |
| VideoMAE prediction = original's | +3.5 [0.0, 7.0] | +2.8 [−0.8, 6.3] | **+5.0** [1.0, 9.0] | – |
| Faces still present after masking | 54.7% | 62.2% | 79.8% | 64.2% |

The v0.2-gen2 entries are differences of the per-arm deltas against the original (not paired contrasts), so they have no CI.

### 9.2 Privacy (all arms remove identity)

- **Cross-video identification on VoxCeleb2** (detector attacker, rank-1 per clip): transform 0.0% in v0.1, 3.3% in v0.2 and gate32, 5.0% in gen2. Blur is 0–1.7%; original is 100%. All CIs overlap.
- **Same-location ArcFace re-ID:** transform 11–13% vs blur 14.5–15.6%. These numbers include the impostor floor, i.e. the rate at which a different person's face matches.

### 9.3 Findings

1. **Each iteration widened the surrogate's advantage over blur without costing privacy.**
   - Animating the surrogate (v0.1 → v0.2) roughly doubled the pose advantage. It also removed v0.1's significant X-CLIP deficit.
   - Lowering the surrogate gate to 32 px (gate32) is the best configuration. It gives a significant gain over blur on pose, VideoMAE action and prediction agreement.
2. **Masking is not utility-neutral.** Every arm, the surrogate included, loses about 8–12 points of action top-1 against the original, and none passes the ±2-point equivalence test. Unlike the text study, the claim here is "better than redaction", not "equivalent to the original".
3. **Person detection on MOT17 is essentially unaffected by every arm** (AP50 within ±1.2 points). Faces are a small part of a whole-body task.
4. **The cost of gate32** is a slightly larger person-count error on MOT17 (+0.26 people/frame vs blur), because small-face surrogates sometimes perturb nearby detections.
5. **EgoBlur Gen2 @0.10 performs on par with Gen1 @0.10.** Gen1 stays the default.
   - Gen2 helps action recognition a little (+3.8 vs +2.5) and disturbs MOT17 detection less, because it makes fewer false-positive masks.
   - It re-identifies slightly more (VoxCeleb2 5.0% vs 3.3%, n.s.) and fragments tracks on talking-head videos (308 vs 81 VoxCeleb2 tracks, more blur fallbacks).
   - It is about 30% slower (median 1.9 vs 2.7 fps on Kinetics, same host).
   - Gen2's official camera-rgb threshold (0.674) is unsuitable for de-identification: on the demo street clips it misses most small faces.

### 9.4 Recommended configuration

- **Default:** EgoBlur Gen1 @0.10 with `--surrogate-min-face 32` (the gate32 setting).
- **Next steps:**
  - Make gate32 the CLI default after checking the MOT17 count regression.
  - Merge tracks of the same person, so they get one surrogate identity.
  - Re-run on a larger VoxCeleb2 probe set to tighten the privacy CIs.

Reproduce: `CUDA_VISIBLE_DEVICES=<gpu> WORKERS=4 custodian_video/eval/run_all.sh <tag> git:<sha> <baseline>`. Pass per-arm options with `EXTRA_ARGS`, but never `--face-threshold`, because it overrides the `reencode` null arm. Results are in `custodian_video/eval/results/<tag>/summary.md`.

## 10. Surrogate realism v2 and fail-closed face masking (2026-10)

The demo's `transform` faces looked pasted on: a dark smear over one cheek (the paste-back mask included LivePortrait's warped hair / background), a doubled hairline, a much younger "beauty-filter" identity with lip make-up, a face narrower than the head, and skin cleaner than the surrounding footage. Experiment `custodian_video/experiments/face_e/` rebuilt the backend behind the same `surrogate_face.py` interface (`realism="v2"`, default; `--surrogate-realism v1` / `CUSTODIAN_FACE_REALISM=v1` restores the old path, identical up to GPU run-to-run noise).

### 10.1 What changed

**Identity generation.**
- The local Qwen3-VL-8B (Apache-2.0; 4B fallback on a full GPU) reads coarse attributes of the keyframe face: apparent age band, gender presentation, skin tone, apparent ancestry, hair, facial hair, glasses. They are kept in memory only (never logged) and become the prompt, with a negative prompt against make-up, airbrushing and glamour. SD-1.5 renders unretouched faces about a decade older than prompted, so the age text is shifted down.
- SD-1.5 img2img + **ControlNet v1.1 depth** (`lllyasviel/control_v11f1p_sd15_depth`, OpenRAIL) on a **blurred** Depth-Anything-V2-**Small** depth map (Apache-2.0): head shape, width and pose are kept, nose / eye / lip relief is free. `sd-vae-ft-mse` decoder (MIT). Grey-world white balance of the SD input, so a warm colour cast is not read as darker skin. Seed-chosen feature words ("narrow nose", "deep-set eyes", ...) make different tracks different people.
- **Generation-time guard:** up to 10 deterministic candidates per track are re-enacted and pasted on 16 sampled frames; the first whose 75th-percentile SFace cosine to the original is <= 0.22 is kept (else the lowest).

**Paste-back.** Every frame is rendered in `prepare()`, so alignment and masks are smoothed zero-phase. The render is landmarked and similarity-aligned to the original's **jaw / cheek outline** (fixes the size / offset mismatch behind the smear and the doubled hairline, without imposing the original's eye / nose / mouth layout). The replaced region is the face hull intersected with the rendered face's own hull (the surrogate's hair / background is never pasted); in a forehead band only where the surrogate shows skin. Per-frame LAB colour match, low-frequency illumination match, blur to the original's sharpness and luma grain up to the original's (MAD) noise level. Poisson blending is available (`blend="poisson"`) but was not better.

**Fail closed (privacy first).** In v2 a face is painted with a surrogate only if its detector box is >= 64 px, it has landmarks, and the SFace guard is available; the guard now also aligns with YuNet on the original (the evaluator's alignment) and uses 0.25 instead of 0.30 for boxes under 120 px. If a frame still matches after covering more of the face and damping the expression, that face is **blurred** (v1 fell back to an unverified static warp). For every config: small faces get up to +50 % mask padding, the mask box is the envelope of the smoothed and unsmoothed track (no lag on fast motion), and a **YuNet recall net** (score >= 0.7, MIT) adds faces EgoBlur missed; those are always blurred (`--face-recall-yunet 0`, `--small-face-extra 0` restore the old masks).

### 10.2 Realism, motion, speed (face-only harness, same tracks; v1 = before, v2 = after, before the fail-closed gate)

| clip | SFace re-ID v1 / v2 (max cos) | ArcFace re-ID, eval only, v1 / v2 | per-track mouth r v1 / v2 | blink r | yaw r | Qwen3-VL prefers v2 (20 pairs, order randomised) | face-channel fps v1 / v2 |
|---|---|---|---|---|---|---|---|
| clip1 interview | 0 / 0 (0.31 / 0.30) | 0 / 0.4 % (2 frames at 0.22, thr 0.217) | 0.93 / 0.93 ; 0.84 / 0.84 | 0.94 / 0.91 | 0.98 / 0.99 | 19 / 20 | 6.2 / 3.2 |
| clip4 close-up | 0 / 0 (0.31 / 0.20) | 0 / 0 | 0.66 / 0.70 | 0.95 / 0.95 | 0.99 / 0.99 | 20 / 20 | 9.9 / 3.8 |
| clip5 presenter | 0 / 0 (0.29 / 0.31) | 0 / 0 | 0.83 / 0.90 | 0.94 / 0.94 | 0.95 / 0.95 | 20 / 20 | 12.3 / 4.0 |
| clip6 group (small faces) | 12.1 / 14.8 % | 24.8 / 30.2 % | (faces mostly < 64 px) | | | 7 / 20 | 2.1 / 0.8 |

On clip6 v2 was *more* re-identifiable than v1 and the VLM did not prefer it: below ~64 px an attribute-matched surrogate leaks more than blur. That motivated the fail-closed gate: with it, the harness clip6 SFace re-ID drops to 3.8 % (clip1 unchanged at 0 %, per-track motion unchanged). A boundary-seam metric (colour step across the edit's outer boundary where the original has no edge) is slightly *worse* for v2 (e.g. clip1 9.3 vs 7.8 dE): v2's edit ends at the face outline, v1's in hair / background, so this metric does not capture the smear and identity defects that dominated the visual verdict. v2 costs 55-75 s more preparation per large-face track (VLM + up to 10 ControlNet candidates); per-frame paste speed is unchanged.

Keyframes (original | old | new): `custodian_video/experiments/face_e/media/clip{1,4,5,6}_keyframes_original_old_new.jpg`; VLM pairs in `media/vlm_pairs/`.

### 10.3 Face privacy over the web-demo outputs (every clip x config)

Reference faces are found independently with YuNet on the original (every 2nd frame, >= 24 px). "Before" is the demo as it was at the start of this work (v1 surrogates, old masks); "after" is the regenerated demo (v2 + fail-closed gate, recall net, small-face padding). Full per-config tables: `custodian_video/experiments/face_e/out/privacy_{before,after}.{md,json}` (`privacy_report.py`).

| clip | confident faces (YuNet >= 0.8) | coverage before -> after (EgoBlur alone, after) | SFace re-ID of confident faces, % before -> after: blur / box / surrogate configs (min-max) | ArcFace (eval) same, % | ArcFace impostor floor after, % | all YuNet refs coverage before -> after |
|---|---|---|---|---|---|---|
| clip1 | 227 | 100.0 -> 100.0 (100.0) | 0.0/0.0/0.0-0.0 -> 0.0/0.0/0.0-0.0 | 0.0/0.0/0.0-0.0 -> 0.0/0.0/0.0-0.0 | 50.0 | 95.8 -> 97.5 |
| clip2 | 247 | 99.2 -> 100.0 (100.0) | 8.1/2.8/2.0-8.1 -> 1.2/0.0/1.2-1.2 | 33.2/2.8/19.8-33.6 -> 17.0/1.2/16.6-17.4 | 33.1 | 83.7 -> 91.3 |
| clip3 | 0 | n/a -> n/a (n/a) | n/a/n/a/n/a -> n/a/n/a/n/a | n/a/n/a/n/a -> n/a/n/a/n/a | n/a | 0.0 -> 50.0 |
| clip4 | 179 | 100.0 -> 100.0 (100.0) | 0.0/0.0/0.0-0.0 -> 0.0/0.0/0.0-0.0 | 0.0/0.0/0.0-0.0 -> 0.0/0.0/0.0-0.0 | n/a | 100.0 -> 100.0 |
| clip5 | 179 | 100.0 -> 100.0 (100.0) | 0.0/0.0/0.0-0.0 -> 0.0/0.0/0.0-0.0 | 0.0/0.0/0.0-0.0 -> 0.0/0.0/0.0-0.0 | n/a | 100.0 -> 100.0 |
| clip6 | 806 | 99.5 -> 100.0 (100.0) | 3.1/1.6/4.5-9.2 -> 0.4/0.4/0.2-0.5 | 28.7/13.7/16.8-27.5 -> 20.6/13.8/20.1-21.2 | 28.7 | 97.7 -> 99.5 |

- **Coverage.** Every confidently detected face (YuNet >= 0.8) is masked in every clip and config, and EgoBlur alone already covers 100 % of them after the padding change, so this is not circular with the YuNet recall net. Coverage of *all* YuNet detections stays below 95 % on clip2 (91-94 %) and clip3 (50 %), and we inspected every uncovered one: dogs, a hand, flowers, the back of a head (clip2), and a building facade, clock tower and trees (clip3, which has no detectable real face). The same goes for clip1's few misses: T-shirt print, hands and an object at the top edge. Lowering EgoBlur's threshold below 0.10 did not add real faces (0.05 and 0.03 give identical results).
- **Re-identification.** SFace (the pipeline's own guard model) matches 0 % of confident faces on clips 1, 4, 5 and <= 1.2 % (clip2) / <= 0.5 % (clip6) after the fix, versus up to 8.1 % / 9.2 % before.
- **ArcFace on small faces.** The eval-only ArcFace-R50 still reports 17-21 % on the small-face clips, but its impostor floor there is just as high: 29-33 % of the time it "matches" an original face with a *different* person's masked face in the same frame. So at that size it measures context, not identity. A solid box, which leaves no face pixels, also gets 1-14 %.
- **Clip1 floor.** Its 50 % impostor floor rests on only 2 pairs.


### 10.4 Remaining weaknesses

- SD-1.5 still skews older and occasionally rough-skinned for some faces; hairline / glasses differences show at the mask edge on large faces.
- Hair, ears and the face outline stay the original's, and on small faces the context (hairnet, hair, earrings, clothing) still lets SFace / ArcFace "match" even a solid box; face masking alone cannot reach 0 % there. Head / whole-body anonymisation would be needed.
- Preparation time grows with the number of large-face tracks.

## 11. Research-only synthetic-identity face swap (experiment face_g, 2026-10)

**Licence status: research-only, not for commercial use.** Non-commercial model licences are acceptable for now (no commercial use yet), so we tested face swapping to a synthetic identity. Backend: `custodian_video/surrogate_face_swap.py`, opt-in via `--face-backend research-swap`. The default backend is unchanged.

### 11.1 Method (configuration "R")

**Synthetic identity.** Each face track gets one identity that belongs to no real person:
- The local Qwen3-VL reads coarse attributes of the face.
- SD-1.5 text-to-image renders 8 portraits of people who do not exist. The original face is never given to the generator. Portraits with hair over the forehead are dropped.
- The portrait whose swap is least similar to the original under SFace is kept (75th percentile over sampled frames). The evaluation-only ArcFace is not used for this choice.

**Per-frame swap** on the original frame (no re-enactment):
- Swapper: HyperSwap 1b 256, rendered at 512 px with pixel boost.
- Identity: the portrait's ArcFace w600k_r50 embedding, pushed away from the original's (push 0.5).

**Paste-back mask.** Face parsing (BiSeNet) of the original and of the swapped crop, intersected with an occlusion mask (XSeg) and the swapper's own mask. Hair, ears, neck and background always stay the original's. This removes the ghost hair, background bleed and forehead blotches of v2/v3.

**Mouth.** The original mouth and lips are kept, so speech is followed. Without this the swapper tends to close the mouth.

**Fail closed.** As in v2:
- Faces below the size gate, or without landmarks, are blurred.
- Every frame passes the SFace guard or is blurred.
- Guard trips are merged into contiguous blur segments. A track tripping on more than 50 % of its frames is blurred entirely. In the face-only harness this happened to the glasses wearer on clip 1. In the web-demo run, which picked a different synthetic identity, 38 of her 84 frames were swapped and 46 blurred.

**Licences:**

| Component | Licence |
|---|---|
| HyperSwap (FaceFusion) | ResearchRAIL |
| InsightFace ArcFace / buffalo_l | non-commercial |
| BiSeNet parsing | MIT code, non-commercial CelebAMask-HQ training data |
| XSeg | GPL-3.0 |
| SD-1.5 | OpenRAIL-M |
| Qwen3-VL | Apache-2.0 |
| SFace / YuNet | Apache-2.0 / MIT |

### 11.2 Results (face-only harness; same tracks as v2)

**Privacy.** Same-place re-ID was measured with three recognisers:
- **SFace.**
- **ArcFace-R50**, evaluation-only. It is also the swapper's own identity space, so a low score here could be circular.
- **glintr100** (antelopev2), an independent recogniser. Threshold 0.2367 = FAR 1e-3 on 1.77 M VoxCeleb2 gallery impostor pairs.

Findings:
- R re-identifies **0 % on clips 1, 4 and 5** with all three recognisers (max cosine 0.26 / 0.17 / 0.14). The cross-clip impostor floor is 0.01 / 0.66 / 0.69 %.
- On clip 6 every face fails the size gate or the guard, so the output equals Blur. Re-ID is 2.3 / 7.3 / 8.1 %, against an in-frame floor of 1.9 / 5.4 / 9.0 %.

**Motion** (correlation with the original, R vs v2):

| clip | mouth | blink | yaw |
|---|---|---|---|
| clip 1 | 0.98 vs 0.76 | 0.96 vs 0.69 | 0.997 vs 0.986 |
| clip 4 | 0.98 vs 0.70 | 0.943 vs 0.943 | 0.997 vs 0.990 |
| clip 5 | 0.97 vs 0.91 | **0.917 vs 0.946** | 0.989 vs 0.950 |

Eyebrow motion follows the original less well on clip 1 (0.55 vs 0.83).

**Naturalness.** Qwen3-VL pairwise judge, R vs v2, 40 pairs per clip:

| clip | R wins | note |
|---|---|---|
| clip 4 | 77.5 % | |
| clip 5 | 40 % | judge chose the second image 90 % of the time; with both orders shown, 30 of 40 pairs are ties |
| clip 1 | 47.5 % | v2 wins mostly where R is blurred; on frames where R shows a face, R wins 18 to 7 |

**By eye** (`custodian_video/experiments/face_g/media/clip{1,4,5}_zoom_original_v2_R.jpg`):
- No ghosting, bleed, blotches or doubled hairline.
- Edges are clean.
- Speech is followed.

### 11.3 Verdict and remaining weaknesses

R misses part of the formal bar:
- The VLM prefers it by at least 70 % on clip 4 only.
- Blink fidelity on clip 5 is slightly below v2.

The by-eye review prefers it clearly. Remaining weaknesses:
- The synthetic identities often carry a different brow and eye expression (a frown on clip 1, wide eyes on clip 5).
- The swap is slightly softer than the original at 1:1.
- Faces whose identity cannot be removed (glasses wearer, small faces) are blurred.

**Web demo.** R runs on the demo page as "Surrogate face · research models (experimental)" (`--face-backend research-swap --pii-backend guardian-replace`). It is labelled research-only with non-commercial licences. Since 2026-10-08 it is the page's default configuration (Blur stays the safest one); see §12. The face masks are identical to Blur's.

**Stabilisation (2026-10-07; `stabilize=True`, now the default of this backend; `stabilize=False` gives the R behaviour).** The first page version switched faces between swap and blur: the glasses wearer on clip 1 was blurred for her last ~1.5 s, and the main speaker toggled. Changes:

1. **Identity choice.** The synthetic identity is chosen by the share of guard-tripping frames on 48 evenly spaced frames of the track, not by mean similarity. If more than 10 % trip, a stronger push (+0.3) is tried. If the best identity still trips on more than 25 %, the whole track is blurred: one consistent blur, no toggling.
2. **Per-frame retries.** A tripping frame is re-rendered with push +0.3 and then +0.6 before it is blurred.
3. **Blur segments.** Trips are padded by ±2 frames and blur segments less than 15 frames apart are merged. Swap runs shorter than 12 frames that touch a blur segment are blurred. Every one of these steps only adds blur.
4. **Cross-fade.** Swap and blur are cross-faded over 4 frames; both parts of the blend are verified.
5. **Landmark smoothing.** The 5 alignment landmarks are smoothed with a zero-phase one-euro filter (forward and backward pass).

Replacing the original mouth with the swapped one, plus a push of up to 1.1, did make the glasses wearer pass the guard in the face-only harness. Her face then looked grimacing, so mouth replacement is not used.

Page outputs, before → after:

| clip | track | swapped share | transitions | longest blur run |
|---|---|---|---|---|
| clip 1 | speaker | 94.7 → 98.7 % | 3 → 1 | 13 → 5 frames |
| clip 1 | glasses wearer | 45 → 87 % | 2 → 2 | 39 → 6 frames, only her last frames when she looks up |
| clip 4 | | 99.4 → 99.4 % | unchanged | |
| clip 5 | | 94.4 → 99.4 % | 5 → 1 | 9 → 2 frames |

Landmark jitter relative to the original (1 = as steady as the original face):

| clip | before | after |
|---|---|---|
| clip 1 speaker | 0.91 | 0.88 |
| clip 1 glasses wearer | 1.01 | 0.93 |
| clip 4 | 1.40 | 1.35 |
| clip 5 | 1.00 | 0.93 |

**Re-ID** (SFace / ArcFace / glintr100, same place) did not get worse:

| clip | before | after | note |
|---|---|---|---|
| clip 1 | 0 / 3.2 / 3.2 % | 0 / 3.2 / 3.2 % | identical to Blur on the same faces |
| clip 4 | 0 / 0.3 / 0 % | 0 / 0 / 0 % | |
| clip 5 | 0 / 0 / 0 % | 0 / 0 / 0 % | |
| clips 2 and 6 | | | unchanged; every face blurred |

On the glasses wearer's frames in the face-only harness, re-ID is 0 % on all three recognisers (max SFace 0.27).

**Per-track motion after** (mouth / blink / yaw / brow raise):

| track | after | before |
|---|---|---|
| clip 1 speaker | 0.98 / 0.96 / 1.00 / 0.87 | unchanged |
| clip 4 | 0.98 / 0.86 / 1.00 / 0.64 | |
| clip 5 | 0.98 / 0.92 / 0.99 / 0.64 | |
| clip 1 glasses wearer | 0.99 / 0.73 / 1.00 / 0.25 | 0.99 / 0.87 / 1.00 / 0.50 on her 38 swapped frames |

The glasses wearer's after figures cover 73 frames, mostly with the eyes looking down.

Frame strip: `custodian_video/experiments/face_g/media/page_clip1_glasses_wearer_original_before_after.jpg`.

Tried and not better: inswapper_128, HyperSwap 1a/1c, GFPGAN/CodeFormer restoration, temporal smoothing (it hurts blinks), keeping the original eyes or brows (brows: SFace 0.42, too identifying), and iterated swaps.

## 12. Person-filter fix, gender/age-matched photoreal identities, surrogate as page default (experiment face_h, 2026-10-08)

Feedback on the default-to-be configuration (clip 1, last segment, the second speaker): (a) several blurred rectangles on the curtain / bookshelf next to her head, and (b) "a man's face put on the woman". Code: `custodian_video/person_filter.py`, `pipeline.py`, `face_attributes.py`, `surrogate_face_swap.py`; harnesses in `custodian_video/experiments/face_h/`.

### 12.1 (a) Background face false positives

Clip 1, 14.8-18.2 s, EgoBlur tracks besides her face (`face-5`, score 1.0):

| track | what it is (crops by eye) | max score | kept masked because | after |
|---|---|---|---|---|
| face-6, 83 frames | dark plant pattern on the curtain, left of her | 0.37 | `near_frame_edge` (the 1.25x-dilated MASK box reached the top 3 %; the raw box is 80 px inside) | unmasked |
| face-8, 23 frames | curtain above-left of her head, grazing her hair | 0.15 | `near_frame_edge` (raw box 8-17 px from the top) | unmasked |
| face-7, 28 frames | same curtain pattern; YuNet scored it >= 0.7 on 2 frames | 0.59 | `yunet_recall_net` | still masked (rule unchanged) |

Filter changes (all fail closed; they apply only to tracks scoring < 0.4 whose raw box is >= 64 px, i.e. not to small / distant faces):
- the edge rule uses the RAW detector box, not the dilated mask box; such low-score large boxes may touch the TOP border (a person cut off at the top still has a body in the frame, which the person detector sees);
- a head-region touch counts only if the box centre is in the region or >= 40 % of the box is (grazing the 25 %-widened margin no longer counts);
- a touch is excused when that person is explained by exactly one confident face of another track on the same frame (EgoBlur >= 0.9, >= 96 px), the candidate's centre lies outside that face's head estimate (+25 % width, +35 % height up) and it is at least half that face's size. Zero or several confident faces, or a person seen from behind, still keep the track.
A first version without the 64 px limit unmasked a real, partly hidden head on clip 2 (track 50, 25 px): the limit was added for that reason.

Acceptance (same protocol as before, `web_accuracy.py faces/vlm`): offline old-vs-new comparison on dumped pass-1 data (`face_h/eval_fp_filter.py`): 5 more tracks unmasked of 199 (28 -> 33): clip 1 the plant and curtain tracks above, clip 2 one single-frame track on dogs' backs (animal box), clip 3 two van-headlight tracks. Every one was inspected by eye (`face_h/out/changed/`). Real-face recall is unchanged in every clip and configuration (same reference misses, 0 new; confident-face coverage 100 %). Sampled face-mask precision (all page configs share the masks): 68.0 % -> 69.0 % (204 -> 207 of 300), frame-weighted 69.8 % -> 72.6 %; clip 1 66 % -> 86 % (frame-weighted 84.5 % -> 98.6 %). On clips 2 and 3 the sampled numbers moved down by 4 and 3 of 50 only because removing tracks changes the round-robin sample; no masked track became a worse one.

### 12.2 (b) Gender / age / skin-matched, better-looking synthetic identities

The old pick was by guard trips only, among SD-1.5 portraits that were never checked: on the second speaker the identity that passed the guard was a male-looking face. New identity pipeline (research-only models allowed):
1. **Generator: RealVisXL V5.0** (SDXL photoreal fine-tune, CreativeML OpenRAIL++-M, + SDXL-VAE fp16-fix, MIT), 16 candidates at 1024 px from the Qwen3-VL attributes (prompt under 77 CLIP tokens, age text shifted ~3-5 years down, no glasses in the portrait: they pulled the identity towards the wearer). Every second candidate omits the apparent-ancestry word: attribute-matched faces of the same appearance were all too close to the second speaker for the guard (100 % trips at level 0 with 8 candidates, 35-38 % at push +0.9); with the mixed set one passes.
2. **Attribute check before any ranking:** Qwen3-VL reads every portrait; a candidate is rejected if its apparent gender presentation differs, its age band is more than one band off, its skin tone more than one step off, or it has major artefacts (another batch if < 2 match; none -> the whole track is blurred; VLM unavailable -> tracks with attributes are blurred, never an unverified identity). Clip 1: 15/16 and 14/16 kept (rejections: age), clips 4/5: 16/16.
3. **Quality:** Qwen3-VL compares the best 8 candidates pairwise, every pair in both orders (A/B swapped), score = wins. A single numbered grid was useless: the judge picked the first tile in either order.
4. **Guard / privacy:** among candidates passing the dense guard test at a level (trips <= 10 %), only those within 0.03 of the most private candidate's p75 SFace AND with p75 <= 0.18 are acceptable; the best-looking of them is used; else the next push level (+0, +0.3, +0.6, +0.9). Without the 0.18 target the clip-4 pick re-identified 2.5 % (then 1.1 % with the margin alone) of frames under the evaluation-only ArcFace; with it 0 %.

Contact sheet (original | previous SD 1.5 page output | new page output | new portrait): `custodian_video/experiments/face_g/media/identities_clip1_4_5_original_previous_new.jpg`. Clip-1 ending strip (original | before | after): `media/page_clip1_ending_original_before_after_{surrogate_research,blur}.jpg`.

**Privacy (same place, SFace / ArcFace / glintr100, re-ID rate; max cosine):**

| clip | before | after | floor |
|---|---|---|---|
| clip 1 | 0 / 3.2 / 3.2 % (max 0.28) | 0 / 3.2 / 3.2 % (max 0.27) | identical to Blur on the same (blurred) frames |
| clip 4 | 0 / 0 / 0 % (max 0.30 / 0.21 / 0.21) | 0 / 0 / 0 % (max 0.21 / 0.16 / 0.13) | cross-clip 0 / 0 / 0 % |
| clip 5 | 0 / 0 / 0 % (max 0.18 / 0.14 / 0.16) | 0 / 0 / 0 % (max 0.22 / 0.02 / 0.02) | cross-clip 0 / 0 / 0 % |
| clips 2, 6 | unchanged (every face blurred; < 64 px or guard) | unchanged | in-frame floor as before |

`privacy_report.py` (confident faces, SFace / ArcFace): identical to before on all six clips.

**Motion** (mouth / blink / yaw / brow raise, per-clip correlation with the original): clip 1 0.98 / 0.85 -> 0.94 / 1.00 / 0.69 -> 0.52; clip 4 0.98 / 0.86 -> 0.96 / 1.00 / 0.64 -> 0.79; clip 5 0.98 / 0.92 -> 0.94 / 0.99 / 0.64 -> 0.59. Blinks improve everywhere; brow raise drops on clip 1 (the second speaker needs push 1.4, which gives a firmer brow) and clip 5.

**Stability:** clip 4 and 5 unchanged (99.4 % swapped, 1 transition; landmark jitter ratio clip 4 1.35 -> 0.91, clip 5 0.93 -> 0.98). Clip 1: second speaker 86.9 % swapped as before (2 short blur runs, longest 6 frames, incl. her last frames); main speaker 98.7 -> 96.5 % swapped, 1 -> 3 transitions, longest blur 5 -> 8 frames (her identity now needs push +0.3).

**Remaining weaknesses:** the second speaker on clip 1 needs a strong identity push (1.4): her synthetic face is clearly female and natural but has a slight frown; attribute-matched identities are closer to the person, so more faces need stronger pushes; Qwen3-VL's quality judgement is coarse; the 4B VLM fallback is not installed, so on a full GPU the check fails and those faces are blurred (fail closed); one curtain pattern on clip 1 (face-7) stays masked because the YuNet recall net fired on it.

**Page.** "Surrogate face · research models (experimental)" is now the page default (manifest flag `default`, localStorage key `vdemo3-cfg`), with a "default" badge; Blur keeps the "safest" badge. The accuracy-at-a-glance table shows Surrogate face first, then Blur (safest reference) and Blur + Guardian Layer. The page states that faces smaller than 64 px are always blurred (so clips 2 and 6 look like Blur). Guardian Layer: all values from the cache, 0 new API calls.

### Appendix: files

The prototype lives in `custodian_video/`:
- **Package** (`custodian_video/custodian_video/`): `pipeline.py`, `detectors.py`, `tracking.py`, `text_channel.py`, `audio_channel.py`, `surrogate_face.py`, `masking.py`, `pii.py`, `scene_context.py` (v0.2) and `cli.py`.
- **Demo** (`custodian_video/demo/`):
  - `make_demo_inputs.py`, `evaluate.py`, `keyframes.py`, `vlm_check.py` and `run_demo.sh`
  - `inputs/` and `outputs/` (videos, audit JSON, eval JSON)
  - `keyframes/` (original | redact | transform PNGs)
