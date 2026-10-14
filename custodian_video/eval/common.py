"""Shared paths, video I/O and face models for the video Guardian evaluation."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parents[1]
DATA = Path(os.environ.get("VIDEO_EVAL_DATA", str(REPO / "data" / "video_eval")))
RAW, CLIPS, MODELS = DATA / "raw", DATA / "clips", DATA / "models"
OUTPUTS = DATA / "outputs"          # outputs/<tag>/<arm>/<clip>.mp4
CACHE = DATA / "cache"              # version-independent caches (original-arm features)
RESULTS = EVAL_DIR / "results"      # results/<tag>/...
EMB = DATA / "embeddings"           # embeddings/<tag>/<arm>/<clip>.npz (kept out of git: ~40 MB/tag)
MANIFEST = EVAL_DIR / "manifest.json"

# Arms: name -> extra CLI args for `custodian-video` (None = original, not processed).
# Only the face channel is enabled: this benchmark measures face de-identification.
ARMS = {
    "original": None,
    # null arm: same decode -> pipeline -> libx264 path, but the face threshold is unreachable so
    # nothing is masked. Separates codec/re-encoding noise from the effect of masking.
    "reencode": ["--mode", "redact", "--face-threshold", "1.01"],
    "redact_blur": ["--mode", "redact", "--face-style", "blur"],
    "redact_box": ["--mode", "redact", "--face-style", "box"],
    "transform": ["--mode", "transform"],
}


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text())


def arm_video(clip: dict, arm: str, tag: str) -> Path:
    if arm == "original":
        return CLIPS / clip["dataset"] / f"{clip['id']}.mp4"
    return OUTPUTS / tag / arm / f"{clip['id']}.mp4"


def video_info(path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = dict(n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=cap.get(cv2.CAP_PROP_FPS),
                w=int(cap.get(3)), h=int(cap.get(4)))
    cap.release()
    return info


def read_frames(path, idxs=None) -> dict[int, np.ndarray]:
    """Decode sequentially (robust to inaccurate seeking). idxs=None -> all frames."""
    cap = cv2.VideoCapture(str(path))
    out, i = {}, 0
    want = None if idxs is None else set(int(x) for x in idxs)
    last = None if idxs is None else max(want)
    while True:
        ok, fr = cap.read()
        if not ok or (last is not None and i > last):
            break
        if want is None or i in want:
            out[i] = fr
        i += 1
    cap.release()
    return out


def uniform_idxs(n: int, k: int) -> list[int]:
    if n <= 0:
        return []
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def write_video(frames, path, fps):
    """frames: iterable of BGR uint8 arrays (same size). libx264 crf 17 yuv420p."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    it = iter(frames)
    first = next(it)
    h, w = first.shape[:2]
    cmd = [ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
           "-pix_fmt", "yuv420p", "-an", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(first.tobytes())
    for fr in it:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


# ----------------------------------------------------------------------------- faces
ARCFACE_SRC = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


class FaceModels:
    """YuNet detector (MIT), SFace (Apache-2.0) and ArcFace-R50 w600k (InsightFace
    buffalo_l, non-commercial research licence) -- evaluation-only attackers."""

    def __init__(self, gpu=True):
        import onnxruntime as ort
        self.yunet_path = str(MODELS / "face_detection_yunet_2023mar.onnx")
        self._det = {}
        self.sface = cv2.FaceRecognizerSF.create(str(MODELS / "face_recognition_sface_2021dec.onnx"), "")
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.arc = ort.InferenceSession(str(MODELS / "buffalo_l" / "w600k_r50.onnx"), so, providers=prov)
        self.arc_in = self.arc.get_inputs()[0].name

    def detect(self, img, thr=0.6):
        h, w = img.shape[:2]
        key = (w, h, thr)
        if key not in self._det:
            self._det[key] = cv2.FaceDetectorYN.create(self.yunet_path, "", (w, h), thr, 0.3, 5000)
        _, faces = self._det[key].detect(img)
        return np.zeros((0, 15), np.float32) if faces is None else faces

    @staticmethod
    def lm5(row) -> np.ndarray:
        return np.asarray(row[4:14], np.float32).reshape(5, 2)

    def arc_embed(self, img, rows) -> np.ndarray:
        """ArcFace embeddings for faces given as YuNet rows (landmarks used for alignment)."""
        if len(rows) == 0:
            return np.zeros((0, 512), np.float32)
        crops = []
        for r in rows:
            M, _ = cv2.estimateAffinePartial2D(self.lm5(r), ARCFACE_SRC, method=cv2.LMEDS)
            crops.append(cv2.warpAffine(img, M, (112, 112), borderValue=0.0))
        x = np.stack(crops)[..., ::-1].astype(np.float32)
        x = ((x - 127.5) / 127.5).transpose(0, 3, 1, 2)
        e = self.arc.run(None, {self.arc_in: np.ascontiguousarray(x)})[0]
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def sface_embed(self, img, rows) -> np.ndarray:
        out = []
        for r in rows:
            f = self.sface.feature(self.sface.alignCrop(img, np.asarray(r, np.float32)))
            out.append(f.ravel() / (np.linalg.norm(f) + 1e-9))
        return np.stack(out) if out else np.zeros((0, 128), np.float32)


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def xywh2xyxy(r):
    return [float(r[0]), float(r[1]), float(r[0] + r[2]), float(r[1] + r[3])]
