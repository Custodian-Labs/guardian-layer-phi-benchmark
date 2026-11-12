"""Independent EVAL-ONLY recogniser: InsightFace antelopev2 glintr100 (IResNet-100
trained on Glint360K; non-commercial research licence).  It is NOT the identity
space the swappers are driven by (those use buffalo_l w600k_r50), so it checks
that a low ArcFace-R50 score is not just the swapper optimising its own metric.
Threshold: FAR = 1e-3 on impostor pairs of the 118-identity VoxCeleb2 gallery
(glint_calib.py -> out/glint_threshold.json)."""
import json, os
from pathlib import Path
import cv2, numpy as np

P = os.path.expanduser(os.environ.get("GLINT_ONNX", "~/.cache/custodian_video/eval_models/antelopev2_dl/antelopev2/glintr100.onnx"))
SRC = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
THR_FILE = Path(__file__).resolve().parent / "out" / "glint_threshold.json"


class Glint:
    def __init__(self, gpu=False):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 8
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        self.s = ort.InferenceSession(P, so, providers=prov); self.i = self.s.get_inputs()[0].name

    def embed(self, img, rows):
        if len(rows) == 0:
            return np.zeros((0, 512), np.float32)
        cs = []
        for r in rows:
            M, _ = cv2.estimateAffinePartial2D(np.asarray(r[4:14], np.float32).reshape(5, 2), SRC, method=cv2.LMEDS)
            cs.append(cv2.warpAffine(img, M, (112, 112), borderValue=0.0))
        x = ((np.stack(cs)[..., ::-1].astype(np.float32) - 127.5) / 127.5).transpose(0, 3, 1, 2)
        e = self.s.run(None, {self.i: np.ascontiguousarray(x)})[0]
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def threshold():
    return json.loads(THR_FILE.read_text())["thr_far1e-3"]
