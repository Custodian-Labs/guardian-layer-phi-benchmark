"""Stage 0: fetch raw public data + evaluation model weights (idempotent).

Everything lands under $VIDEO_EVAL_DATA (default /data/qbao775/custodianai/data/video_eval),
which is git-ignored. See custodian_video/eval/README.md for licences.

  python download.py            # all
  python download.py --only k400,mot17,vox2,models
"""
from __future__ import annotations

import argparse
import io
import time
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

DATA = Path(os.environ.get("VIDEO_EVAL_DATA", "/data/qbao775/custodianai/data/video_eval"))
RAW = DATA / "raw"
MODELS = DATA / "models"

MOT17_SEQS = ["02", "04", "05", "09", "10", "11", "13"]


def k400(shards=(0,)):
    # Kinetics-400 validation split, shards of 20 (~1k clips each), CVDF tarball mirror.
    # Shard 0 is the v0.1 set; more shards are only needed for --k400-extend in prepare.py.
    d = RAW / "k400"
    hf_hub_download("kiyoonkim/kinetics-400-targz", "annotations/val.csv", repo_type="dataset", local_dir=d)
    for k in shards:
        hf_hub_download("kiyoonkim/kinetics-400-targz", f"val/part_{k}.tar.gz", repo_type="dataset", local_dir=d)
        out = d / f"val_part{k}"
        if not out.exists():
            out.mkdir(parents=True)
            subprocess.run(["tar", "-xzf", str(d / f"val/part_{k}.tar.gz"), "-C", str(out)], check=True)


def mot17():
    """Only the frames of the 3 x 150-frame windows used by prepare.py are fetched
    (HF rate-limits per-file requests; we retry after the 5-min window)."""
    import time
    pats = []
    for s in MOT17_SEQS:
        L = MOT17_LEN[s]
        pats += [f"train/MOT17-{s}-FRCNN/gt/*", f"train/MOT17-{s}-FRCNN/seqinfo.ini"]
        for k in range(3):
            st = int(round(k * (L - 150) / 2)) + 1
            pats += [f"train/MOT17-{s}-FRCNN/img1/{f:06d}.jpg" for f in range(st, st + 150)]
    files = []
    for s in MOT17_SEQS:
        files += [f"train/MOT17-{s}-FRCNN/gt/gt.txt", f"train/MOT17-{s}-FRCNN/seqinfo.ini"]
    files += [p for p in pats if p.endswith(".jpg")]
    todo = [f for f in files if not (RAW / "mot17" / f).exists()]
    print(f"  {len(todo)} of {len(files)} MOT17 files to fetch", flush=True)
    if not todo:
        return
    # Official MOTChallenge zip, read remotely with HTTP range requests (only needed members).
    # Official MOTChallenge zip, read remotely with HTTP range requests (only needed members;
    # members are interleaved in the archive, so small blocks + 8 parallel readers).
    from concurrent.futures import ThreadPoolExecutor
    url = "https://motchallenge.net/data/MOT17.zip"
    todo = sorted(todo)

    def work(chunk):
        z = zipfile.ZipFile(_RangeFile(url, block=1 << 20))
        names = {n.split("/", 1)[1] if n.startswith("MOT17/") else n: n for n in z.namelist()}
        for f in chunk:
            dst = RAW / "mot17" / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            tmp.write_bytes(z.read(names[f]))
            tmp.rename(dst)
        return len(chunk)

    chunks = [todo[i::8] for i in range(8)]
    with ThreadPoolExecutor(8) as ex:
        print("  fetched", sum(ex.map(work, chunks)), flush=True)


class _RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests with an 8 MB read-ahead block."""

    def __init__(self, url, block=8 << 20):
        self.url, self.block, self.pos = url, block, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])
        self._buf_start, self._buf = -1, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _fetch(self, start):
        end = min(self.size, start + self.block) - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    self._buf_start, self._buf = start, r.read()
                return
            except Exception as e:
                print(f"  range fetch retry {attempt}: {e}", flush=True)
                time.sleep(5)
        raise RuntimeError("range fetch failed")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            if not (self._buf_start <= self.pos < self._buf_start + len(self._buf)):
                self._fetch(self.pos)
            o = self.pos - self._buf_start
            chunk = self._buf[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


MOT17_LEN = {"02": 600, "04": 1050, "05": 837, "09": 525, "10": 654, "11": 900, "13": 750}


def vox2():
    d = RAW / "vox2"
    for f in ["vox2/vox2_meta.csv", "vox2/vox2_test_mp4.zip"]:
        hf_hub_download("ProgramComputer/voxceleb", f, repo_type="dataset", local_dir=d)
    out = d / "test_mp4"
    if not out.exists():
        with zipfile.ZipFile(d / "vox2/vox2_test_mp4.zip") as z:
            z.extractall(out)


def models():
    MODELS.mkdir(parents=True, exist_ok=True)
    # ArcFace R50 (WebFace600K) from InsightFace buffalo_l: NON-COMMERCIAL RESEARCH ONLY.
    z = MODELS / "buffalo_l.zip"
    if not (MODELS / "buffalo_l" / "w600k_r50.onnx").exists():
        urllib.request.urlretrieve(
            "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip", z)
        with zipfile.ZipFile(z) as f:
            f.extractall(MODELS / "buffalo_l")
    # OpenCV-zoo YuNet (MIT) + SFace (Apache-2.0)
    for name, url in [
        ("face_detection_yunet_2023mar.onnx",
         "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
        ("face_recognition_sface_2021dec.onnx",
         "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx")]:
        p = MODELS / name
        cached = Path.home() / ".cache/custodian_video" / name
        if not p.exists():
            if cached.exists():
                p.write_bytes(cached.read_bytes())
            else:
                urllib.request.urlretrieve(url, p)
    for rid in ["MCG-NJU/videomae-base-finetuned-kinetics", "microsoft/xclip-base-patch32"]:
        snapshot_download(rid, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.bin"]
                          if "xclip" in rid else None)
    try:  # torchvision COCO Keypoint R-CNN weights (BSD-3 code; COCO-trained)
        import torchvision
        torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)
    except ImportError:
        print("torchvision not available in this env; weights fetched lazily at eval time")


STAGES = {"k400": k400, "mot17": mot17, "vox2": vox2, "models": models}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(STAGES))
    ap.add_argument("--k400-shards", default="0", help="K400 val shards to fetch, e.g. 0,1,2")
    a = ap.parse_args()
    for s in a.only.split(","):
        print(f"== {s}", flush=True)
        if s == "k400":
            k400(tuple(int(x) for x in a.k400_shards.split(",")))
        else:
            STAGES[s]()
        print(f"== {s} done", flush=True)
