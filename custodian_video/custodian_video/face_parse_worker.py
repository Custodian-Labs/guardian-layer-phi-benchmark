"""Worker for face_parse.py: MediaPipe "selfie multiclass" image segmenter
(Apache-2.0 runtime; model card: selfie_multiclass_256x256, Apache-2.0) run under
a Python that has `mediapipe` (the custodian-video env does not).

Protocol (binary, stdin/stdout): request = uint32 n (0 = quit) followed by n
256x256x3 BGR uint8 images; reply = n x 256x256x3 uint8 confidences for
[face-skin, hair, others(accessories: glasses frames, earrings)] x 255.
  python face_parse_worker.py <model.tflite>
"""
import struct
import sys

import numpy as np


def main():
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=mpt.BaseOptions(model_asset_path=sys.argv[1]), output_confidence_masks=True,
        output_category_mask=False))
    rin, wout = sys.stdin.buffer, sys.stdout.buffer
    S = 256 * 256 * 3
    while True:
        hdr = rin.read(4)
        if len(hdr) < 4:
            break
        n = struct.unpack("<I", hdr)[0]
        if n == 0:
            break
        buf = rin.read(n * S)
        ims = np.frombuffer(buf, np.uint8).reshape(n, 256, 256, 3)
        out = np.empty((n, 256, 256, 3), np.uint8)
        for i in range(n):
            r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(ims[i, :, :, ::-1])))
            cm = r.confidence_masks
            for j, k in enumerate((3, 1, 5)):    # face-skin, hair, others
                out[i, :, :, j] = np.clip(np.squeeze(cm[k].numpy_view()) * 255 + 0.5, 0, 255)
        wout.write(out.tobytes())
        wout.flush()
    seg.close()


if __name__ == "__main__":
    main()
