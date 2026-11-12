"""Dump the person-context filter's inputs (face tracks, person boxes) for one clip,
using the pipeline's own pass 1 (faces channel only), so filter variants can be
compared offline.  Usage: dump_fp_inputs.py <input.mp4> <out.pkl>"""
import pickle
import sys

from custodian_video import pipeline as P


class _Stop(Exception):
    pass


def main(inp, outp):
    orig = P._face_fp_filter

    def hook(inp_, ft, persons, person_det, p, frame_wh, log):
        pickle.dump({"inp": inp_, "ft": ft, "persons": persons, "frame_wh": frame_wh}, open(outp, "wb"))
        raise _Stop()
    P._face_fp_filter = hook
    try:
        P.run(inp, "/dev/null", P.Config(mode="redact", channels=("faces",)))
    except _Stop:
        print("dumped", outp)
    finally:
        P._face_fp_filter = orig


if __name__ == "__main__":
    main(*sys.argv[1:3])
