"""Regenerate the submission:  python solution/run.py [--out path.csv]"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import atd, features, model, smart_tagging, whistle  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "wo_ai_ni.csv"))
    ap.add_argument("--whistle", action="store_true",
                    help="also use the referee-whistle detector (needs ffmpeg; not used for the submission)")
    args = ap.parse_args()
    d = atd.load()
    blocks = [(a, b) for _, a, b in half_video_blocks(GAME)]
    f = features.compute(d, blocks)
    tags = smart_tagging.load()
    labels, _ = model.decode(f, blocks, tags, whistle.detect() if args.whistle else ())
    with open(args.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    for a, b in blocks:
        seg = labels[a:b + 1]
        ch = np.flatnonzero(seg[1:] != seg[:-1])
        st = np.concatenate(([0], ch + 1))
        kinds = seg[st]
        print("half {}-{}: H {} A {} D {} | spells H {} A {} dead {}".format(
            a, b, (seg == "H").sum(), (seg == "A").sum(), (seg == "D").sum(),
            (kinds == "H").sum(), (kinds == "A").sum(), (kinds == "D").sum()))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
