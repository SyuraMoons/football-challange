"""Ground-truth-free summary of a labels file on the challenge match (no public window involved).

    python tools/gtfree.py evaluation/results/wo_ai_ni.csv [other.csv ...]

Prints possession and stoppage counts, label shares, and the Smart Tagging phase check of the
technical document (share of Build Up / Progression frames given to the tagged team).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "solution"), os.path.join(ROOT, "evaluation"), ROOT):
    sys.path.insert(0, p)
import smart_tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402


# phases in which the tagged team has the ball
BUILD = ("BUILD UP", "PROGRESSION")
ATTACK = BUILD + ("BALL IN FINAL THIRD", "PLAYERS IN THE BOX", "BALL IN THE BOX", "CREATING CHANCES",
                  "LONG BALL", "ATTACKING TRANSITION", "GOALS")


def summary(path, tags, runs):
    pr = pm.label_codes(pm.read_label_csv(path))
    segs = pm.segments(pr, runs)
    sp = [(a, b) for a, b, k in segs if k < 2]
    n = sum(b - a for a, b in runs)
    lens = np.array([b - a for a, b in sp]) / 25
    checks = []
    for codes in (BUILD, ATTACK):
        own = opp = tot = 0
        for t in tags:
            code, tm = t["code"].upper(), t["tags"].get("Team")
            if code not in codes or tm in (None, "N/A"):
                continue
            k = 0 if tm == "Barcelona" else 1
            seg = pr[t["start_frame"]:t["end_frame"]]
            own += (seg == k).sum(); opp += (seg == 1 - k).sum(); tot += len(seg)
        checks += [own / tot, opp / tot]
    sh = [sum(((pr[a:b]) == k).sum() for a, b in runs) / n for k in range(3)]
    print("{}: possessions {}, stoppages {}, median possession {:.1f} s, under 2 s {:.0%} | H {:.0%} A {:.0%} D {:.0%} | build-up tags own {:.3f} opp {:.3f} | attacking tags own {:.3f} opp {:.3f}".format(
        os.path.basename(path), len(sp), sum(1 for s in segs if s[2] == 2), np.median(lens), (lens < 2).mean(),
        *sh, *checks))


if __name__ == "__main__":
    runs = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    tags = smart_tagging.load()
    for p in sys.argv[1:]:
        summary(p, tags, runs)
