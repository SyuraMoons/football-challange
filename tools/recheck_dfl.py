"""Second look at the constants chosen by tools/choose.py, with the ball-loss simulation matched
more closely to the challenge tracking: the ball stays in view during about a third of stoppages
(P_HIDE_DEAD 0.65) and is lost more often in open play (20 gaps per half). Compares the shipped
radii and the chosen ones at several values of TEMPER, against the shipped constants.

    python tools/recheck_dfl.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_dfl as v  # noqa: E402

SIM = dict(BALL_SMOOTH=8, P_HIDE_DEAD=0.65, N_LIVE_GAPS=20)
RADII = (("shipped radii", dict(CONTROL_R=1.5, FADE_R=3.5)), ("chosen radii", dict(CONTROL_R=0.75, FADE_R=3.0)))


def per_match(rows, ref, key):
    d = {}
    for r in rows:
        d.setdefault(r["match"], []).append(r[ref][key])
    return np.array([np.mean(d[m]) for m in sorted(d)])


def main():
    res = {}
    for name, radii in RADII:
        for t in (0.07, 0.1, 0.15, 0.2):
            res[name, t] = v.run(dict(SIM, TEMPER=t, **radii), seeds=(1, 2))
    base = res["shipped radii", 0.15]
    for (name, t), rows in res.items():
        parts = []
        for ref in v.MIN_CONTROL:
            s, b = per_match(rows, ref, "spell"), per_match(base, ref, "spell")
            parts.append("ref {:.1f}s {:.3f} ({:+.3f}, {}/7 up)".format(ref / 25, s.mean(), (s - b).mean(), int((s > b).sum())))
        print("{:13s} TEMPER {:.2f}  alive/dead F1 {:.4f}  team {:.4f} | {}".format(
            name, t, per_match(rows, 50, "ad").mean(), per_match(rows, 50, "team").mean(), "  ".join(parts)), flush=True)


if __name__ == "__main__":
    main()
