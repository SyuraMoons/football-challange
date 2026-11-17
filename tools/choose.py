"""Choose decoder constants from the DFL grids. The rule is fixed before looking at the results:

1. Score of a setting = its mean spellF1 gain over the shipped constants, averaged over both ball
   models (challenge-like smoothed ball, raw DFL ball), the three references for 'controlled'
   possession (1.5, 2 and 3 s) and the seven matches.
2. Take the best setting by that score.
3. Keep it only if, with the challenge-like ball and the 2 s reference, it beats the shipped
   constants on at least 6 of 7 matches, and choosing by leave-one-match-out gains on at least
   6 of 7 held-out matches. Otherwise keep the shipped constants.

The challenge match plays no part in this.
"""
import json
import os

import numpy as np

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dfl")
SHIP = json.dumps(dict(CONTROL_R=1.5, C_SWITCH=9.0, FADE_R=3.5, TEMPER=0.15), sort_keys=True)
GRIDS = ("grid_smooth8", "grid_raw")
REFS = ("38", "50", "75")


def table(grid, ref):
    res = json.load(open(os.path.join(DIR, grid + ".json")))
    tags = sorted(res)
    ms = sorted(res[SHIP][ref])
    return tags, ms, np.array([[res[t][ref][m] for m in ms] for t in tags])


def main():
    tags = None
    gains = []
    for g in GRIDS:
        for r in REFS:
            t, ms, M = table(g, r)
            tags = tags or t
            assert t == tags
            gains.append(M - M[tags.index(SHIP)])
    G = np.array(gains)                                 # (conditions, settings, matches)
    score = G.mean(axis=(0, 2))
    order = np.argsort(-score)
    print("settings", len(tags))
    for i in order[:8]:
        print("  {:+.4f}  per condition {}  {}".format(score[i], " ".join("{:+.3f}".format(x) for x in G[:, i].mean(1)), tags[i]))
    best = order[0]
    _, ms, M = table("grid_smooth8", "50")
    s = tags.index(SHIP)
    up = int((M[best] > M[s]).sum())
    loo = []
    for k in range(len(ms)):
        rest = [j for j in range(len(ms)) if j != k]
        Gr = G[:, :, rest].mean(axis=(0, 2))
        pick = int(np.argmax(Gr))
        loo.append(M[pick, k] - M[s, k])
        print("  hold out {}: pick {} -> held-out gain {:+.4f}".format(ms[k], tags[pick], loo[-1]))
    loo_up = int((np.array(loo) > 0).sum())
    print("best {}: smoothed ball, 2 s reference: spellF1 {:.4f} vs shipped {:.4f}, up on {}/7; "
          "leave-one-out gain {:+.4f}, positive {}/7".format(tags[best], M[best].mean(), M[s].mean(), up, np.mean(loo), loo_up))
    print("DECISION:", tags[best] if up >= 6 and loo_up >= 6 else "keep shipped")


if __name__ == "__main__":
    main()
