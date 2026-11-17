"""Build a local page for checking the labels against the video by eye.

    python tools/review.py [--labels evaluation/results/wo_ai_ni.csv]

Writes data/preview/review.html (git-ignored, nothing leaves this machine). Open it in Chrome and
pick data/tactical/BAR-ATH_tactical.mp4 with the button. The page shows the decoded label under the
video, the public ground truth where it exists, a timeline, and a queue of moments worth a look.

The queue is built without ground truth, except for the public 5 minutes. It is a debugging aid:
watching is allowed, but what you see must not become labels, corrections or frame-specific rules.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "solution"), os.path.join(ROOT, "evaluation"), ROOT):
    sys.path.insert(0, p)
import atd, features, model, smart_tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

NAMES = "HAD"


def runs_of(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def build_queue(pr, gt, f, d, tags, blocks):
    q = []
    add = lambda cat, a, b, text: q.append(dict(cat=cat, a=int(a), b=int(b), text=text))
    full = [(a, b + 1) for a, b in blocks]
    pub = pm.valid_runs(gt)
    vis, L, W = f["vis"], d["pitch"][0], d["pitch"][1]
    ev = model.team_evidence(f)

    # 1. the public window, where the truth is known
    gs, ps = pm.possession_spells(gt, pub), pm.possession_spells(pr, pub)
    ov = pm.spell_overlaps(gs, ps)
    hit = pm.matched_gt_flags(ov, len(gs), 0.5)
    for (a, b, k), ok in zip(gs, hit):
        if not ok:
            add("public: true possession missed", a, b, "truth says {} for {:.1f}s".format(NAMES[k], (b - a) / 25))
    used = set()
    for iou, i, j in ov:
        if iou >= 0.5 and j not in used and hit[i]:
            used.add(j)
    for j, (a, b, k) in enumerate(ps):
        if j not in used:
            add("public: predicted possession unmatched", a, b, "we say {} for {:.1f}s".format(NAMES[k], (b - a) / 25))

    segs = pm.segments(pr, full)
    for i, (a, b, k) in enumerate(segs):
        prev = segs[i - 1] if i and segs[i - 1][1] == a else None
        nxt = segs[i + 1] if i + 1 < len(segs) and segs[i + 1][0] == b else None
        if k == 2:
            # 2. stoppage that begins with the ball vanishing well inside the pitch
            if prev and not (f["out"][max(a - 12, 0):a + 25] > 0.5).any() and vis[a:a + 25].mean() <= 0.5:
                seen = np.flatnonzero(vis[max(a - 250, 0):a + 1])
                if len(seen):
                    t = max(a - 250, 0) + seen[-1]
                    m = min(d["bx"][t], L - d["bx"][t], d["by"][t], W - d["by"][t])
                    if m > 5:
                        add("dead after ball lost inside pitch", a, b,
                            "stoppage {:.0f}s; ball last seen {:.0f} m from a line. Was play still on?".format((b - a) / 25, m))
            continue
        n = b - a
        if n < 50:
            add("possession under 2s", a, b, "{} for {:.1f}s. Real control or a touch?".format(NAMES[k], n / 25))
        elif n < 100 and nxt and nxt[2] == 2 and prev and prev[2] != 2:
            add("short possession then stoppage", a, b, "{} for {:.1f}s then dead. Tackle + whistle?".format(NAMES[k], n / 25))
        # 3. other-team control inside a possession that did not pay the switch cost
        sg = 1.0 if k == 0 else -1.0
        for s, e in runs_of(ev[a:b] * sg < -0.5):
            if e - s >= 15 or (ev[a:b] * sg < -0.5)[max(s - 25, 0):e + 25].sum() >= 20:
                add("other team near ball, no switch", a + s, a + e,
                    "{} keeps it, but {} controls for {} frames".format(NAMES[k], NAMES[1 - k], e - s))
                break
        # 4. play decoded on with no ball in the tracking
        for s, e in runs_of(~vis[a:b]):
            if e - s >= 75:
                add("alive with ball unseen 3s+", a + s, a + e, "{} for {:.1f}s with no ball tracked".format(NAMES[k], (e - s) / 25))

    # 5. Smart Tagging phases that name a team in possession, where we give the other one
    for t in tags:
        code, tm = t["code"].upper(), t["tags"].get("Team")
        if code not in ("BUILD UP", "PROGRESSION") or tm in (None, "N/A"):
            continue
        own = 0 if tm == "Barcelona" else 1
        seg = pr[t["start_frame"]:t["end_frame"]]
        if len(seg) and (seg == 1 - own).mean() > 0.5:
            add("tag says other team", t["start_frame"], t["end_frame"],
                "{} tagged for {}, we give {:.0%} to the other team".format(code.title(), NAMES[own], (seg == 1 - own).mean()))
    q.sort(key=lambda x: x["a"])
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(ROOT, "evaluation", "results", "wo_ai_ni.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "preview", "review.html"))
    args = ap.parse_args()
    d = atd.load()
    blocks = [(a, b) for _, a, b in half_video_blocks(GAME)]
    f = features.compute(d, blocks)
    pr = pm.label_codes(pm.read_label_csv(args.labels))
    gt = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", GAME + "_possession_gt_public.csv")))
    q = build_queue(pr, gt, f, d, smart_tagging.load(), blocks)
    data = dict(
        n=len(pr), fps=25,
        pred=[[a, b, k] for a, b, k in pm.segments(pr, [(a, b + 1) for a, b in blocks])],
        gt=[[a, b, k] for a, b, k in pm.segments(gt, pm.valid_runs(gt))],
        unseen=[[a, b] for a, b in runs_of(~f["vis"]) if b - a >= 25],
        queue=q, labels=os.path.basename(args.labels))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "review.html")) as fh:
        page = fh.read().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(page)
    cats = {}
    for x in q:
        cats[x["cat"]] = cats.get(x["cat"], 0) + 1
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print("{:4d}  {}".format(v, k))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
