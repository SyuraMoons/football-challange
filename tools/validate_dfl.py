"""Run the decoder on the public DFL matches and score it with the organisers' metric.

    python tools/validate_dfl.py                     # shipped constants, every match
    python tools/validate_dfl.py C_SWITCH=6,9,12     # one constant at several values

The DFL ball is tracked on every frame; the challenge ball is not. `hide_ball` removes the ball
the way the challenge tracker does (long gaps, mostly during stoppages), using only run-length
figures measured on the challenge tracking, never its labels. Scores are reported both ways.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "solution"), os.path.join(ROOT, "evaluation"), os.path.dirname(os.path.abspath(__file__))):
    sys.path.insert(0, p)
import dfl, features, model  # noqa: E402
import possession_metrics as pm  # noqa: E402

# ---- ball-loss simulation ----
# Challenge tracking, graded halves: the ball is missing on 41 % of frames, in about 75 gaps per
# half, almost all longer than 3 s (median about 10 s). The gaps sit on stoppages.
P_HIDE_DEAD = 0.85       # share of stoppages of 4 s or more during which the tracker loses the ball
LOSE = (-40, 75)         # frames from the stop to the loss of the ball (uniform)
FIND = (-75, 40)         # frames from the restart to the ball being found again (uniform)
MIN_GAP = 75             # shorter gaps do not occur
N_LIVE_GAPS = 6          # gaps per half that fall in open play
LIVE_GAP = (100, 300)    # their length in frames
BALL_SMOOTH = 0          # frames either side of a centred moving average on the ball (0 = raw DFL)


def runs_of(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def hide_ball(d, seed):
    rng = np.random.default_rng(seed)
    gt = d["gt"]
    hide = np.zeros(len(gt), dtype=bool)
    for a, b in d["blocks"]:
        for s, e in runs_of(gt[a:b + 1] == 2):
            if e - s < 100 or rng.random() > P_HIDE_DEAD:
                continue
            u, v = a + s + rng.integers(*LOSE), a + e + rng.integers(*FIND)
            if v - u >= MIN_GAP:
                hide[max(u, a):min(v, b + 1)] = True
        for _ in range(N_LIVE_GAPS):
            u = rng.integers(a, b)
            hide[u:min(u + rng.integers(*LIVE_GAP), b + 1)] = True
    out = dict(d)
    bx, by = smooth(d["bx"]), smooth(d["by"])
    out["bx"] = np.where(hide, np.nan, bx).astype(np.float32)
    out["by"] = np.where(hide, np.nan, by).astype(np.float32)
    return out


def smooth(x):
    """The challenge ball is smoother than the DFL one (lower frame-to-frame acceleration, about
    half the detectable touches, further from the nearest player). BALL_SMOOTH reproduces that."""
    if not BALL_SMOOTH:
        return x
    return features.rolling_nanmean(x.astype(np.float64), BALL_SMOOTH).astype(np.float32)


def kickoff_tags(d):
    """Kick-off windows as Smart Tagging gives them: a restart with the ball on the centre spot."""
    L, W = d["pitch"]
    gt, tags = d["gt"], []
    for a, b in d["blocks"]:
        starts = [a] + [a + s for s in np.flatnonzero((gt[a + 1:b + 1] != 2) & (gt[a:b] == 2)) + 1]
        for t in starts:
            if np.hypot(d["bx"][t] - L / 2, d["by"][t] - W / 2) < 1.5:
                tags.append(dict(code="SET PIECES", start_frame=max(a, t - 62), end_frame=min(b, t + 63),
                                 tags={"Type": "Kick Off Start" if t == a else "Kick Off"}))
    return tags


# The DFL possession flag changes team on every touch: about 8 possessions per minute of play,
# against about 4 in the challenge reference, whose definition asks for controlled possession and
# says a touch, a deflection or a duel is not a change. A flip shorter than MIN_CONTROL frames
# that is taken from the other team in open play is therefore given back to the team that had the
# ball. Scores are reported for several values, since the true threshold is not known.
MIN_CONTROL = (38, 50, 75)


def controlled(gt, min_len):
    gt = gt.copy()
    for a, b in pm.valid_runs(gt):
        t = a
        while t < b:
            e = t
            while e < b and gt[e] == gt[t]:
                e += 1
            if gt[t] != 2 and e - t < min_len and t > a and gt[t - 1] != 2:
                gt[t:e] = gt[t - 1]
            t = e
    return gt


SIM_KEYS = ("P_HIDE_DEAD", "MIN_GAP", "N_LIVE_GAPS", "BALL_SMOOTH")


def set_constants(cfg):
    """Decoder constants go to model / features, simulation settings to this module."""
    for k, v in cfg.items():
        if k in SIM_KEYS:
            globals()[k] = v
            continue
        mod = model if hasattr(model, k) else features
        if not hasattr(mod, k):
            raise KeyError(k)
        setattr(mod, k, v)


def one(job):
    match, seed, cfg, decode = job
    set_constants(cfg)
    full = dfl.load(match)
    d = full if seed is None else hide_ball(full, seed)
    blocks = [tuple(x) for x in d["blocks"]]
    f = features.compute(d, blocks)
    labels, _ = (decode or model.decode)(f, blocks, kickoff_tags(full), verbose=False)
    pred = pm.label_codes(labels)
    raw = d["gt"].astype(pred.dtype)
    out = dict(match=match, seed=seed, unseen=float((~f["vis"])[d["gt"] != 3].mean()))
    for m in MIN_CONTROL:
        r = pm.evaluate(controlled(raw, m), pred)
        out[m] = dict(spell=r["spell_f1_mean"], spell50=r["spell_f1_at_headline"],
                      macro=r["macro_f1"], n_gt=r["n_gt_spells"], n_pred=r["n_pred_spells"],
                      team=r["team_acc_both_alive"], ad=r["alive_dead_f1"],
                      p50=r["spell_f1"][0.5]["precision"], r50=r["spell_f1"][0.5]["recall"])
    return out


def run(cfg=None, seeds=(None, 1, 2), matches=None, decode=None, procs=7):
    """Mean scores per ball regime. seeds: None = ball always tracked, an int = simulated losses."""
    jobs = [(m, s, cfg or {}, decode) for m in (matches or dfl.matches()) for s in seeds]
    with Pool(procs, maxtasksperchild=1) as pool:     # no constant leaks from one job to the next
        return pool.map(one, jobs)


def summary(rows):
    out = {}
    for name, sel in (("full", lambda r: r["seed"] is None), ("hidden", lambda r: r["seed"] is not None)):
        rs = [r for r in rows if sel(r)]
        for m in MIN_CONTROL if rs else ():
            o = {k: float(np.mean([r[m][k] for r in rs])) for k in rs[0][m]}
            per = {}
            for r in rs:
                per.setdefault(r["match"], []).append(r[m]["spell"])
            o["per_match"] = [float(np.mean(v)) for _, v in sorted(per.items())]
            out[name, m] = o
    return out


def show(label, rows, base=None):
    """One line per ball regime and reference. With base, also the number of matches that improved."""
    sm, sb = summary(rows), summary(base) if base else None
    for (name, m), s in sm.items():
        extra = ""
        if sb:
            dif = np.array(s["per_match"]) - np.array(sb[name, m]["per_match"])
            extra = "  d {:+.3f} ({}/{} up)".format(dif.mean(), int((dif > 0).sum()), len(dif))
        print("{:22s} {:6s} ref {:.1f}s  spellF1 {:.3f} @0.5 {:.3f} (P {:.2f} R {:.2f})  macroF1 {:.3f}  spells {:.0f}/{:.0f}  team {:.3f}  a/d {:.3f}{}".format(
            label, name, m / 25, s["spell"], s["spell50"], s["p50"], s["r50"], s["macro"], s["n_pred"], s["n_gt"],
            s["team"], s["ad"], extra))


def main():
    base = run()
    show("shipped", base)
    for arg in sys.argv[1:]:
        name, vals = arg.split("=")
        for v in vals.split(","):
            show("{}={}".format(name, v), run({name: float(v) if "." in v else int(v)}), base)


if __name__ == "__main__":
    main()
