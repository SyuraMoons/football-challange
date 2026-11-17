"""Candidate decoder changes, tried on the DFL matches before any of them goes into solution/.

    python tools/variants.py C_LOSE=2,4,6 LIVE_GAPS=12
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_dfl as v  # noqa: E402
from validate_dfl import dfl, model  # noqa: E402

C_LOSE = 4.0        # cost of the tracker losing the ball while play is on
C_FIND = 0.0        # cost of finding it again
BIG = 1e3


def decode6(f, blocks, tags=(), whistles=(), params=None, verbose=True):
    """model.decode with two more states: H and A with the ball lost by the tracker.

    In the shipped model a missing ball is an independent piece of evidence for 'dead' on every
    frame, so a gap of a few seconds outweighs anything the players do. Here losing the ball is an
    event with a one-off cost, as it already is for the dead states, and during the gap only the
    players' activity decides between play and stoppage."""
    m = model
    if params is None:
        params = m.fit_alive_dead(f, blocks)
    ll, _ = m.alive_dead_loglik(f, params)
    ll = ll * m.TEMPER
    vis = f["vis"]
    alive = ll[:, 0] - m.TEMPER * np.where(vis, np.log(params["p_vis"][0]), np.log(1 - params["p_vis"][0]))
    ev = m.team_evidence(f)
    ts = m.KAPPA * ev
    snap_ev = np.clip(ev + f["touch"], -1.0, 1.0)
    forced = m.forced_dead(f)
    ko_dead, ko_alive = m.kickoff_constraints(f["n"], blocks, tags, vis)
    labels = np.full(f["n"], "X", dtype="<U1")
    S, Dd, Dx, Cl, Cf = m.C_SWITCH, m.C_DEAD, m.C_DSUB, C_LOSE, C_FIND
    # states: H, A, dead-seen, dead-unseen, H-unseen, A-unseen
    trans = -np.array([[0, S, Dd, Dd, Cl, Cl + S],
                       [S, 0, Dd, Dd, Cl + S, Cl],
                       [Dd, Dd, 0, Dx, Dd, Dd],
                       [Dd, Dd, Dx, 0, Dd, Dd],
                       [Cf, S + Cf, Dd, Dd, 0, S],
                       [S + Cf, Cf, Dd, Dd, S, 0.0]])
    fold = np.array([0, 1, 2, 2, 0, 1], dtype=np.int8)
    for a, b in blocks:
        sl = slice(a, b + 1)
        seen = vis[sl]
        score = np.stack([alive[sl] + ts[sl] - BIG * ~seen, alive[sl] - ts[sl] - BIG * ~seen,
                          ll[sl, 1], ll[sl, 2],
                          alive[sl] - BIG * seen, alive[sl] - BIG * seen], axis=1)
        stop = forced[sl] | ko_dead[sl]
        score[np.ix_(stop, [0, 1, 4, 5])] -= m.FORCE
        score[np.ix_(ko_alive[sl], [2, 3])] -= m.FORCE
        path = fold[m.viterbi(score, trans)]
        path = m.drop_short_dead(path)
        path = m.refine_restarts(path, f["bs"][sl], vis[sl])
        path = m.snap_switches(path, snap_ev[sl])
        labels[sl] = m.LETTERS[path]
    return labels, params


VIS_W = 1.0         # share of the 'ball missing' evidence against live play that is kept
KNOBS = ("C_LOSE", "C_FIND", "VIS_W")
DEFAULTS = dict(C_LOSE=C_LOSE, C_FIND=C_FIND, VIS_W=VIS_W)


def decode_vw(f, blocks, tags=(), whistles=(), params=None, verbose=True):
    """model.decode with the evidence a missing ball gives against live play scaled by VIS_W."""
    m = model
    if params is None:
        params = m.fit_alive_dead(f, blocks)
    orig = m.alive_dead_loglik

    def patched(ff, p):
        ll, aux = orig(ff, p)
        ll[:, 0] -= (1 - VIS_W) * np.where(ff["vis"], 0.0, np.log(1 - p["p_vis"][0]))
        return ll, aux
    m.alive_dead_loglik = patched
    try:
        return m.decode(f, blocks, tags, whistles, params, verbose)
    finally:
        m.alive_dead_loglik = orig


def job(a):
    match, seed, cfg, name = a
    import variants as me          # by name: under spawn the script itself is __mp_main__ in a worker
    for k, val in cfg.items():
        mod = me if k in KNOBS else v if hasattr(v, k) else None
        if mod is not None:
            setattr(mod, k, val)
    rest = {k: val for k, val in cfg.items() if k not in KNOBS and not hasattr(v, k)}
    try:
        return v.one((match, seed, rest, getattr(me, name) if name else None))
    finally:                       # workers are reused: put every knob back
        for k in cfg:
            if k in KNOBS:
                setattr(me, k, DEFAULTS[k])


def run(cfg=None, seeds=(None, 1, 2), name="decode6"):
    with Pool(7, maxtasksperchild=1) as pool:
        return pool.map(job, [(mt, s, cfg or {}, name) for mt in dfl.matches() for s in seeds])


def grid(name, sims, cfgs, seeds=(1, 2)):
    """Every cfg under every simulation setting, against the shipped decoder under the same one."""
    for sim in sims:
        base = run(dict(sim), seeds, None)
        print("--", sim)
        v.show("shipped", base)
        for cfg in cfgs:
            v.show(" ".join("{}={}".format(k, x) for k, x in cfg.items()), run(dict(sim, **cfg), seeds, name), base)


if __name__ == "__main__":
    grid("decode_vw", [dict(N_LIVE_GAPS=n) for n in (6, 24)], [dict(VIS_W=w) for w in (0.0, 0.25, 0.5, 0.75)])
