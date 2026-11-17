"""Possession decoder: a 3-state (H, A, D) hidden Markov model over the tactical tracking.

Two evidence streams:
  * alive vs dead -- collective player activity, ball visibility, ball outside the pitch, ball
    speed. The emission parameters are fitted WITHOUT labels by EM on the match itself.
  * home vs away  -- which team has a player in control distance of the ball. A ball that is
    travelling, or contested, is neutral evidence, so possession stays with the last team.
"""
import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])

# ---- fixed constants, never fitted to this match; CONTROL_R and FADE_R checked on other matches (DFL, wo_ai_ni.md §3) ----
TEMPER = 0.15          # consecutive frames are highly correlated: down-weight alive/dead evidence
OUT_MARGIN = 0.5       # m outside the lines before the ball counts as out
CONTROL_R = 0.75       # m: a player this close to the ball can be controlling it (DFL, §3)
FADE_R = 3.0           # m: beyond this, proximity says nothing (DFL, §3)
KAPPA = 0.6            # log-likelihood per frame of clear, uncontested control
PASS_SPEED = 7.0       # m/s: faster than a dribble
PLAYED_WIN = 12        # frames either side of a touch in which proximity counts in full
PASSBY = 0.0           # weight of proximity evidence for a fast ball nobody has touched
C_SWITCH = 9.0         # cost of a change of possession (about 0.6 s of clear control)
C_DEAD = 9.0           # cost of entering / leaving dead ball
OUT_CLEAR = 1.0        # m outside the lines at which the ball is certainly out
MIN_DEAD = 75          # frames (3 s): shortest stoppage we believe in
FORCE = 50.0           # per-frame penalty on 'alive' while the ball is certainly out
RESTART_LOOK = 100     # frames after a decoded restart in which the ball may still be placed
REST_SPEED = 1.0       # m/s: ball at rest
GO_SPEED = 2.5         # m/s: ball has left its rest
STRUCK_SPEED = 8.0     # m/s: ball has been struck, play is certainly on
WHISTLE_LIVE = 25      # frames before a whistle that must be live play for it to be a stoppage
WHISTLE_SPAN = 50      # frames after such a whistle over which play is penalised
WHISTLE_W = 9.0        # total penalty: the whistle pays the cost of entering dead ball, no more
KICKOFF_HOLD = 250     # frames (10 s) play is held alive after a kick-off while the ball is unseen
C_DSUB = 2.0           # cost of the tracker losing / finding the ball while it is dead


def _norm_logpdf(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd)


# alive/dead components: 0 = alive, 1 = dead with the ball in view (placed, carried, out of the
# pitch), 2 = dead with the ball lost by the tracker
N_COMP = 3


def alive_dead_loglik(f, p):
    """(n, 3) log-likelihood of the alive/dead features under [alive, dead-seen, dead-unseen]."""
    n = f["n"]
    ll = np.zeros((n, N_COMP))
    vis = f["vis"]
    is_out = f["out"] > OUT_MARGIN
    has_bs = vis & ~np.isnan(f["bmax"])
    lbs = np.log(np.where(has_bs, f["bmax"], 1.0) + 0.3)
    for s in range(N_COMP):
        ll[:, s] += _norm_logpdf(f["act"], p["act_mu"][s], p["act_sd"][s])
        ll[:, s] += np.where(vis, np.log(p["p_vis"][s]), np.log(1 - p["p_vis"][s]))
        ll[:, s] += np.where(vis, np.where(is_out, np.log(p["p_out"][s]), np.log(1 - p["p_out"][s])), 0.0)
        ll[:, s] += np.where(has_bs, _norm_logpdf(lbs, p["bs_mu"][s], p["bs_sd"][s]), 0.0)
    return ll, dict(is_out=is_out, lbs=lbs, has_bs=has_bs)


def _forward_backward(ll, T):
    """Posterior state probabilities of a Markov chain (scaled forward-backward)."""
    n, k = ll.shape
    b = np.exp(ll - ll.max(axis=1, keepdims=True))
    alpha = np.zeros((n, k)); beta = np.ones((n, k))
    a = b[0] / k; alpha[0] = a / a.sum()
    for t in range(1, n):
        a = (alpha[t - 1] @ T) * b[t]
        alpha[t] = a / a.sum()
    for t in range(n - 2, -1, -1):
        v = T @ (b[t + 1] * beta[t + 1])
        beta[t] = v / v.sum()
    g = alpha * beta
    return g / g.sum(axis=1, keepdims=True)


def fit_alive_dead(f, blocks, n_iter=20):
    """Unsupervised EM for the alive/dead emission model, over the graded halves only."""
    idx = np.concatenate([np.arange(a, b + 1) for a, b in blocks])
    sub = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in f.items()}
    sub["n"] = len(idx)
    # initial guess from football common sense: play is faster, the ball is usually seen and
    # moving; a dead ball is either lying/carried in view or out of the tracker's sight
    p = dict(act_mu=np.array([2.3, 1.1, 1.1]), act_sd=np.array([0.8, 0.5, 0.5]),
             p_vis=np.array([0.90, 0.95, 0.05]), p_out=np.array([0.005, 0.15, 0.15]),
             bs_mu=np.array([2.4, 1.0, 1.0]), bs_sd=np.array([0.6, 0.7, 0.7]))
    e = 1 / 250.0   # expected stretch of ~10 s; fixed, not learned
    T = np.array([[1 - 2 * e, e, e], [e, 1 - 2 * e, e], [e, e, 1 - 2 * e]])
    for _ in range(n_iter):
        ll, aux = alive_dead_loglik(sub, p)
        g = _forward_backward(ll * TEMPER, T)
        for s in range(N_COMP):
            w = g[:, s]
            p["act_mu"][s] = np.sum(w * sub["act"]) / w.sum()
            p["act_sd"][s] = max(0.25, np.sqrt(np.sum(w * (sub["act"] - p["act_mu"][s]) ** 2) / w.sum()))
            wv = w * sub["vis"]
            p["p_vis"][s] = np.clip(wv.sum() / w.sum(), 0.02, 0.98)
            p["p_out"][s] = np.clip(np.sum(wv * aux["is_out"]) / max(wv.sum(), 1e-9), 1e-3, 0.9)
            wb = w * aux["has_bs"]
            if wb.sum() > 50:
                p["bs_mu"][s] = np.sum(wb * aux["lbs"]) / wb.sum()
                p["bs_sd"][s] = max(0.3, np.sqrt(np.sum(wb * (aux["lbs"] - p["bs_mu"][s]) ** 2) / wb.sum()))
        p["share"] = g.mean(axis=0)
    return p


def team_evidence(f):
    """Signed per-frame evidence: > 0 favours home, < 0 away, 0 neutral (loose / travelling ball)."""
    dH, dA = f["dH"], f["dA"]
    near = np.minimum(dH, dA)
    with np.errstate(invalid="ignore"):
        gate = np.clip((FADE_R - near) / (FADE_R - CONTROL_R), 0.0, 1.0)
        margin = np.clip((dA - dH) / 1.0, -1.0, 1.0)
    e = np.where(f["vis"] & np.isfinite(near), gate * margin, 0.0)
    e = np.nan_to_num(e)
    # a ball travelling at passing speed that nobody has played is going past, not being controlled
    played = np.convolve(np.abs(f["touch"]), np.ones(2 * PLAYED_WIN + 1), mode="same") > 0
    passing = np.nan_to_num(f["bs"], nan=0.0) > PASS_SPEED
    return np.where(passing & ~played, PASSBY * e, e)



def viterbi(score, trans):
    n, k = score.shape
    back = np.zeros((n, k), dtype=np.int8)
    v = score[0].copy()
    for t in range(1, n):
        cand = v[:, None] + trans          # from i to j
        back[t] = cand.argmax(axis=0)
        v = cand.max(axis=0) + score[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = v.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def snap_switches(path, ev, thr=0.5):
    """Viterbi is indifferent to where a switch sits inside a neutral stretch. Possession changes
    when the new team gains control, so move each H<->A switch to the first frame of new-team
    evidence that follows the last frame of old-team evidence."""
    path = path.copy()
    sw = np.flatnonzero((path[1:] != path[:-1]) & (path[1:] != D) & (path[:-1] != D)) + 1
    for t in sw:
        old, new = path[t - 1], path[t]
        sign_new = 1.0 if new == H else -1.0
        a = t
        while a > 0 and path[a - 1] == old:
            a -= 1
        b = t
        while b < len(path) and path[b] == new:
            b += 1
        seg = ev[a:b] * sign_new
        old_idx = np.flatnonzero(seg[:t - a] < -thr)      # old-team control before the switch
        start = old_idx[-1] + 1 if len(old_idx) else 0
        new_idx = np.flatnonzero(seg[start:] > thr)
        if len(new_idx) == 0:
            continue
        cut = a + start + new_idx[0]
        path[a:cut] = old
        path[cut:b] = new
    return path


def _runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def forced_dead(f):
    """Frames where the ball is certainly dead: it is tracked clearly outside the pitch.

    The stoppage starts when the ball crosses the line and lasts at least MIN_DEAD frames after
    the ball was last seen outside. One exception: a ball that went over a touchline and is seen
    back inside the pitch sooner. Play restarts there with a throw-in, taken by hand from outside
    the line, so the ball coming back is the throw-in itself (or it was a high ball that only
    looked out) and the certainty ends on that frame. Behind a goal line the ball is fetched and
    placed for a goal kick or corner, so coming back in says nothing and the hold stays."""
    forced = np.zeros(f["n"], dtype=bool)
    outside = f["out"] > 0.0
    for a, b in zip(*_runs(outside)):
        if f["out"][a:b].max() > OUT_CLEAR and b - a >= 3:
            hold = MIN_DEAD
            if f["out_side"][a:b].max() >= f["out"][a:b].max():      # out over a touchline
                back = np.flatnonzero(f["vis"][b:b + MIN_DEAD])
                if len(back):
                    hold = back[0]
            forced[a:b + hold] = True
    return forced


def drop_short_dead(path):
    """A stoppage shorter than MIN_DEAD is not a stoppage: give it back to the surrounding play."""
    path = path.copy()
    for a, b in zip(*_runs(path == D)):
        if b - a >= MIN_DEAD:
            continue
        if a > 0:
            path[a:b] = path[a - 1]
        elif b < len(path):
            path[a:b] = path[b]
    return path


def refine_restarts(path, f_bs, f_vis):
    """A free kick, goal kick or corner is taken from a ball at rest. If the decoder resumes play
    while the ball is still being rolled into place -- it comes to rest within RESTART_LOOK frames
    without first being struck -- the restart is postponed until the ball leaves that rest."""
    path = path.copy()
    n = len(path)
    starts = np.flatnonzero((path[1:] != D) & (path[:-1] == D)) + 1
    for t in starts:
        end = t
        while end < n and path[end] != D:
            end += 1
        stop = min(end, t + RESTART_LOOK)
        bs = f_bs[t:stop]
        ok = f_vis[t:stop] & ~np.isnan(bs)
        struck = np.flatnonzero(ok & (bs > STRUCK_SPEED))
        limit = struck[0] if len(struck) else len(bs)
        rest = np.flatnonzero(ok[:limit] & (bs[:limit] < REST_SPEED))
        if len(rest) == 0:
            continue
        r = rest[-1]
        go = np.flatnonzero(ok[r:] & (bs[r:] > GO_SPEED))
        cut = t + r + (go[0] if len(go) else 1)
        if cut < end:
            path[t:cut] = D
    return path


def kickoff_constraints(n, blocks, tags, vis):
    """Smart Tagging marks every kick-off with a ~5 s window that brackets the kick. The ball is
    dead as the window opens and alive as it closes; and before the kick-off that starts a half
    there has been no play at all. If the tracker has not found the ball again by the end of the
    window, play is still on: it stays alive until the ball is seen, for at most KICKOFF_HOLD."""
    dead = np.zeros(n, dtype=bool)
    alive = np.zeros(n, dtype=bool)
    for e in tags:
        kind = e["tags"].get("Type", "").lower()
        if e["code"].upper() != "SET PIECES" or not kind.startswith("kick off"):
            continue
        s, t = e["start_frame"], e["end_frame"]
        dead[s:s + 25] = True
        alive[t - 25:t] = True
        unseen = np.flatnonzero(vis[t:t + KICKOFF_HOLD])
        alive[t:t + (unseen[0] if len(unseen) else KICKOFF_HOLD)] = True
        if "start" in kind:
            for a, b in blocks:
                if a <= t <= b:
                    dead[a:s + 25] = True
    return dead, alive


def whistle_penalty(path, whistles, offset, n):
    """Per-frame penalty on live play after each whistle blown while the ball was in play.

    A whistle during a stoppage is a restart signal or nothing; a whistle during play stops it.
    Which one it is comes from a first decode without the audio. The penalty only cancels the cost
    of entering dead ball, so the tracking still has to agree that play stopped."""
    pen = np.zeros(n)
    for w in whistles:
        t = w - offset
        if t < WHISTLE_LIVE or t >= n:
            continue
        if (path[t - WHISTLE_LIVE:t] != D).all():
            pen[t:t + WHISTLE_SPAN] = WHISTLE_W / WHISTLE_SPAN
    return pen


def decode(f, blocks, tags=(), whistles=(), params=None, verbose=True):
    """Labels for every frame of the video: H/A/D inside the graded halves, X elsewhere."""
    if params is None:
        params = fit_alive_dead(f, blocks)
    if verbose:
        for k, v in params.items():
            print("  {:7s} alive {:.3f}  dead-seen {:.3f}  dead-unseen {:.3f}".format(k, *v))
    ll, _ = alive_dead_loglik(f, params)
    ll = ll * TEMPER
    ev = team_evidence(f)
    ts = KAPPA * ev
    snap_ev = np.clip(ev + f["touch"], -1.0, 1.0)
    forced = forced_dead(f)
    ko_dead, ko_alive = kickoff_constraints(f["n"], blocks, tags, f["vis"])
    labels = np.full(f["n"], "X", dtype="<U1")
    # states: H, A, dead-seen, dead-unseen
    S, Dd, Dx = C_SWITCH, C_DEAD, C_DSUB
    trans = -np.array([[0, S, Dd, Dd],
                       [S, 0, Dd, Dd],
                       [Dd, Dd, 0, Dx],
                       [Dd, Dd, Dx, 0.0]])
    for a, b in blocks:
        sl = slice(a, b + 1)
        score = np.stack([ll[sl, 0] + ts[sl], ll[sl, 0] - ts[sl],
                          ll[sl, 1], ll[sl, 2]], axis=1)
        score[forced[sl] | ko_dead[sl], 0] -= FORCE
        score[forced[sl] | ko_dead[sl], 1] -= FORCE
        score[ko_alive[sl], 2] -= FORCE
        score[ko_alive[sl], 3] -= FORCE
        path = np.minimum(viterbi(score, trans), D).astype(np.int8)
        if len(whistles):
            pen = whistle_penalty(path, whistles, a, b + 1 - a)
            score[:, 0] -= pen
            score[:, 1] -= pen
            path = np.minimum(viterbi(score, trans), D).astype(np.int8)
        path = drop_short_dead(path)
        path = refine_restarts(path, f["bs"][sl], f["vis"][sl])
        path = snap_switches(path, snap_ev[sl])
        labels[sl] = LETTERS[path]
    return labels, params
