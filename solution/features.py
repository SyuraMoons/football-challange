"""Per-frame features computed from the tactical ATD. Everything is in metres and m/s."""
import numpy as np


def rolling_nanmean(x, half):
    """Centred nan-aware mean over +/- half frames."""
    ok = ~np.isnan(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    k = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - half)
    hi = np.minimum(len(x), i + half + 1)
    n = k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (c[hi] - c[lo]) / n, np.nan)


def compute(d):
    px, py, ps, team = d["px"], d["py"], d["ps"], d["team"]
    bx, by = d["bx"].astype(np.float64), d["by"].astype(np.float64)
    L, W = d["pitch"]
    n = len(bx)
    vis = ~np.isnan(bx)

    # collective activity: mean tracked-player speed, smoothed over one second
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # frames with nobody tracked
        act = np.nanmean(np.where(np.isnan(ps), np.nan, np.minimum(ps, 10.0)), axis=1)
    act = rolling_nanmean(act, 12)
    act = np.where(np.isnan(act), np.nanmedian(act), act)

    # ball speed from a centred difference over +/-2 frames (needs both ends visible)
    bs = np.full(n, np.nan)
    bs[2:-2] = np.hypot(bx[4:] - bx[:-4], by[4:] - by[:-4]) * 25.0 / 4.0
    bs_s = rolling_nanmean(bs, 6)

    # fastest the ball has moved within +/-1.5 s: a placed or hand-carried ball stays slow
    bmax = np.full(n, np.nan)
    filled = np.where(np.isnan(bs_s), -1.0, bs_s)
    half = 37
    pad = np.concatenate((np.full(half, -1.0), filled, np.full(half, -1.0)))
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    bmax = win.max(axis=1)
    bmax = np.where(bmax < 0, np.nan, bmax)

    # how far outside the pitch the ball is (0 inside)
    out = np.maximum.reduce([-bx, bx - L, -by, by - W, np.zeros(n)])
    out = np.where(vis, out, 0.0)

    # nearest player of each team to the ball
    dist = np.hypot(px - bx[:, None].astype(np.float32), py - by[:, None].astype(np.float32))
    dist = np.where(np.isnan(dist), np.inf, dist)
    dH = dist[:, team == 0].min(axis=1)
    dA = dist[:, team == 1].min(axis=1)

    touch = touches(bx, by, dH, dA)

    return dict(vis=vis, act=act, bs=bs_s, bmax=bmax, out=out, dH=dH, dA=dA, touch=touch, n=n)


TOUCH_K = 8          # frames either side over which the ball velocity is measured
TOUCH_DV = 4.0       # m/s: change of ball velocity that counts as a touch
TOUCH_SEP = 8        # frames: two touches closer than this are one
TOUCH_REACH = 3.0    # m: a player must have been this close for the touch to be theirs
TOUCH_MARGIN = 0.7   # m: the other team must be this much further away, or the touch is contested
TOUCH_WIN = 6        # frames either side in which the toucher is looked for


def _rolling_min(a, half):
    pad = np.concatenate((np.full(half, np.inf), a, np.full(half, np.inf)))
    return np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1).min(axis=1)


def touches(bx, by, dH, dA):
    """Signed touch events: +1 on a frame where a home player plays the ball, -1 away, 0 otherwise.

    The ball only changes velocity when somebody plays it (or it bounces). The tracked ball is
    smoothed over roughly half a second, so the velocity before and after a frame is taken over
    TOUCH_K frames and the toucher is the team that came closest within +/- TOUCH_WIN frames. A
    velocity change with nobody in reach (a bounce, a tracker jump) or with both teams equally
    close (a challenge) is not attributed to anyone."""
    n = len(bx)
    K = TOUCH_K
    step = np.full((n, 2), np.nan)
    step[K:, 0] = (bx[K:] - bx[:-K]) * 25.0 / K          # velocity over the K frames before
    step[K:, 1] = (by[K:] - by[:-K]) * 25.0 / K
    before = step
    after = np.full((n, 2), np.nan)
    after[:-K] = step[K:]
    dv = np.hypot(*(after - before).T)
    # the ball must be tracked on every frame of the window: a re-acquired ball is not a touch
    seen = ~np.isnan(bx)
    c = np.concatenate(([0], np.cumsum(seen)))
    i = np.arange(n)
    full = (c[np.minimum(n, i + K + 1)] - c[np.maximum(0, i - K)]) == 2 * K + 1
    x = np.where(full & ~np.isnan(dv), dv, 0.0)
    pad = np.concatenate((np.zeros(TOUCH_SEP), x, np.zeros(TOUCH_SEP)))
    peak = x == np.lib.stride_tricks.sliding_window_view(pad, 2 * TOUCH_SEP + 1).max(axis=1)
    mH, mA = _rolling_min(dH, TOUCH_WIN), _rolling_min(dA, TOUCH_WIN)
    hit = peak & (x > TOUCH_DV) & (np.minimum(mH, mA) < TOUCH_REACH)
    out = np.zeros(n)
    with np.errstate(invalid="ignore"):          # inf - inf where neither team is tracked
        out[hit & (mA - mH > TOUCH_MARGIN)] = 1.0
        out[hit & (mH - mA > TOUCH_MARGIN)] = -1.0
    return out
