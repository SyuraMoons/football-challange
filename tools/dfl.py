"""Load the public DFL matches (Bassek et al. 2025, CC BY, doi:10.6084/m9.figshare.28196177) in the
same array layout as solution/atd.py, with the per-frame ball status and possession as labels.

These are seven OTHER matches with official labels. They are used to measure the decoder and to
choose constants; nothing from the challenge match is involved.

    data/dfl/DFL_02_01_matchinformation_*.xml
    data/dfl/DFL_04_03_positions_raw_observed_*.xml
"""
import glob
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data"), "dfl")
GAP = 500   # frames of nothing between the halves, so no window reaches across the break

_SET = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" PersonId="([^"]*)"')
_FRM = re.compile(r'<Frame N="(\d+)" T="[^"]*" X="([-\d.]+)" Y="([-\d.]+)"(?: Z="[-\d.]+")? D="[-\d.]+" S="([-\d.]+)"')
_BALL = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


def matches():
    return sorted(p.split("_")[-1][:-4] for p in glob.glob(os.path.join(DIR, "DFL_04_03_positions_raw_observed_*.xml")))


def load(match):
    """dict with px, py, ps, bx, by, team, pitch as in atd.load, plus gt (0 H, 1 A, 2 D, 3 none)
    and blocks [(first, last)] for the two halves."""
    cache = os.path.join(DIR, "_cache_{}.npz".format(match))
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    info = open(glob.glob(os.path.join(DIR, "DFL_02_01_matchinformation_*{}.xml".format(match)))[0], encoding="utf-8").read()
    home = re.search(r'HomeTeamId="([^"]*)"', info).group(1)
    guest = re.search(r'GuestTeamId="([^"]*)"', info).group(1)
    L, W = (float(re.search(r'Pitch{}="([\d.]+)"'.format(a), info).group(1)) for a in "XY")
    rows = {}        # (section, person) -> list of (N, x, y, s)
    ball = {}        # section -> list of (N, x, y, possession, status)
    teams = {}
    cur = None
    with open(glob.glob(os.path.join(DIR, "DFL_04_03_positions_raw_observed_*{}.xml".format(match)))[0], encoding="utf-8") as fh:
        for line in fh:
            m = _FRM.match(line)
            if m:
                if cur is None:
                    continue
                if cur[1] == "BALL":
                    b = _BALL.search(line)
                    ball.setdefault(cur[0], []).append((int(m[1]), float(m[2]), float(m[3]), int(b[1]), int(b[2])))
                else:
                    rows[cur].append((int(m[1]), float(m[2]), float(m[3]), float(m[4])))
                continue
            m = _SET.match(line)
            if m:
                sec, tm, pid = m.groups()
                if tm == "BALL":
                    cur = (sec, "BALL")
                elif tm in (home, guest) and sec in ("firstHalf", "secondHalf"):
                    cur = (sec, pid)
                    rows.setdefault(cur, [])
                    teams[pid] = 0 if tm == home else 1
                else:
                    cur = None
    slots = sorted(teams)
    col = {p: i for i, p in enumerate(slots)}
    spans, start = {}, 0
    for sec in ("firstHalf", "secondHalf"):
        b = np.array(ball[sec])
        n0, n1 = int(b[:, 0].min()), int(b[:, 0].max())
        spans[sec] = (start, n0, n1 - n0 + 1)
        start += n1 - n0 + 1 + GAP
    n = start - GAP
    k = len(slots)
    px = np.full((n, k), np.nan, dtype=np.float32)
    py = np.full((n, k), np.nan, dtype=np.float32)
    ps = np.full((n, k), np.nan, dtype=np.float32)
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)
    gt = np.full(n, 3, dtype=np.int8)
    blocks = []
    for sec, (off, n0, ln) in spans.items():
        blocks.append((off, off + ln - 1))
        b = np.array(ball[sec])
        i = off + b[:, 0].astype(int) - n0
        bx[i] = b[:, 1] + L / 2
        by[i] = b[:, 2] + W / 2
        gt[i] = np.where(b[:, 4] == 0, 2, np.where(b[:, 3] == 1, 0, 1))
        for (s, pid), r in rows.items():
            if s != sec or not r:
                continue
            r = np.array(r)
            i = off + r[:, 0].astype(int) - n0
            ok = (i >= off) & (i < off + ln)
            px[i[ok], col[pid]] = r[ok, 1] + L / 2
            py[i[ok], col[pid]] = r[ok, 2] + W / 2
            ps[i[ok], col[pid]] = r[ok, 3] / 3.6          # km/h -> m/s
    out = dict(px=px, py=py, ps=ps, bx=bx, by=by, team=np.array([teams[p] for p in slots], dtype=np.int8),
               pitch=np.array([L, W]), gt=gt, blocks=np.array(blocks))
    np.savez(cache, **out)
    return out
