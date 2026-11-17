"""Parse the raw tactical ATD into dense numpy arrays (cached as .npz inside the data dir)."""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402


def load(game=GAME, cache=True):
    """Returns dict: px, py, ps (n_frames, n_slots) in metres / m/s, bx, by (n_frames,) in metres,
    team (n_slots,) 0 home 1 away, pitch (L, W). NaN where missing."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    cache_path = os.path.join(os.path.dirname(raw_path), "_cache_atd.npz")
    if cache and os.path.exists(cache_path) and os.path.getmtime(cache_path) > os.path.getmtime(raw_path):
        z = np.load(cache_path)
        return {k: z[k] for k in z.files}

    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    slots = [p.get("teamId") for p in root.iter("Player")]
    team = np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)
    n = source_frames("tactical", game)
    k = len(team)
    L, W = GAMES[game]["pitch"]
    px = np.full((n, k), np.nan, dtype=np.float32)
    py = np.full((n, k), np.nan, dtype=np.float32)
    ps = np.full((n, k), np.nan, dtype=np.float32)
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            for s, g in enumerate(parts[1].split(";")):
                if s >= k or g.startswith("NaN"):
                    continue
                v = g.split(",")
                px[f, s] = float(v[0]) * L
                py[f, s] = float(v[1]) * W
                if len(v) > 2:
                    ps[f, s] = float(v[2])
            b = parts[2].split(",")
            if not b[0].startswith("NaN"):
                bx[f] = float(b[0]) * L
                by[f] = float(b[1]) * W
    out = dict(px=px, py=py, ps=ps, bx=bx, by=by, team=team, pitch=np.array([L, W]))
    if cache:
        np.savez(cache_path, **out)
    return out
