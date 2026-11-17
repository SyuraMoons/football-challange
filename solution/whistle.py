"""Referee-whistle detector on the audio track of the tactical video.

A referee's whistle is a short, loud, narrow-band tone between about 3 and 4.5 kHz. Crowd noise
raises the whole band at once and crowd whistling is spread over many pitches, so the detector asks
for two things on a frame: the strongest few frequency bins stand far above their own recent
level, and they hold most of the energy of the band.

Needs `ffmpeg` on the PATH to decode the audio. Returns the first frame of every whistle.
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

SR = 12000                 # Hz; one video frame is exactly SR / 25 = 480 samples
N_FFT = 1024
BAND = (2900.0, 4500.0)    # Hz
LEVEL_HALF = 500           # frames either side for each bin's reference level (20 s)
LEVEL_STEP = 50
TOP_BINS = 3
PEAK_DB = 28.0             # strongest bins this far above their reference level
CONC_BINS = 6
CONC = 0.5                 # ... and the 6 strongest bins hold at least half the band energy
JOIN = 6                   # frames: detections closer than this are one blast
MIN_LEN, MAX_LEN = 4, 60   # frames: a blast lasts between 0.16 s and 2.4 s
INCIDENT_GAP = 50          # frames: blasts closer than this are one incident


def _audio(video):
    cmd = ["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16)


def _band_spectrogram(a, n_frames, hop):
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    keep = (freqs >= BAND[0]) & (freqs <= BAND[1])
    win = np.hanning(N_FFT).astype(np.float32)
    out = np.zeros((n_frames, int(keep.sum())), dtype=np.float32)
    offs = np.arange(N_FFT) - N_FFT // 2 + hop // 2
    for s in range(0, n_frames, 20000):
        e = min(n_frames, s + 20000)
        idx = np.clip((np.arange(s, e) * hop)[:, None] + offs[None, :], 0, len(a) - 1)
        x = a[idx].astype(np.float32) / 32768.0 * win
        out[s:e] = (np.abs(np.fft.rfft(x, axis=1)) ** 2)[:, keep]
    return out


def detect(game=GAME, cache=True):
    """Sorted array of video frames on which a whistle incident starts."""
    video = require(["video"], game)[0]
    path = os.path.join(os.path.dirname(video), "_cache_whistle.npy")
    if cache and os.path.exists(path):
        return np.load(path)
    n = source_frames("tactical", game)
    hop = SR // int(GAMES[game]["fps"])
    a = _audio(video)
    n_audio = min(n, len(a) // hop)
    X = _band_spectrogram(a, n_audio, hop)
    # each bin relative to its own median level over the surrounding 20 s
    idx = np.arange(0, n_audio, LEVEL_STEP)
    med = np.stack([np.median(X[max(0, i - LEVEL_HALF):i + LEVEL_HALF], axis=0) for i in idx])
    t = np.arange(n_audio)
    for j in range(X.shape[1]):
        X[:, j] /= np.interp(t, idx, med[:, j]) + 1e-14
    srt = np.sort(X, axis=1)
    peak_db = 10.0 * np.log10(srt[:, -TOP_BINS:].mean(axis=1) + 1e-9)
    conc = srt[:, -CONC_BINS:].sum(axis=1) / (X.sum(axis=1) + 1e-9)
    on = (peak_db >= PEAK_DB) & (conc >= CONC)
    d = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
    blasts = []
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if blasts and s - blasts[-1][1] <= JOIN:
            blasts[-1][1] = e
        else:
            blasts.append([s, e])
    starts, last_end = [], -10 ** 9
    for s, e in blasts:
        if not MIN_LEN <= e - s <= MAX_LEN:
            continue
        if s - last_end > INCIDENT_GAP:
            starts.append(s)
        last_end = e
    starts = np.array(starts, dtype=np.int64)
    if cache:
        np.save(path, starts)
    return starts


if __name__ == "__main__":
    w = detect(cache=False)
    print(len(w), "whistle incidents; first ten:", w[:10])
