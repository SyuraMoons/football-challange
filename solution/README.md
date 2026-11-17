# Possession decoder — reproduction

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r solution/requirements.txt
# unpack the match bundle into ./data (see DATA.md) or set MTAH_DATA_DIR
.venv/bin/python solution/run.py --out wo_ai_ni.csv
.venv/bin/pip install -r requirements.txt            # only for the organisers' scorer
.venv/bin/python evaluation/validate_submission.py wo_ai_ni.csv
```

The run takes about 20 s on a laptop, uses no randomness, and reads only
`BAR-ATH_tactical_atd.{xml,txt}` and `BAR-ATH_tactical_smart_tagging.xml`.
The first run writes a parse cache, `data/tactical/_cache_atd.npz` (git-ignored with the rest of `data/`).

Built with Python 3.14.3 and numpy 2.5.5; SHA-1 of the file it produced there:
`ae3140e736a84d6ece909ae3915a5ce0e3252e7c`. The code uses only numpy and the standard library;
other versions should work but were not tested.

| file | what |
|---|---|
| `atd.py` | parses the raw ATD into arrays (metres, m/s) |
| `features.py` | per-frame features: activity, ball speed, ball outside the pitch, nearest player per team, touches |
| `model.py` | the decoder: unsupervised alive/dead model + H/A/D Viterbi + restart rules |
| `smart_tagging.py` | Smart Tagging loader |
| `whistle.py` | referee-whistle detector on the video's audio; experimental, only with `run.py --whistle` (needs `ffmpeg`), not used for the submission |
| `run.py` | entry point |
| `wo_ai_ni.md` | the technical document |

## Validation on other matches (`tools/`, not needed to produce the submission)

Download the seven matches of the public DFL dataset (figshare doi:10.6084/m9.figshare.28196177,
CC BY 4.0) into `data/dfl/` (`DFL_02_01_matchinformation_*.xml`, `DFL_04_03_positions_raw_observed_*.xml`).
Each script uses 7 processes; times are for a laptop.

```bash
.venv/bin/python tools/validate_dfl.py                           # current constants, every match (~2 min)
.venv/bin/python tools/validate_dfl.py C_SWITCH=6,12             # one constant at several values
.venv/bin/python tools/combo.py grid_smooth8 BALL_SMOOTH=8       # joint grid, challenge-like ball (~1 h)
.venv/bin/python tools/combo.py grid_raw                         # joint grid, raw DFL ball (~1 h)
.venv/bin/python tools/choose.py                                 # the selection rule of wo_ai_ni.md §3
.venv/bin/python tools/recheck_dfl.py                            # closer ball-loss simulation (~5 min)
.venv/bin/python tools/gtfree.py evaluation/results/wo_ai_ni.csv # ground-truth-free summary on this match
```

The grids set the four constants they search explicitly; `choose.py` compares against the previous
values (radii 1.5 / 3.5 m, alive/dead weight 0.15, switch cost 9). `tools/variants.py` holds the decoder variants of §2.1 that were
tried on DFL and dropped; `tools/review.py` builds a local page to watch the labels against the
video (it writes into the git-ignored `data/preview/`).
