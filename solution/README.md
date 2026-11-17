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
`662f08d72b2374ba0b4c006e2b8a6376138a1dbd`. The same file comes out with numpy 2.4.4. The code
uses only numpy and the standard library; other versions should work but were not tested.

| file | what |
|---|---|
| `atd.py` | parses the raw ATD into arrays (metres, m/s) |
| `features.py` | per-frame features: activity, ball speed, ball outside the pitch, nearest player per team, touches |
| `model.py` | the decoder: unsupervised alive/dead model + H/A/D Viterbi + restart rules |
| `smart_tagging.py` | Smart Tagging loader |
| `whistle.py` | referee-whistle detector on the video's audio; experimental, only with `run.py --whistle` (needs `ffmpeg`), not used for the submission |
| `run.py` | entry point |
| `wo_ai_ni.md` | the technical document |
