"""Joint grid over the constants the single-constant sweep flagged, on the DFL matches with the
ball hidden and smoothed like the challenge tracking. Results go to data/dfl/<name>.json.

    python tools/combo.py NAME SIM_KEY=value ...
"""
import itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import validate_dfl as v

GRID = dict(FADE_R=(2.0, 2.5, 3.0, 3.5), CONTROL_R=(0.75, 1.0, 1.5), TEMPER=(0.07, 0.1, 0.15), C_SWITCH=(9.0, 12.0, 15.0))
SEEDS = (1, 2)
REF = 50

if __name__ == "__main__":
    name, sim = sys.argv[1], {}
    for a in sys.argv[2:]:
        k, x = a.split("=")
        sim[k] = float(x) if "." in x else int(x)
    out = os.path.join(v.dfl.DIR, name + ".json")
    res = json.load(open(out)) if os.path.exists(out) else {}
    keys = list(GRID)
    for vals in itertools.product(*GRID.values()):
        cfg = dict(zip(keys, vals))
        tag = json.dumps(cfg, sort_keys=True)
        if tag in res:
            continue
        rows = v.run(dict(sim, **cfg), seeds=SEEDS)
        per = {}
        for r in rows:
            for ref in v.MIN_CONTROL:
                per.setdefault(str(ref), {}).setdefault(r["match"], []).append(r[ref]["spell"])
        res[tag] = {ref: {m: float(np.mean(x)) for m, x in d.items()} for ref, d in per.items()}
        json.dump(res, open(out, "w"))
        print(tag, "%.4f" % np.mean(list(res[tag][str(REF)].values())), flush=True)
