"""Smart Tagging loader (standard library only): [{code, start_frame, end_frame, tags}]."""
import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402


def load(game=GAME):
    path = require(["smart_tagging"], game)[0]
    fps = GAMES[game]["fps"]
    out = []
    for inst in ET.parse(path).iter("instance"):
        out.append({
            "code": inst.findtext("code"),
            "start_frame": int(round(float(inst.findtext("start")) * fps)),
            "end_frame": int(round(float(inst.findtext("end")) * fps)),
            "tags": {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")},
        })
    return out
