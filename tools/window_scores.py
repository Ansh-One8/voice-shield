"""Score every window of many clips in several input variants, for tuning the decision rule.

Saves results/window_scores.json:
  {clip_path: {"label": real|fake|?, "variant": {name: [raw p_fake per window]}}}

Usage:
    python tools/window_scores.py
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vs import audio                 # noqa: E402
from vs.detector import Detector     # noqa: E402

DOWNLOADS = Path.home() / "Downloads"
OUT = ROOT / "results" / "window_scores.json"


def band8k(x):
    return audio.resample(audio.resample(x, audio.SR, audio.PHONE_SR), audio.PHONE_SR, audio.SR)


VARIANTS = {"as_is": lambda x: x, "band8k": band8k, "phone": lambda x: audio.phone_simulate(x, rng=np.random.default_rng(0))}


def clips() -> list[tuple[str, str]]:
    out = []
    for f in sorted(DOWNLOADS.glob("*.wav")):
        n = f.stem.lower()
        if n.startswith(("human_mic", "ai_mic", "recording_")):
            label = "real" if n.startswith("human_mic") else "fake" if n.startswith("ai_mic") else "?"
            out.append((str(f), label))
    rows = list(csv.DictReader(open(ROOT / "kaggle_upload/vs_recordings/vs_labels.csv", encoding="utf-8")))
    for r in rows:
        if r["split"] == "test":
            out.append((str(ROOT / "kaggle_upload/vs_recordings" / r["file"]), r["label"]))
    for f in glob.glob(str(ROOT / "test_set/sanity/real_*")):
        out.append((f, "real"))
    return out


def main() -> None:
    d = Detector()
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [(p, l) for p, l in clips() if p not in done]
    print(f"{len(todo)} clips to score")
    for i, (p, label) in enumerate(todo, 1):
        x = audio.trim_silence(audio.load_audio(p))
        rec = {"label": label, "variant": {}}
        for name, fn in VARIANTS.items():
            wins = audio.windows(fn(x), hop=audio.SR)
            rec["variant"][name] = [float(v) for v in d.p_fake_windows(wins, raw=True)]
        done[p] = rec
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(done))
        print(f"{i}/{len(todo)} {Path(p).name} {label}", flush=True)


if __name__ == "__main__":
    main()
