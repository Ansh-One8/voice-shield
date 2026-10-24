"""Score our own test set and print the results table.

labels.csv columns (header required):
    file,label,language,source
    clips/ansh_hi_01.wav,real,hi,team
    clips/ansh_hi_01_xtts.wav,fake,hi,xtts

label is 'real' or 'fake'. language is anything (hi, en, hinglish).

Usage:
    python tools/score_folder.py test_set/labels.csv
    python tools/score_folder.py test_set/labels.csv --modes clean phone --out results/scores.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vs import audio                       # noqa: E402
from vs.detector import Detector           # noqa: E402
from vs.metrics import compute_eer, rates_at  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--modes", nargs="+", default=["clean", "phone"], choices=["clean", "phone"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--out", default="results/scores.csv")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    rows = list(csv.DictReader(labels_path.open(encoding="utf-8")))
    det = Detector(args.checkpoint)
    rng = np.random.default_rng(0)

    out_rows = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        x = audio.load_audio(labels_path.parent / r["file"])
        for mode in args.modes:
            clip = audio.phone_simulate(x, rng=rng) if mode == "phone" else x
            res = det.score_clip(clip, trim=not args.no_trim)
            out_rows.append({**r, "mode": mode, "p_fake": f"{res.p_fake:.4f}", "verdict": res.label})
        print(f"\r{i}/{len(rows)} clips  ({time.time() - t0:.0f}s)", end="", flush=True)
    print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nScores saved to {out}\n")
    print(f"{'language':<10} {'mode':<6} {'real':>5} {'fake':>5} {'EER':>7} {'threshold':>9} "
          f"{'false alarms @0.5':>18}")
    groups = sorted({(r["language"], r["mode"]) for r in out_rows}) + \
        [("ALL", m) for m in args.modes]
    for lang, mode in groups:
        sel = [r for r in out_rows if r["mode"] == mode and (lang == "ALL" or r["language"] == lang)]
        real = np.array([float(r["p_fake"]) for r in sel if r["label"] == "real"])
        fake = np.array([float(r["p_fake"]) for r in sel if r["label"] == "fake"])
        if len(real) == 0 or len(fake) == 0:
            print(f"{lang:<10} {mode:<6} {len(real):>5} {len(fake):>5}   needs both real and fake clips")
            continue
        eer, thr = compute_eer(real, fake)
        fa = rates_at(real, fake, 0.5)["false_alarm_on_real"]
        print(f"{lang:<10} {mode:<6} {len(real):>5} {len(fake):>5} {eer * 100:6.2f}% {thr:9.3f} "
              f"{fa * 100:17.1f}%")


if __name__ == "__main__":
    main()
