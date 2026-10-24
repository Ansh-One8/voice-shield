"""Save the cut-off for a model, so the app's real / unsure / AI bands line up with it.

Use the "threshold" the Kaggle notebook printed for the dev groups (the score
where false alarms and missed fakes are equal). Scores are re-centred so that
cut-off shows as 50% in the app.

Usage:
    python tools/set_threshold.py 0.008                       # newest model in models/
    python tools/set_threshold.py 0.008 models/voice_shield_ft.pth
    python tools/set_threshold.py --clear                     # back to raw scores
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"


def newest_model() -> Path:
    for n in ("voice_shield_final.pth", "voice_shield_ft.pth", "sls_v1_epoch2.pth"):
        if (MODELS / n).exists():
            return MODELS / n
    raise SystemExit("No model file in models/.")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    model = Path(args[1]) if len(args) > 1 else newest_model()
    calib = model.with_suffix(".calib.json")
    if args[0] == "--clear":
        calib.unlink(missing_ok=True)
        print(f"Removed {calib.name}: the app shows raw scores for {model.name}.")
        return
    t = float(args[0])
    if not 0 < t < 1:
        raise SystemExit("The threshold must be between 0 and 1.")
    calib.write_text(json.dumps({"threshold": t}, indent=1))
    print(f"Saved {calib.name}: a raw score of {t} now shows as 50% for {model.name}. Restart the server.")


if __name__ == "__main__":
    main()
