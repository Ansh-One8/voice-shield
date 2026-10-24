"""High-level deepfake detector used by the server and the scoring scripts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import audio
from .model import download_checkpoint, load_sls


@dataclass
class Thresholds:
    """Fake-probability cut-offs. Between the two the verdict is 'unsure'.

    These are starting values. Replace them with the threshold the evaluation
    script reports for our own test set.
    """
    real_below: float = 0.30
    fake_above: float = 0.70

    def label(self, p_fake: float) -> str:
        if p_fake >= self.fake_above:
            return "fake"
        if p_fake <= self.real_below:
            return "real"
        return "unsure"


def load_calibration(path: Path) -> float | None:
    """Cut-off saved by tools/set_threshold.py next to the model, if any."""
    env = os.environ.get("VS_CALIB_THRESHOLD")
    if env:
        return float(env)
    if path.exists():
        return float(json.loads(path.read_text())["threshold"])
    return None


@dataclass
class ClipResult:
    p_fake: float                      # mean over windows
    label: str
    window_p_fake: list[float] = field(default_factory=list)
    window_starts_s: list[float] = field(default_factory=list)
    seconds: float = 0.0


class Detector:
    def __init__(self, checkpoint: str | None = None, device: str | None = None,
                 thresholds: Thresholds | None = None, threads: int | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if threads:
            torch.set_num_threads(threads)
        # newest model first: run B's final model, then run A's, then the ready-made one
        models = Path(__file__).resolve().parents[1] / "models"
        local = next((models / n for n in ("voice_shield_final.pth", "voice_shield_ft.pth", "sls_v1_epoch2.pth")
                      if (models / n).exists()), None)
        checkpoint = (checkpoint or os.environ.get("VS_CHECKPOINT")
                      or (str(local) if local else download_checkpoint()))
        self.checkpoint = checkpoint
        self.model = load_sls(checkpoint, self.device)
        self.thresholds = thresholds or Thresholds()
        self.target_dbfs = float(os.environ.get("VS_TARGET_DBFS", "-23"))
        self.calib_path = Path(checkpoint).with_suffix(".calib.json")
        self.calib_threshold = load_calibration(self.calib_path)

    def calibrate(self, p_raw: np.ndarray) -> np.ndarray:
        """Re-centre raw scores so the measured best cut-off becomes 0.5.

        The fine-tuned model squeezes its scores near 0 (its best cut-off can be
        ~0.005). Shifting in log-odds keeps the ranking but makes 50% mean
        "at the cut-off", so the app's real / unsure / AI bands work again.
        """
        if self.calib_threshold is None:
            return p_raw
        p = np.clip(p_raw, 1e-7, 1 - 1e-7)
        t = self.calib_threshold
        z = np.log(p / (1 - p)) - np.log(t / (1 - t))
        return 1 / (1 + np.exp(-z))

    @torch.inference_mode()
    def p_fake_windows(self, wins: list[np.ndarray], batch: int = 4, raw: bool = False) -> np.ndarray:
        """Probability of 'AI voice' for each 64600-sample window (calibrated unless raw=True).

        Every window is set to the same loudness first. The ready-made model
        calls quiet audio "AI" (a loudness shortcut), and laptop mics are quiet.
        """
        wins = [audio.normalize_loudness(w, self.target_dbfs) for w in wins]
        out = []
        for i in range(0, len(wins), batch):
            x = torch.from_numpy(np.stack(wins[i:i + batch])).float().to(self.device)
            out.append(self.model(x)[:, 0].exp().cpu().numpy())   # index 0 = fake; exact for tiny scores
        p = np.concatenate(out)
        return p if raw else self.calibrate(p)

    def prepare(self, x: np.ndarray, trim: bool = True, loudness: bool = False) -> np.ndarray:
        if trim:
            x = audio.trim_silence(x)
        if loudness:
            x = audio.normalize_loudness(x)
        return x

    def score_clip(self, x: np.ndarray, trim: bool = True, loudness: bool = False,
                   phone: bool = False, hop_s: float = 1.0,
                   max_windows: int | None = None) -> ClipResult:
        """Score a 16 kHz clip of any length.

        Long clips use sliding windows; max_windows limits how many (spread
        evenly over the clip) so long uploads stay fast on a CPU.
        """
        if phone:
            x = audio.phone_simulate(x)
        x = self.prepare(x, trim, loudness)
        hop = int(hop_s * audio.SR)
        starts = audio.window_starts(len(x), hop=hop, max_windows=max_windows)
        wins = audio.windows(x, hop=hop, max_windows=max_windows)
        p = self.p_fake_windows(wins)
        # average in log-odds: confident windows count fully, so a few weak
        # seconds can't hide an AI clip (tested on our 36 test clips + new TTS clips)
        q = np.clip(p, 1e-6, 1 - 1e-6)
        mean = float(1 / (1 + np.exp(-np.mean(np.log(q / (1 - q))))))
        return ClipResult(
            p_fake=mean,
            label=self.thresholds.label(mean),
            window_p_fake=[float(v) for v in p],
            window_starts_s=[round(s / audio.SR, 2) for s in starts],
            seconds=len(x) / audio.SR,
        )
