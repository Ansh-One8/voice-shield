"""Equal Error Rate and related numbers for the results table."""

from __future__ import annotations

import numpy as np


def compute_eer(p_fake_real: np.ndarray, p_fake_fake: np.ndarray) -> tuple[float, float]:
    """EER and the threshold where it happens.

    p_fake_real: fake-probabilities given to real (bonafide) clips
    p_fake_fake: fake-probabilities given to AI (spoof) clips
    Returns (eer as a fraction, threshold on p_fake).
    """
    scores = np.concatenate([p_fake_real, p_fake_fake])
    labels = np.concatenate([np.zeros(len(p_fake_real)), np.ones(len(p_fake_fake))])
    order = np.argsort(scores)
    scores, labels = scores[order], labels[order]
    n_real, n_fake = len(p_fake_real), len(p_fake_fake)
    # threshold just above scores[i]: everything <= scores[i] is called real
    false_alarm = 1 - np.cumsum(labels == 0) / n_real     # real clips called fake
    miss = np.cumsum(labels == 1) / n_fake                # fake clips called real
    i = int(np.argmin(np.abs(false_alarm - miss)))
    return float((false_alarm[i] + miss[i]) / 2), float(scores[i])


def rates_at(p_fake_real: np.ndarray, p_fake_fake: np.ndarray, threshold: float) -> dict:
    """False-alarm and miss rates if we call 'fake' when p_fake > threshold."""
    return {
        "false_alarm_on_real": float(np.mean(p_fake_real > threshold)),
        "missed_fakes": float(np.mean(p_fake_fake <= threshold)),
    }
