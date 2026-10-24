"""Training-time augmentation.

Goal: the model must not be able to use recording conditions (volume,
silence, microphone, phone line, room, noise) to tell real from fake. Every
augmentation is applied at random to real and fake clips alike, so these
conditions carry no label information.

The last two steps always run, because the detector does the same at
inference: trim silence at the ends, and set the window to -23 dBFS.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from . import audio


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.normal(size=n))
    f = np.arange(len(spec))
    spec /= np.sqrt(np.maximum(f, 1))
    y = np.fft.irfft(spec, n)
    return (y / (np.std(y) + 1e-9)).astype(np.float32)


def add_noise(x: np.ndarray, rng: np.random.Generator, snr_db: tuple = (8, 30)) -> np.ndarray:
    noise = _pink_noise(len(x), rng) if rng.random() < 0.5 else rng.normal(size=len(x)).astype(np.float32)
    p = np.mean(x ** 2) + 1e-12
    snr = rng.uniform(*snr_db)
    return x + noise * np.sqrt(p / 10 ** (snr / 10))


def add_reverb(x: np.ndarray, rng: np.random.Generator, sr: int = audio.SR) -> np.ndarray:
    """Synthetic room: exponentially decaying noise as the impulse response."""
    rt60 = rng.uniform(0.15, 0.7)
    n = int(sr * rt60)
    t = np.arange(n) / sr
    rir = rng.normal(size=n) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0
    rir /= np.sqrt(np.sum(rir ** 2))
    wet = fftconvolve(x, rir)[:len(x)]
    mix = rng.uniform(0.3, 1.0)
    return ((1 - mix) * x + mix * wet).astype(np.float32)


def random_crop(x: np.ndarray, rng: np.random.Generator, length: int = audio.WINDOW) -> np.ndarray:
    if len(x) <= length:
        return audio.repeat_pad(x, length)
    s = rng.integers(0, len(x) - length + 1)
    return x[s:s + length]


def augment(x: np.ndarray, rng: np.random.Generator, p_phone: float = 0.5,
            p_noise: float = 0.3, p_reverb: float = 0.2, p_clip: float = 0.1) -> np.ndarray:
    """One random training view of a 16 kHz clip, ready for the model."""
    x = audio.trim_silence(x)
    # take a bit more than one window so later steps have context, then crop
    x = random_crop(x, rng, min(len(x), audio.WINDOW + audio.SR))
    x = audio.normalize_loudness(x, rng.uniform(-40, -12))     # random input volume
    if rng.random() < p_reverb:
        x = add_reverb(x, rng)
    if rng.random() < p_noise:
        x = add_noise(x, rng)
    if rng.random() < p_clip:
        x = np.clip(x * rng.uniform(2, 6), -1, 1)               # overdriven mic
    if rng.random() < p_phone:
        x = audio.phone_simulate(x, rng=rng, bandpass=rng.random() < 0.6,
                                 codec=rng.choice([None, None, "amr", "gsm"]),
                                 noise_snr_db=rng.uniform(20, 40) if rng.random() < 0.3 else None)
    x = random_crop(x, rng)
    return audio.normalize_loudness(x)


def eval_view(x: np.ndarray, phone: bool, seed: int) -> np.ndarray:
    """Deterministic view for evaluation, matching what the app does."""
    x = audio.trim_silence(x)
    if phone:
        x = audio.phone_simulate(x, rng=np.random.default_rng(seed))
    return audio.normalize_loudness(audio.repeat_pad(x))
