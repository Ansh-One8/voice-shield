"""Audio helpers: loading, clean-up, and phone-call simulation.

Everything works on mono float32 numpy arrays in the range [-1, 1].
The detector expects 16 kHz audio; phone audio (Twilio) is 8 kHz mu-law.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

SR = 16000            # detector sample rate
PHONE_SR = 8000       # telephone sample rate
WINDOW = 64600        # samples the SLS model was trained on (~4.04 s at 16 kHz)


# ---------------------------------------------------------------- loading

def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def _load_with_av(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode formats libsndfile can't (m4a, aac, webm, opus in mp4) with PyAV."""
    import av
    with av.open(str(path)) as c:
        stream = c.streams.audio[0]
        rate = stream.codec_context.sample_rate
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        parts = []
        for frame in c.decode(stream):
            for f in resampler.resample(frame):
                parts.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):
            parts.append(f.to_ndarray().reshape(-1))
    return np.concatenate(parts).astype(np.float32), rate


def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """Load an audio file as mono at `sr` (wav, flac, ogg, mp3, and m4a/aac via PyAV)."""
    try:
        x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:
        x, file_sr = _load_with_av(path)
    return resample(x, file_sr, sr)


def to_mono_float(raw: bytes, sr_in: int, sr: int = SR) -> np.ndarray:
    """Decode an uploaded file's bytes."""
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        return load_audio(tmp, sr)
    finally:
        Path(tmp).unlink(missing_ok=True)


# ---------------------------------------------------------------- clean-up

def normalize_loudness(x: np.ndarray, target_dbfs: float = -23.0) -> np.ndarray:
    """Scale to a fixed RMS level so loudness can't be used as a shortcut."""
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    y = x * (10 ** (target_dbfs / 20) / rms)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def trim_silence(x: np.ndarray, sr: int = SR, top_db: float = 35.0,
                 frame_ms: float = 25.0) -> np.ndarray:
    """Remove silence at the start and end (energy based).

    Silence length alone predicts real vs fake in ASVspoof 2019, so we cut it.
    """
    frame = int(sr * frame_ms / 1000)
    if len(x) < frame * 2:
        return x
    n = len(x) // frame
    energy = np.array([np.mean(x[i * frame:(i + 1) * frame] ** 2) for i in range(n)])
    db = 10 * np.log10(energy + 1e-12)
    keep = np.where(db > db.max() - top_db)[0]
    if len(keep) == 0:
        return x
    return x[keep[0] * frame:(keep[-1] + 1) * frame]


def repeat_pad(x: np.ndarray, length: int = WINDOW) -> np.ndarray:
    """Cut or repeat-pad to `length` samples (same as the SLS training code)."""
    if len(x) >= length:
        return x[:length]
    reps = length // max(len(x), 1) + 1
    return np.tile(x, reps)[:length]


def window_starts(n: int, length: int = WINDOW, hop: int = SR, max_windows: int | None = None) -> list[int]:
    """Start samples of model-sized windows; with max_windows, spread evenly over the clip."""
    if n <= length:
        return [0]
    if max_windows:
        k = min(max_windows, (n - length) // hop + 2)
        return [int(round(v)) for v in np.linspace(0, n - length, k)]
    starts = list(range(0, n - length + 1, hop))
    if starts[-1] + length < n:
        starts.append(n - length)
    return starts


def windows(x: np.ndarray, length: int = WINDOW, hop: int = SR,
            max_windows: int | None = None) -> list[np.ndarray]:
    """Split a long clip into model-sized windows (all, or at most max_windows spread out)."""
    if len(x) <= length:
        return [repeat_pad(x, length)]
    return [x[s:s + length] for s in window_starts(len(x), length, hop, max_windows)]


# ---------------------------------------------------------------- G.711 mu-law
# Twilio Media Streams always send 8 kHz G.711 mu-law, mono.

def _build_g711_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.float32)
    for i in range(256):
        u = ~i & 0xFF
        sign = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        sample = ((mantissa << 3) + 0x84) << exponent
        sample -= 0x84
        table[i] = (-sample if sign else sample) / 32768.0
    return table


_G711 = _build_g711_table()


def g711_decode(b: bytes) -> np.ndarray:
    return _G711[np.frombuffer(b, dtype=np.uint8)]


def g711_encode(x: np.ndarray) -> bytes:
    pcm = np.clip(x * 32768.0, -32635, 32635).astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0)
    mag = np.abs(pcm) + 0x84
    exponent = np.clip(np.floor(np.log2(mag)).astype(np.int32) - 7, 0, 7)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()


# ---------------------------------------------------------------- phone simulation

_BANDPASS = butter(4, [300, 3400], btype="bandpass", fs=PHONE_SR, output="sos")


def _ffmpeg_codec(x8k: np.ndarray, codec: str) -> np.ndarray | None:
    """Round-trip 8 kHz audio through a real phone codec using ffmpeg, if present."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        return None
    fmt = {"amr": ("libopencore_amrnb", "amr", ["-b:a", "12.2k"]),
           "gsm": ("libgsm", "gsm", [])}[codec]
    with tempfile.TemporaryDirectory() as d:
        src, enc, out = Path(d, "in.wav"), Path(d, f"enc.{fmt[1]}"), Path(d, "out.wav")
        sf.write(src, x8k, PHONE_SR)
        ok = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src), "-ar", "8000",
                             "-ac", "1", "-c:a", fmt[0], *fmt[2], str(enc)],
                            capture_output=True).returncode == 0
        if not ok:
            return None
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(enc), "-ar", "8000",
                        str(out)], capture_output=True)
        y, _ = sf.read(out, dtype="float32")
        return y[:len(x8k)]


def phone_simulate(x: np.ndarray, sr: int = SR, codec: str | None = None,
                   noise_snr_db: float | None = None,
                   rng: np.random.Generator | None = None,
                   bandpass: bool = True) -> np.ndarray:
    """Make clean audio sound like a phone call, then return it at `sr`.

    Steps: downsample to 8 kHz, telephone band (300-3400 Hz), optional real codec
    (AMR-NB or GSM via ffmpeg), optional background noise, G.711 mu-law (what
    Twilio delivers), back up to `sr`.
    """
    rng = rng or np.random.default_rng()
    y = resample(x, sr, PHONE_SR)
    if bandpass:
        y = sosfilt(_BANDPASS, y).astype(np.float32)
    if codec:
        coded = _ffmpeg_codec(y, codec)
        if coded is not None:
            y = coded
    if noise_snr_db is not None:
        p = np.mean(y ** 2) + 1e-12
        y = y + rng.normal(0, np.sqrt(p / 10 ** (noise_snr_db / 10)), len(y)).astype(np.float32)
    y = g711_decode(g711_encode(np.clip(y, -1, 1)))
    return resample(y, PHONE_SR, sr)


def has_ffmpeg_codecs() -> bool:
    return shutil.which("ffmpeg") is not None
