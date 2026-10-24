"""Live call analysis: takes Twilio Media Streams messages, returns verdicts.

The laptop test page sends exactly the same messages Twilio sends, so this
code does not change when we move to real phone calls.

Twilio messages handled: connected, start, media, stop.
Extra for laptop testing: start.mediaFormat.encoding may be "audio/x-l16"
(16-bit little-endian PCM at 16 kHz) to test clean, non-phone audio.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field

import numpy as np
import torch

from . import audio
from .risk import build_verdict

VAD_CHUNK = 512           # Silero VAD frame at 16 kHz (32 ms)
VAD_THRESHOLD = 0.5
HANGOVER_CHUNKS = 6       # keep ~200 ms after speech stops so words aren't clipped
MIN_SPEECH_S = 2.0        # first verdict after this much speech (marked as early)
FULL_SPEECH_S = audio.WINDOW / audio.SR   # ~4.04 s: a full model window
SCORE_EVERY_S = 1.0
WINDOW_FRAMES = audio.WINDOW // VAD_CHUNK + 1   # frames in one model window
MIN_SPEECH_SHARE = 0.4    # skip windows that are mostly silence


class VAD:
    """Silero VAD, one 32 ms frame at a time."""

    def __init__(self):
        from silero_vad import load_silero_vad
        self.model = load_silero_vad()

    def reset(self):
        self.model.reset_states()

    @torch.inference_mode()
    def speech_prob(self, frame16k: np.ndarray) -> float:
        return float(self.model(torch.from_numpy(frame16k), audio.SR).item())


@dataclass
class CallSession:
    vad: VAD
    encoding: str = "audio/x-mulaw"
    sample_rate: int = 8000
    call_id: str = ""
    pending: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    # Recent audio, kept continuous. Cutting pauses out and gluing speech
    # together creates joins that the model reads as synthetic, so we only use
    # VAD to decide when to score, never to edit the audio.
    frames: list[np.ndarray] = field(default_factory=list)
    is_speech: list[bool] = field(default_factory=list)
    speech_samples: int = 0
    total_samples: int = 0
    last_scored_at: int = 0
    hangover: int = 0
    timeline: list[dict] = field(default_factory=list)

    def handle_start(self, msg: dict) -> None:
        start = msg.get("start", {})
        fmt = start.get("mediaFormat", {})
        self.encoding = fmt.get("encoding", "audio/x-mulaw")
        self.sample_rate = int(fmt.get("sampleRate", 8000))
        self.call_id = start.get("callSid") or start.get("streamSid") or "local"
        self.vad.reset()

    def decode(self, payload_b64: str) -> np.ndarray:
        raw = base64.b64decode(payload_b64)
        if self.encoding == "audio/x-l16":
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        else:
            x = audio.g711_decode(raw)
        return audio.resample(x, self.sample_rate, audio.SR)

    def add_audio(self, x16k: np.ndarray) -> None:
        """Store the audio and mark which 32 ms frames contain speech."""
        self.total_samples += len(x16k)
        buf = np.concatenate([self.pending, x16k])
        n = len(buf) // VAD_CHUNK
        for i in range(n):
            frame = buf[i * VAD_CHUNK:(i + 1) * VAD_CHUNK]
            if self.vad.speech_prob(frame) >= VAD_THRESHOLD:
                self.hangover = HANGOVER_CHUNKS
                speech = True
            elif self.hangover > 0:
                self.hangover -= 1
                speech = True
            else:
                speech = False
            self.frames.append(frame)
            self.is_speech.append(speech)
            self.speech_samples += VAD_CHUNK if speech else 0
        self.pending = buf[n * VAD_CHUNK:]
        keep = WINDOW_FRAMES * 2                                   # cap memory
        if len(self.frames) > keep:
            self.frames, self.is_speech = self.frames[-keep:], self.is_speech[-keep:]

    def ready_to_score(self) -> bool:
        recent = self.is_speech[-WINDOW_FRAMES:]
        return (self.speech_samples >= MIN_SPEECH_S * audio.SR and
                self.speech_samples - self.last_scored_at >= SCORE_EVERY_S * audio.SR and
                sum(recent) / max(len(recent), 1) >= MIN_SPEECH_SHARE)

    def window(self) -> np.ndarray:
        """The most recent ~4 s of continuous audio, without leading silence."""
        self.last_scored_at = self.speech_samples
        frames, flags = self.frames[-WINDOW_FRAMES:], self.is_speech[-WINDOW_FRAMES:]
        first = flags.index(True) if True in flags else 0
        return audio.repeat_pad(np.concatenate(frames[first:]))

    def verdict(self, p_fake: float, label: str) -> dict:
        speech_s = self.speech_samples / audio.SR
        self.timeline.append({"t": round(self.total_samples / audio.SR, 2), "p_fake": round(p_fake, 4)})
        return build_verdict(
            call_id=self.call_id,
            speech_seconds=speech_s,
            call_seconds=self.total_samples / audio.SR,
            p_fake=p_fake,
            label=label,
            early=speech_s < FULL_SPEECH_S,
            timeline=self.timeline[-60:],
        )
