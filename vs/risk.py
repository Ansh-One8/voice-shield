"""Turns model scores into the verdict message the app shows.

The scam-intent branch (speech-to-text + LLM) is not built yet, so `intent`
is always None for now. A real human voice never produces a "safe" message:
human scammers exist, so the low-risk text still warns about OTPs.
"""

from __future__ import annotations

MESSAGES = {
    "fake": ("This voice is likely AI-generated ({pct}%). Do not share OTPs, PINs or send money.",
             "यह आवाज़ AI से बनी लग रही है ({pct}%)। OTP, PIN शेयर न करें और पैसे न भेजें।"),
    "unsure": ("Not sure yet ({pct}% AI). Keep listening and be careful.",
               "अभी पक्का नहीं कह सकते ({pct}% AI)। सावधान रहें।"),
    "real": ("This sounds like a real human voice. Real people can scam too, so never share OTPs.",
             "यह असली इंसान की आवाज़ लगती है। फिर भी OTP कभी शेयर न करें।"),
}
RISK = {"fake": "high", "unsure": "medium", "real": "low"}


def build_verdict(*, call_id: str, speech_seconds: float, call_seconds: float,
                  p_fake: float, label: str, early: bool, timeline: list[dict],
                  intent: dict | None = None) -> dict:
    pct = round(p_fake * 100)
    en, hi = MESSAGES[label]
    risk = RISK[label]
    if intent and intent.get("flag"):
        risk = "high"
    return {
        "type": "verdict",
        "call_id": call_id,
        "speech_seconds": round(speech_seconds, 2),
        "call_seconds": round(call_seconds, 2),
        "early": early,                    # true while we have less than ~4 s of speech
        "deepfake": {"label": label, "score": round(p_fake, 4)},
        "intent": intent,
        "risk": risk,
        "message_en": en.format(pct=pct),
        "message_hi": hi.format(pct=pct),
        "timeline": timeline,
    }
