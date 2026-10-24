"""Pretend to be Twilio: stream an audio file to the server and print verdicts.

Usage:
    python tools/stream_file.py test_set/sanity/fake_windows_tts.wav
    python tools/stream_file.py clip.wav --realtime      # 20 ms per message, like a real call
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vs import audio  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--url", default="ws://localhost:8000/ws/stream")
    ap.add_argument("--realtime", action="store_true")
    args = ap.parse_args()

    x8k = audio.load_audio(args.file, audio.PHONE_SR)
    payload = audio.g711_encode(x8k)
    frame = 160                                   # 20 ms at 8 kHz
    async with websockets.connect(args.url) as ws:
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({"event": "start", "start": {
            "streamSid": "MZtest", "callSid": "CAtest", "tracks": ["inbound"],
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}}))

        async def reader():
            async for m in ws:
                v = json.loads(m)
                if v.get("type") == "verdict":
                    print(f"speech {v['speech_seconds']:5.1f}s  AI {v['deepfake']['score'] * 100:5.1f}%  "
                          f"{v['deepfake']['label']:6s} {'(early)' if v['early'] else '       '} "
                          f"model {v['model_seconds']}s  | {v['message_en']}")
                else:
                    print(v)

        task = asyncio.create_task(reader())
        t0 = time.time()
        for i in range(0, len(payload), frame):
            await ws.send(json.dumps({"event": "media", "media": {
                "track": "inbound", "chunk": str(i // frame + 1),
                "payload": base64.b64encode(payload[i:i + frame]).decode()}}))
            if args.realtime:
                await asyncio.sleep(max(0, (i // frame + 1) * 0.02 - (time.time() - t0)))
        await asyncio.sleep(3)                    # let the last verdict arrive
        await ws.send(json.dumps({"event": "stop"}))
        task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
