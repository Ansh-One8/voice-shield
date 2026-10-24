"""Voice Shield server.

Run from the voice-shield folder:
    .venv\\Scripts\\python -m uvicorn server.app:app --port 8000

Then open http://localhost:8000 in Chrome or Edge.

Endpoints
    GET  /             laptop test page
    GET  /api/status   is the model loaded?
    POST /api/score    score an uploaded clip (form fields: file, phone=true|false)
    WS   /ws/stream    live audio in Twilio Media Streams format -> verdict messages
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vs import audio                      # noqa: E402
from vs.detector import Detector          # noqa: E402
from vs.risk import build_verdict         # noqa: E402
from vs.stream import VAD, CallSession    # noqa: E402

app = FastAPI(title="Voice Shield")

state = {"detector": None, "vad": None, "error": None, "loading": True, "load_seconds": None}
model_lock = threading.Lock()     # one model call at a time on the CPU
UPLOAD_MAX_WINDOWS = int(os.environ.get("VS_UPLOAD_WINDOWS", "0")) or None   # None = score every window


def _raise_priority():
    """Ask Windows to run this process at high priority and without power throttling.

    Background or app-launched processes can get "efficiency mode", which slows
    the model several times over. Errors are ignored: speed is best effort.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        proc = k32.GetCurrentProcess()
        k32.SetPriorityClass(proc, 0x00000080)               # HIGH_PRIORITY_CLASS

        class PowerThrottling(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]
        state = PowerThrottling(1, 0x1, 0x0)                 # execution speed: never throttle
        k32.SetProcessInformation(proc, 4, ctypes.byref(state), ctypes.sizeof(state))
    except Exception:
        pass


def _load():
    _raise_priority()
    t0 = time.time()
    try:
        threads = int(os.environ.get("VS_THREADS", "0")) or None
        state["vad"] = VAD()
        state["detector"] = Detector(threads=threads)
    except Exception as e:  # shown on the status endpoint and the page
        state["error"] = f"{type(e).__name__}: {e}"
    state["loading"] = False
    state["load_seconds"] = round(time.time() - t0, 1)


@app.on_event("startup")
def startup():
    threading.Thread(target=_load, daemon=True).start()


def _score(x):
    with model_lock:
        det = state["detector"]
        t0 = time.time()
        p = float(det.p_fake_windows([x])[0])
        return p, det.thresholds.label(p), time.time() - t0


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/status")
def status():
    det = state["detector"]
    return {
        "loading": state["loading"],
        "ready": det is not None,
        "error": state["error"],
        "device": det.device if det else None,
        "model": Path(det.checkpoint).name if det else None,
        "load_seconds": state["load_seconds"],
        "ffmpeg_codecs": audio.has_ffmpeg_codecs(),
    }


@app.post("/api/score")
async def score_upload(file: UploadFile = File(...), phone: bool = Form(False)):
    det = state["detector"]
    if det is None:
        return JSONResponse({"error": state["error"] or "Model is still loading."}, status_code=503)
    raw = await file.read()
    try:
        x = audio.to_mono_float(raw, 0)
    except Exception as e:
        return JSONResponse({"error": f"Could not read this audio file ({e}). Try WAV or FLAC."},
                            status_code=400)

    def run():
        with model_lock:
            t0 = time.time()
            res = det.score_clip(x, phone=phone, max_windows=UPLOAD_MAX_WINDOWS)
            return res, time.time() - t0

    res, secs = await asyncio.to_thread(run)
    v = build_verdict(call_id=file.filename or "upload", speech_seconds=res.seconds,
                      call_seconds=len(x) / audio.SR, p_fake=res.p_fake, label=res.label,
                      early=res.seconds < audio.WINDOW / audio.SR,
                      timeline=[{"t": t, "p_fake": round(p, 4)}
                                for t, p in zip(res.window_starts_s, res.window_p_fake)])
    v["model_seconds"] = round(secs, 2)
    v["phone_mode"] = phone
    return v


@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    if state["detector"] is None:
        await ws.send_text(json.dumps({"type": "error",
                                       "message": state["error"] or "Model is still loading."}))
        await ws.close()
        return
    session = CallSession(vad=await asyncio.to_thread(VAD))   # VAD keeps state, so one per call
    scoring: asyncio.Task | None = None

    async def score_and_send(window):
        p, label, secs = await asyncio.to_thread(_score, window)
        v = session.verdict(p, label)
        v["model_seconds"] = round(secs, 2)
        await ws.send_text(json.dumps(v, ensure_ascii=False))

    try:
        while True:
            msg = json.loads(await ws.receive_text())
            event = msg.get("event")
            if event == "start":
                session.handle_start(msg)
            elif event == "media":
                if msg["media"].get("track", "inbound") != "inbound":
                    continue
                x = session.decode(msg["media"]["payload"])
                await asyncio.to_thread(session.add_audio, x)
                # skip a round if the previous score is still running
                if session.ready_to_score() and (scoring is None or scoring.done()):
                    scoring = asyncio.create_task(score_and_send(session.window()))
            elif event == "stop":
                break
    except WebSocketDisconnect:
        pass
    finally:
        if scoring and not scoring.done():
            try:
                await scoring
            except Exception:
                pass          # the page already closed the socket
