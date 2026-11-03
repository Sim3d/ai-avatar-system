"""
Standalone Chatterbox TTS service.

Chatterbox needs torch 2.6 / transformers 5.2, MuseTalk needs torch 2.2 /
transformers 4.37, so they cannot share one virtualenv. This process runs in
its own venv (requirements-tts.txt) and the backend reaches it over HTTP when
CHATTERBOX_URL is set. It deliberately imports nothing from `app`.

    python -m venv venv-tts
    venv-tts/bin/pip install -r requirements-tts.txt
    venv-tts/bin/python tts_server.py            # http://127.0.0.1:8001

Env: TTS_HOST (default 127.0.0.1), TTS_PORT (8001), TTS_DEVICE (auto).
Keep it bound to localhost/private network: it has no authentication.
"""

import asyncio
import hashlib
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tts_server")

MAX_TEXT_CHARS = 1000
MAX_SPEAKER_BYTES = 20 * 1024 * 1024
# Reference WAVs are cached by content hash so a profile is written once,
# not once per sentence.
_SPEAKER_CACHE = Path(tempfile.gettempdir()) / "chatterbox-speakers"

app = FastAPI(title="Chatterbox TTS")
_model = None
# One GPU, one model: serialize generation instead of interleaving it.
_generate_lock = asyncio.Lock()


def _device() -> str:
    forced = os.getenv("TTS_DEVICE")
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
async def _load_model() -> None:
    global _model
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = _device()
    logger.info(f"Loading Chatterbox multilingual on {device}...")
    _model = await asyncio.to_thread(ChatterboxMultilingualTTS.from_pretrained, device=device)
    _SPEAKER_CACHE.mkdir(parents=True, exist_ok=True)
    logger.info(f"Chatterbox ready (sr={_model.sr})")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok" if _model is not None else "loading"}


def _cached_speaker(data: bytes) -> str:
    path = _SPEAKER_CACHE / f"{hashlib.sha256(data).hexdigest()}.wav"
    if not path.exists():
        path.write_bytes(data)
    return str(path)


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    language: str = Form("en"),
    speaker_wav: Optional[UploadFile] = File(None),
) -> Response:
    if _model is None:
        raise HTTPException(503, "Model still loading")
    text = text.strip()
    if not text or len(text) > MAX_TEXT_CHARS:
        raise HTTPException(400, f"text must be 1-{MAX_TEXT_CHARS} chars")

    kwargs = {"language_id": language}
    if speaker_wav is not None:
        data = await speaker_wav.read(MAX_SPEAKER_BYTES + 1)
        if len(data) > MAX_SPEAKER_BYTES:
            raise HTTPException(413, "speaker_wav too large")
        if data:
            kwargs["audio_prompt_path"] = _cached_speaker(data)

    async with _generate_lock:
        try:
            wav = await asyncio.to_thread(_model.generate, text, **kwargs)
        except Exception as e:
            logger.exception("Synthesis failed")
            raise HTTPException(500, f"Synthesis failed: {e}") from e

    buf = io.BytesIO()
    torchaudio.save(buf, wav, _model.sr, format="wav")
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("TTS_HOST", "127.0.0.1"),
        port=int(os.getenv("TTS_PORT", "8001")),
    )
