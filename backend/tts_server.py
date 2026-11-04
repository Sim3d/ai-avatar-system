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
import time
from collections import OrderedDict
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
# Speaker conditionals (voice embedding + prompt tokens) per reference WAV.
# Chatterbox recomputes them from the file on every generate() given a path,
# which costs a few hundred ms per sentence; avatars reuse a handful of voices.
_MAX_CACHED_SPEAKERS = 32
_speaker_conds: "OrderedDict[str, object]" = OrderedDict()
_default_conds = None
# Chatterbox's default 0.8 sampling makes cloned voices stutter/repeat words;
# a lower temperature keeps the voice but stabilises the token stream.
_TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.6"))
# Cloned voices speak at up to ~0.09 s/char, so a tighter limit regenerated
# clean sentences and doubled the gap between clips. This only has to catch
# whole-sentence loops (~2x); doubled words are the ASR verifier's job.
_MAX_SEC_PER_CHAR = float(os.getenv("TTS_MAX_SEC_PER_CHAR", "0.13"))
_MIN_CHECK_SECONDS = 1.5


def _device() -> str:
    forced = os.getenv("TTS_DEVICE")
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
async def _load_model() -> None:
    global _model, _default_conds
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = _device()
    logger.info(f"Loading Chatterbox multilingual on {device}...")
    model = await asyncio.to_thread(ChatterboxMultilingualTTS.from_pretrained, device=device)
    _default_conds = model.conds
    _SPEAKER_CACHE.mkdir(parents=True, exist_ok=True)

    # The first generate() pays CUDA kernel/graph warm-up (~15s measured);
    # pay it here, before /health reports ready, not on a user's first turn.
    t0 = time.perf_counter()
    try:
        await asyncio.to_thread(model.generate, "Hello there.", language_id="en")
        logger.info(f"Chatterbox warm-up done in {time.perf_counter() - t0:.1f}s")
    except Exception:
        logger.exception("Chatterbox warm-up failed (continuing)")
    model.conds = _default_conds
    _model = model
    logger.info(f"Chatterbox ready (sr={_model.sr})")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok" if _model is not None else "loading"}


def _speaker_key(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cached_speaker(key: str, data: bytes) -> str:
    path = _SPEAKER_CACHE / f"{key}.wav"
    if not path.exists():
        path.write_bytes(data)
    return str(path)


def _use_speaker(key: Optional[str], data: Optional[bytes]) -> None:
    """Point the model at a speaker's conditionals (default voice when key is None).

    Must run under _generate_lock: model.conds is shared state, and leaving a
    cloned voice set would make the next default-voice request speak with it.
    """
    if key is None:
        _model.conds = _default_conds
        return
    conds = _speaker_conds.get(key)
    if conds is None:
        _model.prepare_conditionals(_cached_speaker(key, data))
        conds = _model.conds
        _speaker_conds[key] = conds
        if len(_speaker_conds) > _MAX_CACHED_SPEAKERS:
            _speaker_conds.popitem(last=False)
    else:
        _speaker_conds.move_to_end(key)
    _model.conds = conds


def _max_plausible_seconds(text: str) -> float:
    """Longest believable duration for `text`; beyond it the model looped.

    Chatterbox sometimes finishes the sentence and starts it over in the same
    clip ("How can I help you today? How can I help you today?"). Measured
    clean speech is ~0.05-0.09 s/char; a doubled sentence is roughly twice that.
    """
    return max(_MIN_CHECK_SECONDS, 0.3 + _MAX_SEC_PER_CHAR * len(text))


def _generate(text: str, language: str, key: Optional[str], data: Optional[bytes]):
    _use_speaker(key, data)
    wav = _model.generate(text, language_id=language, temperature=_TEMPERATURE)
    limit = _max_plausible_seconds(text)
    seconds = wav.shape[-1] / _model.sr
    if seconds <= limit:
        return wav
    logger.warning(
        f"Audio {seconds:.2f}s exceeds {limit:.2f}s for {len(text)} chars "
        "(likely repetition); regenerating"
    )
    retry = _model.generate(text, language_id=language, temperature=_TEMPERATURE)
    return retry if retry.shape[-1] < wav.shape[-1] else wav


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

    key: Optional[str] = None
    data: Optional[bytes] = None
    if speaker_wav is not None:
        data = await speaker_wav.read(MAX_SPEAKER_BYTES + 1)
        if len(data) > MAX_SPEAKER_BYTES:
            raise HTTPException(413, "speaker_wav too large")
        if data:
            key = _speaker_key(data)

    async with _generate_lock:
        t0 = time.perf_counter()
        try:
            wav = await asyncio.to_thread(_generate, text, language, key, data)
        except Exception as e:
            logger.exception("Synthesis failed")
            raise HTTPException(500, f"Synthesis failed: {e}") from e
        logger.info(
            f"Synthesized {len(text)} chars in {time.perf_counter() - t0:.2f}s "
            f"(speaker={'clone' if key else 'default'})"
        )

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
