"""
Spoken avatar introductions.

The intro is the avatar's `description` metadata read in its own (cloned)
voice. It is synthesized once and cached in storage under a key derived from
the text and voice, so editing either one naturally invalidates the cache.
"""

import asyncio
import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

from app.models import Avatar
from app.services.storage import storage_service
from app.services.tts import tts_service

logger = logging.getLogger(__name__)

_VOICE_INDEX = Path("voice_profiles") / "index.json"
_locks: dict[str, asyncio.Lock] = {}


def read_metadata(avatar: Avatar) -> dict:
    metadata = avatar.avatar_metadata or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            return {}
    return metadata if isinstance(metadata, dict) else {}


def intro_description(avatar: Avatar) -> Optional[str]:
    description = read_metadata(avatar).get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    return None


def intro_key(avatar_id: str, description: str, voice_id: Optional[str]) -> str:
    digest = hashlib.sha256(f"{voice_id or ''}\n{description}".encode("utf-8")).hexdigest()[:16]
    return f"avatars/{avatar_id}/intro-{digest}.wav"


async def _voice_wav_path(voice_id: Optional[str]) -> Optional[str]:
    if not voice_id or not _VOICE_INDEX.exists():
        return None
    try:
        entries = json.loads(await asyncio.to_thread(_VOICE_INDEX.read_text))
    except Exception as e:
        logger.warning(f"Could not read voice index: {e}")
        return None
    entry = next((e for e in entries if e.get("id") == voice_id), None)
    return entry.get("wav_path") if entry else None


async def get_intro_audio(avatar_id: str, description: str, voice_id: Optional[str]) -> bytes:
    """Return the cached intro WAV, synthesizing it on first request."""
    key = intro_key(avatar_id, description, voice_id)
    # One synthesis per avatar at a time: a double click must not run TTS twice.
    async with _locks.setdefault(avatar_id, asyncio.Lock()):
        try:
            return await storage_service.download_file(key)
        except Exception:
            pass

        speaker_wav = await _voice_wav_path(voice_id)
        result_path = Path(tempfile.gettempdir()) / f"{avatar_id}_intro.wav"
        try:
            result = await tts_service.synthesize(
                description,
                str(result_path),
                speaker_wav=speaker_wav,
                voice_id=voice_id,
            )
            audio = Path(result.output_path).read_bytes()
        finally:
            result_path.unlink(missing_ok=True)

        # Never cache a degraded voice: the next click retries the cloned one.
        if speaker_wav and not result.voice_cloned:
            logger.warning(f"Intro for avatar {avatar_id} synthesized without its cloned voice")
            return audio

        await storage_service.upload_file(audio, key, content_type="audio/wav")
        logger.info(f"Intro cached for avatar {avatar_id} ({result.engine}): {key}")
        return audio


async def prewarm_intro(avatar_id: str, description: str, voice_id: Optional[str]) -> None:
    """Background warm-up so the first click plays instantly."""
    try:
        await get_intro_audio(avatar_id, description, voice_id)
    except Exception as e:
        logger.warning(f"Intro prewarm failed for avatar {avatar_id}: {e}")
