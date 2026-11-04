"""
Catch Chatterbox loops by listening back to its output.

Chatterbox sometimes repeats a word ("today today") or the whole sentence.
A doubled word barely changes the clip length, so a duration check misses
it; a tiny Whisper pass on CPU (~0.4 s for a 2 s clip, no GPU contention
with TTS/MuseTalk) hears it. Only *extra* doubled words count: tiny Whisper
often mishears or drops words, which must not trigger a regeneration.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

_model = None
_load_lock = threading.Lock()
_WORD_RE = re.compile(r"[\w']+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _doubled_ngrams(words: list[str]) -> set[tuple[str, ...]]:
    found = set()
    for n in range(1, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i : i + n] == words[i + n : i + 2 * n]:
                found.add(tuple(words[i : i + n]))
    return found


def has_extra_repetition(expected: str, heard: str) -> bool:
    """True when `heard` repeats words back-to-back that `expected` does not."""
    exp, got = _words(expected), _words(heard)
    if len(got) <= len(exp):
        return False
    return bool(_doubled_ngrams(got) - _doubled_ngrams(exp))


def _get_model():
    global _model
    with _load_lock:
        if _model is None:
            from faster_whisper import WhisperModel

            _model = WhisperModel(
                settings.TTS_VERIFY_MODEL, device="cpu", compute_type="int8", cpu_threads=2
            )
            logger.info(f"TTS verifier loaded (whisper {settings.TTS_VERIFY_MODEL}, cpu)")
    return _model


def _transcribe(path: str, language: str) -> str:
    segments, _ = _get_model().transcribe(
        path,
        language=language,
        beam_size=1,
        without_timestamps=True,
        condition_on_previous_text=False,
    )
    return " ".join(s.text for s in segments)


async def sounds_repeated(path: str, text: str, language: str) -> bool:
    """Never raises: a verifier failure must not cost the user their audio."""
    if not settings.TTS_VERIFY_MODEL or not Path(path).exists():
        return False
    try:
        heard = await asyncio.to_thread(_transcribe, path, language)
    except Exception as e:
        logger.warning(f"TTS verification skipped: {e}")
        return False
    if has_extra_repetition(text, heard):
        logger.warning(f"TTS repetition detected: expected {text!r}, heard {heard!r}")
        return True
    return False


async def prewarm() -> None:
    if not settings.TTS_VERIFY_MODEL or not settings.CHATTERBOX_URL:
        return
    try:
        await asyncio.to_thread(_get_model)
    except Exception as e:
        logger.warning(f"TTS verifier prewarm failed: {e}")
