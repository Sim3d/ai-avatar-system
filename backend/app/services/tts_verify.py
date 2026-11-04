"""
Catch Chatterbox loops by listening back to its output.

Chatterbox sometimes repeats a word ("today today") or the whole sentence.
A doubled word barely changes the clip length, so a duration check misses
it; a tiny Whisper pass on CPU (~0.4 s for a 2 s clip, no GPU contention
with TTS/MuseTalk) hears it. Only *extra* doubled words count: tiny Whisper
often mishears or drops words, which must not trigger a regeneration.

A clipped repeat of the last word after a pause is dropped by Whisper; it is
found by audio energy after the last recognised word and trimmed in place,
which is cheaper than regenerating.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_model = None
_load_lock = threading.Lock()
_WORD_RE = re.compile(r"[\w']+")
_TAIL_WINDOW_S = 0.05
_TAIL_GAP_S = 0.25
_TAIL_KEEP_S = 0.15
_TAIL_SILENCE_RMS = 0.005
_TAIL_SPEECH_RMS = 0.03


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


def _transcribe(path: str, language: str) -> tuple[str, list[tuple[str, float]]]:
    """Return the transcript and its (word, end time in s) pairs."""
    segments, _ = _get_model().transcribe(
        path,
        language=language,
        beam_size=1,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    segments = list(segments)
    words = [w for s in segments for w in (s.words or [])]
    return " ".join(s.text for s in segments), [(w.word, w.end) for w in words]


def expected_end(expected: str, heard_words: list[tuple[str, float]]) -> Optional[float]:
    """End time of the sentence's last word as heard, or None if not trustworthy.

    Whisper may transcribe a clipped trailing repeat as a junk word
    ("today? Towinged."), so anchor on the last occurrence of the expected
    final word rather than on the last heard word. The anchor must sit near
    the end of the sentence so a misheard earlier occurrence never cuts real
    speech.
    """
    exp = _words(expected)
    if not exp:
        return None
    index, anchor = -1, None
    for word, end in heard_words:
        for token in _words(word):
            index += 1
            if token == exp[-1]:
                anchor = (index, end)
    if anchor is None or anchor[0] + 1 < len(exp) - 1:
        return None
    return anchor[1]


def trailing_speech_start(
    samples: np.ndarray, sr: int, after: float, gap: float = _TAIL_GAP_S
) -> Optional[float]:
    """Start (s) of sound that comes back after a silent gap past `after`.

    Chatterbox sometimes finishes the sentence, pauses, then mumbles a clipped
    copy of the last word. Whisper drops such fragments from the transcript, so
    the only trace is energy after the last recognised word.
    """
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    win = int(sr * _TAIL_WINDOW_S)
    start = int(sr * after)
    silent_run = 0.0
    for i in range(start, len(samples) - win + 1, win):
        rms = float(np.sqrt(np.mean(samples[i : i + win] ** 2)))
        if rms < _TAIL_SILENCE_RMS:
            silent_run += _TAIL_WINDOW_S
        elif silent_run >= gap and rms >= _TAIL_SPEECH_RMS:
            return i / sr
        elif rms >= _TAIL_SPEECH_RMS:
            silent_run = 0.0
    return None


def trim_trailing_babble(path: str, last_word_end: float) -> bool:
    """Cut a clipped repeat after the final word. Returns True when trimmed."""
    import soundfile as sf

    samples, sr = sf.read(path, dtype="float32")
    tail = trailing_speech_start(samples, sr, last_word_end)
    if tail is None:
        return False
    cut = int(sr * min(tail, last_word_end + _TAIL_KEEP_S))
    fade = min(int(sr * 0.03), cut)
    samples = samples[:cut].copy()
    if fade:
        ramp = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        samples[-fade:] *= ramp if samples.ndim == 1 else ramp[:, None]
    sf.write(path, samples, sr)
    return True


def _verify(path: str, text: str, language: str) -> bool:
    heard, heard_words = _transcribe(path, language)
    if has_extra_repetition(text, heard):
        logger.warning(f"TTS repetition detected: expected {text!r}, heard {heard!r}")
        return True
    # Only trust Whisper's timing when it heard the sentence through to its
    # last word; otherwise the "tail" may be real speech it misheard.
    last_end = expected_end(text, heard_words)
    if last_end is not None:
        if trim_trailing_babble(path, last_end):
            logger.warning(f"TTS trailing babble trimmed after {last_end:.2f}s: {text!r}")
    return False


async def sounds_repeated(path: str, text: str, language: str) -> bool:
    """Check a clip for loops; trims a clipped trailing repeat in place.

    Returns True when the clip should be regenerated. Never raises: a verifier
    failure must not cost the user their audio.
    """
    if not settings.TTS_VERIFY_MODEL or not Path(path).exists():
        return False
    try:
        return await asyncio.to_thread(_verify, path, text, language)
    except Exception as e:
        logger.warning(f"TTS verification skipped: {e}")
        return False


async def prewarm() -> None:
    if not settings.TTS_VERIFY_MODEL or not settings.CHATTERBOX_URL:
        return
    try:
        await asyncio.to_thread(_get_model)
    except Exception as e:
        logger.warning(f"TTS verifier prewarm failed: {e}")
