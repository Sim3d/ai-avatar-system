from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.services import tts_verify
from app.services.tts import TTSService
from app.services.tts_verify import has_extra_repetition


@pytest.mark.parametrize(
    "expected, heard",
    [
        ("How are you doing today?", "How are you doing today? Today."),
        ("How can I help you today?", "How can I help you today? How can I help you today?"),
        ("Great to meet you!", "Great great to meet you."),
    ],
)
def test_doubled_words_are_flagged(expected, heard):
    assert has_extra_repetition(expected, heard)


@pytest.mark.parametrize(
    "expected, heard",
    [
        ("Hi there! Great to meet you!", "either great to meet you"),
        ("How are you doing today?", "How are you doing today?"),
        ("It was very, very good.", "It was very very good."),
        ("Bye bye now.", "Bye bye now."),
        ("I am here.", "I am here, uh, okay."),
    ],
)
def test_mishearing_or_intended_repeats_are_not_flagged(expected, heard):
    assert not has_extra_repetition(expected, heard)


@pytest.mark.asyncio
async def test_repeated_chatterbox_clip_is_regenerated_once(monkeypatch, tmp_path):
    from app.services import tts as tts_module

    service = TTSService()
    out = str(tmp_path / "out.wav")
    monkeypatch.setattr(tts_module.settings, "CHATTERBOX_URL", "http://tts:8001")
    remote = AsyncMock(return_value=out)
    monkeypatch.setattr(service, "_remote_chatterbox", remote)
    monkeypatch.setattr(tts_verify, "sounds_repeated", AsyncMock(side_effect=[True]))

    result = await service.synthesize("How are you doing today?", out, language="en")

    assert result.engine == "chatterbox"
    assert remote.await_count == 2


@pytest.mark.asyncio
async def test_verifier_failure_keeps_the_audio(monkeypatch, tmp_path):
    out = tmp_path / "out.wav"
    out.write_bytes(b"RIFF")
    monkeypatch.setattr(tts_verify.settings, "TTS_VERIFY_MODEL", "tiny")
    monkeypatch.setattr(tts_verify, "_transcribe", lambda *_: (_ for _ in ()).throw(OSError("x")))

    assert await tts_verify.sounds_repeated(str(out), "Hello", "en") is False


SR = 16000


def _clip(*parts):
    """Build audio from (seconds, amplitude) runs of a 200 Hz tone."""
    chunks = []
    for seconds, amp in parts:
        t = np.arange(int(seconds * SR)) / SR
        chunks.append((amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32))
    return np.concatenate(chunks)


def _write(path, samples):
    import soundfile as sf

    sf.write(str(path), samples, SR)


def test_clipped_repeat_after_last_word_is_trimmed(monkeypatch, tmp_path):
    sf = pytest.importorskip("soundfile")
    # Pattern of a real clip: "...today?" ends at 1.3s, silence, then a clipped "today".
    out = tmp_path / "out.wav"
    _write(out, _clip((1.3, 0.2), (0.6, 0.0), (0.3, 0.2)))
    monkeypatch.setattr(
        tts_verify, "_transcribe", lambda *_: (" How are you doing today?", 1.3)
    )

    assert tts_verify._verify(str(out), "How are you doing today?", "en") is False

    trimmed, _ = sf.read(str(out))
    assert len(trimmed) / SR == pytest.approx(1.45, abs=0.01)


def test_clean_clip_is_not_trimmed(monkeypatch, tmp_path):
    sf = pytest.importorskip("soundfile")
    out = tmp_path / "out.wav"
    _write(out, _clip((1.3, 0.2), (0.8, 0.0)))
    monkeypatch.setattr(tts_verify, "_transcribe", lambda *_: (" Hi there.", 1.3))

    tts_verify._verify(str(out), "Hi there.", "en")

    assert len(sf.read(str(out))[0]) == int(2.1 * SR)


def test_tail_is_kept_when_whisper_missed_the_last_word(monkeypatch, tmp_path):
    sf = pytest.importorskip("soundfile")
    out = tmp_path / "out.wav"
    _write(out, _clip((1.0, 0.2), (0.4, 0.0), (0.5, 0.2)))
    monkeypatch.setattr(tts_verify, "_transcribe", lambda *_: (" Hi there, great", 1.0))

    tts_verify._verify(str(out), "Hi there, great to meet you.", "en")

    assert len(sf.read(str(out))[0]) == int(1.9 * SR)
