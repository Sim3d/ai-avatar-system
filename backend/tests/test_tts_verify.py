from unittest.mock import AsyncMock

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
