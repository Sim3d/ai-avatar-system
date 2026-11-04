"""
Tests for the TTS fallback chain and LLM provider wiring.

The TTS degradation path matters for UX: when Chatterbox can't load (no GPU,
missing model), users should get Microsoft Edge neural voices — not the
robotic gTTS — and the result must be labelled so the UI can warn that voice
cloning was dropped.
"""

from unittest.mock import AsyncMock

import pytest

from app.services.tts import TTSService, speakable_text

pytestmark = pytest.mark.asyncio


def test_legacy_coqui_provider_aliases_to_chatterbox(monkeypatch):
    """Old .env files with TTS_PROVIDER=coqui must keep working."""
    from app.services import tts as tts_module

    monkeypatch.setattr(tts_module.settings, "TTS_PROVIDER", "coqui")
    service = TTSService()
    assert service.provider == "chatterbox"


async def test_tts_falls_back_to_edge_when_chatterbox_unavailable(monkeypatch, tmp_path):
    service = TTSService()
    out = str(tmp_path / "out.wav")

    monkeypatch.setattr(service, "initialize", AsyncMock(side_effect=RuntimeError("no chatterbox")))
    monkeypatch.setattr(service, "_edge_fallback", AsyncMock(return_value=out))
    gtts = AsyncMock()
    monkeypatch.setattr(service, "_gtts_fallback", gtts)

    result = await service.synthesize("Hello world", out, speaker_wav=None, language="en")

    assert result.engine == "edge-tts"
    assert result.fallback is True
    assert result.voice_cloned is False
    gtts.assert_not_awaited()  # gTTS is last resort only


async def test_tts_falls_back_to_gtts_when_edge_also_fails(monkeypatch, tmp_path):
    service = TTSService()
    out = str(tmp_path / "out.wav")

    monkeypatch.setattr(service, "initialize", AsyncMock(side_effect=RuntimeError("no chatterbox")))
    monkeypatch.setattr(service, "_edge_fallback", AsyncMock(side_effect=RuntimeError("edge down")))
    monkeypatch.setattr(service, "_gtts_fallback", AsyncMock(return_value=out))

    result = await service.synthesize("Hello world", out, speaker_wav=None, language="en")

    assert result.engine == "gtts"
    assert result.fallback is True


async def test_tts_uses_selected_preset_voice(monkeypatch, tmp_path):
    service = TTSService()
    out = str(tmp_path / "out.wav")
    edge = AsyncMock(return_value=out)
    initialize = AsyncMock()
    monkeypatch.setattr(service, "_edge_fallback", edge)
    monkeypatch.setattr(service, "initialize", initialize)

    result = await service.synthesize("Hello world", out, voice_id="default-warm")

    assert result.engine == "edge-tts"
    assert result.fallback is False
    assert result.voice_cloned is False
    edge.assert_awaited_once_with("Hello world", out, "en", voice="en-US-JennyNeural")
    initialize.assert_not_awaited()


async def test_preset_voice_falls_back_when_edge_fails(monkeypatch, tmp_path):
    service = TTSService()
    out = str(tmp_path / "out.wav")
    monkeypatch.setattr(service, "_edge_fallback", AsyncMock(side_effect=RuntimeError("offline")))
    gtts = AsyncMock(return_value=out)
    monkeypatch.setattr(service, "_gtts_fallback", gtts)

    result = await service.synthesize("Hola", out, language="es", voice_id="default-es")

    assert result.engine == "gtts"
    assert result.fallback is True
    gtts.assert_awaited_once_with("Hola", out, "es")


def test_speakable_text_drops_emoji_but_keeps_words():
    assert speakable_text("How are you doing today? 😊") == "How are you doing today?"
    assert speakable_text("Great 👍🏽 job ❤️ team 👨‍👩‍👧!") == "Great job team !"
    assert speakable_text("It is 25°C") == "It is 25°C"
    assert speakable_text("🎉🎉") == ""


def test_speakable_text_drops_markdown_marks():
    assert speakable_text("Open **PIM** in *BIMTrack*.") == "Open PIM in BIMTrack."
    assert speakable_text("## Steps\n- Click `Save`\n> Note") == "Steps\nClick Save\nNote"
    assert speakable_text("See [the docs](https://x.io) and __this__.") == "See the docs and this."
    assert speakable_text("Use snake_case names.") == "Use snake_case names."
    assert speakable_text("Total is 5 - 3.") == "Total is 5 - 3."


async def test_tts_never_sends_emoji_to_engine(monkeypatch, tmp_path):
    service = TTSService()
    out = str(tmp_path / "out.wav")
    edge = AsyncMock(return_value=out)
    monkeypatch.setattr(service, "_edge_fallback", edge)

    await service.synthesize("Hi there! 😊", out, voice_id="default-en")

    edge.assert_awaited_once_with("Hi there!", out, "en", voice="en-US-GuyNeural")


async def test_failed_chatterbox_load_is_not_retried_every_sentence(monkeypatch):
    service = TTSService()
    service._load_error = ImportError("No module named 'chatterbox'")

    with pytest.raises(ImportError):
        await service.initialize()


async def test_remote_chatterbox_used_when_url_configured(monkeypatch, tmp_path):
    from app.services import tts as tts_module

    service = TTSService()
    out = str(tmp_path / "out.wav")
    speaker = tmp_path / "me.wav"
    speaker.write_bytes(b"RIFF")
    monkeypatch.setattr(tts_module.settings, "CHATTERBOX_URL", "http://tts:8001")
    remote = AsyncMock(return_value=out)
    initialize = AsyncMock()
    monkeypatch.setattr(service, "_remote_chatterbox", remote)
    monkeypatch.setattr(service, "initialize", initialize)

    result = await service.synthesize("Hello", out, speaker_wav=str(speaker), language="fr")

    assert (result.engine, result.fallback, result.voice_cloned) == ("chatterbox", False, True)
    remote.assert_awaited_once_with("Hello", out, str(speaker), "fr")
    initialize.assert_not_awaited()


async def test_remote_chatterbox_down_falls_back_to_edge(monkeypatch, tmp_path):
    from app.services import tts as tts_module

    service = TTSService()
    out = str(tmp_path / "out.wav")
    monkeypatch.setattr(tts_module.settings, "CHATTERBOX_URL", "http://tts:8001")
    monkeypatch.setattr(
        service, "_remote_chatterbox", AsyncMock(side_effect=ConnectionError("refused"))
    )
    monkeypatch.setattr(service, "_edge_fallback", AsyncMock(return_value=out))

    result = await service.synthesize("Hello", out)

    assert result.engine == "edge-tts"
    assert result.fallback is True


def test_llm_ollama_provider_uses_openai_compatible_client(monkeypatch):
    """LLM_PROVIDER=ollama wires an OpenAI client at the local base URL."""
    from app.services import llm as llm_module

    monkeypatch.setattr(llm_module.settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_module.settings, "OPENAI_BASE_URL", None)
    monkeypatch.setattr(llm_module.settings, "OPENAI_API_KEY", "")

    service = llm_module.LLMService()
    assert service.provider == "openai"  # downstream paths are the OpenAI ones
    assert "localhost:11434" in str(service.client.base_url)


def test_llm_openai_provider_respects_custom_base_url(monkeypatch):
    from app.services import llm as llm_module

    monkeypatch.setattr(llm_module.settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm_module.settings, "OPENAI_BASE_URL", "http://vllm:8001/v1")
    monkeypatch.setattr(llm_module.settings, "OPENAI_API_KEY", "k")

    service = llm_module.LLMService()
    assert "vllm:8001" in str(service.client.base_url)
