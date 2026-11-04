"""
Text-to-Speech service backed by Chatterbox Multilingual (Resemble AI).

Replaces the deprecated Coqui XTTS v2. Voice profile WAVs in `voice_profiles/`
remain compatible — Chatterbox accepts any WAV reference for zero-shot cloning.

Fallback chain when Chatterbox can't load or fails mid-synthesis:

    chatterbox (cloned voice, GPU) → edge-tts (neural, free, no GPU) → gTTS

Chatterbox runs in-process if importable, or in tts_server.py (its own venv,
because its torch pin conflicts with MuseTalk) when CHATTERBOX_URL is set.

Edge TTS uses Microsoft's neural voices — dramatically better prosody than
gTTS — and needs no API key or GPU, so the degraded experience stays good.
gTTS remains as the last-ditch network fallback.

Synthesis result reports which engine produced the audio so the caller can
notify the user when voice cloning was silently dropped during fallback.
"""

import asyncio
import logging
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from app.config import settings
from app.services import tts_verify

logger = logging.getLogger(__name__)


@dataclass
class SynthResult:
    output_path: str
    engine: str  # "chatterbox" or "gtts"
    fallback: bool  # True if the caller's preferred path was not taken
    voice_cloned: bool  # True if a speaker WAV was actually applied


# Microsoft neural voices for the Edge TTS fallback, one per supported
# language (the same 23-language set the voices API allows). Anything not
# listed falls back to the English voice.
_EDGE_VOICES = {
    "ar": "ar-SA-ZariyahNeural",
    "da": "da-DK-ChristelNeural",
    "de": "de-DE-KatjaNeural",
    "el": "el-GR-AthinaNeural",
    "en": "en-US-AriaNeural",
    "es": "es-ES-ElviraNeural",
    "fi": "fi-FI-NooraNeural",
    "fr": "fr-FR-DeniseNeural",
    "he": "he-IL-HilaNeural",
    "hi": "hi-IN-SwaraNeural",
    "it": "it-IT-ElsaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "ms": "ms-MY-YasminNeural",
    "nl": "nl-NL-ColetteNeural",
    "no": "nb-NO-PernilleNeural",
    "pl": "pl-PL-ZofiaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "sv": "sv-SE-SofieNeural",
    "sw": "sw-KE-ZuriNeural",
    "tr": "tr-TR-EmelNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
}

# Voice Studio preset IDs are persisted on avatars. Keep these mapped to real
# neural voices instead of letting them collapse to the language default.
PRESET_EDGE_VOICES = {
    "default-en": "en-US-GuyNeural",
    "default-warm": "en-US-JennyNeural",
    "default-deep": "en-US-DavisNeural",
    "default-es": "es-ES-ElviraNeural",
}

# Symbols worth keeping: engines read them meaningfully ("25 degrees").
_KEEP_SYMBOLS = set("°©®™")
# Emoji glue: zero-width joiner, variation selectors, keycap combiner.
_EMOJI_JOINERS = {"\u200d", "\ufe0e", "\ufe0f", "\u20e3"}
_WHITESPACE_RUN = re.compile(r"[ \t]{2,}")


def speakable_text(text: str) -> str:
    """Drop emoji and pictographs so TTS engines don't read them by name.

    Edge TTS speaks "😊" as "smiling face with smiling eyes". The chat
    transcript keeps the emoji; only the audio path is cleaned.
    """
    cleaned = "".join(
        ch
        for ch in text
        if ch not in _EMOJI_JOINERS
        and (ch in _KEEP_SYMBOLS or unicodedata.category(ch) not in ("So", "Sk", "Cs"))
    )
    return _WHITESPACE_RUN.sub(" ", cleaned).strip()


# Older configs may still say "coqui"/"xtts" (the engine Chatterbox replaced).
# Treat them as chatterbox instead of breaking every synthesis call.
_LEGACY_PROVIDER_ALIASES = {"coqui": "chatterbox", "xtts": "chatterbox", "xtts_v2": "chatterbox"}


class TTSService:
    """Text-to-Speech service. Lazy-loads the model on first synthesis."""

    def __init__(self):
        provider = settings.TTS_PROVIDER
        if provider in _LEGACY_PROVIDER_ALIASES:
            logger.warning(
                f"TTS_PROVIDER={provider!r} is deprecated (Coqui XTTS was replaced "
                f"by Chatterbox) — using 'chatterbox'. Update your .env."
            )
            provider = _LEGACY_PROVIDER_ALIASES[provider]
        self.provider = provider
        self.model = None
        self._load_error: Optional[Exception] = None
        self._init_lock = asyncio.Lock()
        self._warmed_voices: set[str] = set()

    def is_voice_warm(self, speaker_wav: str) -> bool:
        return not settings.CHATTERBOX_URL or speaker_wav in self._warmed_voices

    def _check_cuda(self) -> bool:
        try:
            return torch.cuda.is_available()
        except Exception:
            return False

    async def initialize(self):
        """Load the Chatterbox model (downloaded from HuggingFace on first run).

        Guarded by a lock — without it, concurrent WebSocket sessions each
        call initialize() before the first one finishes, and they all race
        to download/load the same multi-GB model at once, fighting over the
        same HuggingFace cache lock files (worse on Windows, where the cache
        already runs in degraded no-symlink mode).
        """
        if self.model is not None:
            return
        # A failed load (Chatterbox not installed, OOM, no weights) won't fix
        # itself mid-process; retrying on every sentence only adds latency.
        if self._load_error is not None:
            raise self._load_error

        async with self._init_lock:
            if self.model is not None:  # re-check: another coroutine may have just finished
                return
            if self._load_error is not None:
                raise self._load_error

            if self.provider != "chatterbox":
                raise ValueError(f"Unsupported TTS provider: {self.provider}")

            try:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS

                device = "cuda" if self._check_cuda() else "cpu"
                logger.info(f"Loading Chatterbox multilingual TTS on {device}...")
                self.model = await asyncio.to_thread(
                    ChatterboxMultilingualTTS.from_pretrained, device=device
                )
                logger.info(f"Chatterbox loaded (sr={self.model.sr}, device={device})")

            except Exception as e:
                logger.error(
                    f"Failed to load Chatterbox: {e} — using Edge TTS until the backend restarts"
                )
                self._load_error = e
                raise

    async def synthesize(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        voice_id: Optional[str] = None,
    ) -> SynthResult:
        """
        Synthesize speech.

        Args:
            text: Text to speak.
            output_path: Destination WAV path.
            speaker_wav: Optional reference audio for voice cloning (≥10s recommended).
            language: 2-letter code from Chatterbox's 23-language set.
            voice_id: Optional Voice Studio preset ID. Presets use their named
                Edge neural voice; custom profile IDs are resolved to speaker_wav
                by the WebSocket session before synthesis.

        Returns:
            SynthResult describing the WAV path and which engine was used.
            `fallback=True` indicates the preferred Chatterbox path failed
            and gTTS was used instead — voice cloning is lost in that case.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        text = speakable_text(text)
        if not text:
            raise ValueError("Nothing speakable in text")

        preset_voice = PRESET_EDGE_VOICES.get(voice_id or "")
        if preset_voice and not speaker_wav:
            try:
                await self._edge_fallback(text, output_path, language, voice=preset_voice)
                return SynthResult(
                    output_path=output_path,
                    engine="edge-tts",
                    fallback=False,
                    voice_cloned=False,
                )
            except Exception as edge_err:
                logger.warning(
                    f"Preset voice {voice_id!r} failed ({edge_err}), falling back to gTTS"
                )
                await self._gtts_fallback(text, output_path, language)
                return SynthResult(
                    output_path=output_path,
                    engine="gtts",
                    fallback=True,
                    voice_cloned=False,
                )

        try:
            if speaker_wav and not Path(speaker_wav).exists():
                logger.warning(f"Speaker WAV not found: {speaker_wav!r} — using default voice")
                speaker_wav = None

            if settings.CHATTERBOX_URL:
                await self._remote_chatterbox(text, output_path, speaker_wav, language)
                if await tts_verify.sounds_repeated(output_path, text, language):
                    await self._remote_chatterbox(text, output_path, speaker_wav, language)
                return SynthResult(
                    output_path=output_path,
                    engine="chatterbox",
                    fallback=False,
                    voice_cloned=bool(speaker_wav),
                )

            if self.model is None:
                await self.initialize()

            logger.info(f"Synthesizing (chatterbox, lang={language}): {text[:80]}...")

            kwargs = {"language_id": language}
            if speaker_wav:
                kwargs["audio_prompt_path"] = speaker_wav

            wav = await asyncio.to_thread(self.model.generate, text, **kwargs)
            await asyncio.to_thread(torchaudio.save, output_path, wav, self.model.sr)

            logger.info(
                f"Synthesis complete{' (cloned voice)' if speaker_wav else ''}: {output_path}"
            )
            return SynthResult(
                output_path=output_path,
                engine="chatterbox",
                fallback=False,
                voice_cloned=bool(speaker_wav),
            )

        except Exception as e:
            if speaker_wav:
                logger.warning(
                    f"Chatterbox voice-clone failed — cloned voice NOT applied, "
                    f"falling back to a default neural voice. Error: {e}"
                )
            else:
                logger.warning(f"Chatterbox failed ({e}), trying Edge TTS fallback")

            # Prefer Edge TTS (Microsoft neural voices — much better prosody
            # than gTTS, still free and CPU-only); gTTS is the last resort.
            try:
                await self._edge_fallback(text, output_path, language)
                engine = "edge-tts"
            except Exception as edge_err:
                logger.warning(f"Edge TTS failed ({edge_err}), falling back to gTTS")
                await self._gtts_fallback(text, output_path, language)
                engine = "gtts"

            return SynthResult(
                output_path=output_path,
                engine=engine,
                fallback=True,
                voice_cloned=False,
            )

    async def warm_voice(self, speaker_wav: str) -> None:
        """
        Synthesize a throwaway word with this voice so the Chatterbox sidecar
        caches its speaker conditionals before the first real sentence.
        Never raises.
        """
        if not settings.CHATTERBOX_URL or not Path(speaker_wav).exists():
            return
        if speaker_wav in self._warmed_voices:
            return
        fd, out = tempfile.mkstemp(suffix="_warm.wav")
        os.close(fd)
        try:
            await self._remote_chatterbox("Hi.", out, speaker_wav, "en")
            self._warmed_voices.add(speaker_wav)
        except Exception as e:
            logger.warning(f"Voice warm-up failed (first sentence will be slower): {e}")
        finally:
            Path(out).unlink(missing_ok=True)

    async def _remote_chatterbox(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str],
        language: str,
    ) -> str:
        """Synthesize via tts_server.py, which runs Chatterbox in its own venv."""
        import httpx

        url = settings.CHATTERBOX_URL.rstrip("/") + "/synthesize"
        logger.info(f"Synthesizing (chatterbox remote, lang={language}): {text[:80]}...")
        files = None
        if speaker_wav:
            files = {"speaker_wav": ("speaker.wav", Path(speaker_wav).read_bytes(), "audio/wav")}
        # Short connect timeout: if the service is down, fall back fast.
        timeout = httpx.Timeout(settings.CHATTERBOX_TIMEOUT, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, data={"text": text, "language": language}, files=files)
            resp.raise_for_status()
        Path(output_path).write_bytes(resp.content)
        return output_path

    async def _edge_fallback(
        self,
        text: str,
        output_path: str,
        language: str = "en",
        voice: Optional[str] = None,
    ) -> str:
        """Free neural-voice fallback via Microsoft Edge TTS (no key, no GPU)."""
        import edge_tts
        from pydub import AudioSegment

        voice = voice or _EDGE_VOICES.get(language, _EDGE_VOICES["en"])
        logger.info(f"Synthesizing (edge-tts, {voice}): {text[:80]}...")
        mp3_path = str(Path(output_path).with_suffix(".edge.mp3"))

        try:
            await edge_tts.Communicate(text, voice).save(mp3_path)
            await asyncio.to_thread(
                lambda: AudioSegment.from_mp3(mp3_path).export(output_path, format="wav")
            )
            logger.info(f"Edge TTS synthesis complete: {output_path}")
            return output_path
        finally:
            Path(mp3_path).unlink(missing_ok=True)

    @staticmethod
    def _gtts_language(language: str) -> str:
        """
        Map a session language onto one gTTS actually supports.

        gTTS covers 59 languages but not the same 23 this app offers — Hebrew
        ("he") is accepted everywhere else in the stack and rejected here with
        ValueError. Since gTTS is the last link in the chain, letting that
        raise turned a supported language into silence, which is exactly what
        the fallback chain exists to prevent. Speaking the text in an English
        voice is a poor result; producing nothing is a broken one.
        """
        try:
            from gtts.lang import tts_langs

            if language in tts_langs():
                return language
        except Exception:
            # Language list unavailable (offline, API change) — try as given
            # and let the synth attempt decide.
            return language
        logger.warning(f"gTTS does not support {language!r}; speaking with the English voice")
        return "en"

    async def _gtts_fallback(self, text: str, output_path: str, language: str = "en") -> str:
        """Network-only fallback using Google TTS — no GPU/local model required."""
        # Build the temp name from the path's stem rather than replacing the
        # first ".wav" found anywhere in the string, which would corrupt a path
        # like /tmp/a.wav.d/out.wav.
        mp3_path = str(Path(output_path).with_suffix(".gtts.mp3"))
        try:
            from gtts import gTTS
            from pydub import AudioSegment

            lang = self._gtts_language(language)
            logger.info(f"Synthesizing (gTTS, lang={lang}): {text[:80]}...")

            await asyncio.to_thread(lambda: gTTS(text=text, lang=lang, slow=False).save(mp3_path))
            await asyncio.to_thread(
                lambda: AudioSegment.from_mp3(mp3_path).export(output_path, format="wav")
            )

            logger.info(f"gTTS synthesis complete: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"gTTS also failed: {e}")
            raise
        finally:
            # In a finally so a failed mp3->wav conversion doesn't strand the
            # intermediate file in the session directory.
            Path(mp3_path).unlink(missing_ok=True)

    async def synthesize_bytes(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> bytes:
        """Synthesize and return WAV bytes (used by REST callers)."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            await self.synthesize(text, tmp_path, speaker_wav, language)
            return Path(tmp_path).read_bytes()
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# Suppress unused-name warning — re-exported for type hints elsewhere
__all__ = ["TTSService", "SynthResult", "speakable_text", "tts_service"]


# Global instance
tts_service = TTSService()
