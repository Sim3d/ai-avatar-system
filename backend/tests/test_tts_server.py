"""Repetition guard in the Chatterbox sidecar (tts_server._generate)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

import tts_server  # noqa: E402

SR = 24000


class FakeModel:
    sr = SR

    def __init__(self, durations):
        self.durations = list(durations)
        self.calls = 0
        self.conds = None

    def generate(self, text, language_id, temperature):
        self.calls += 1
        return torch.zeros(1, int(self.durations.pop(0) * SR))


@pytest.fixture
def fake(monkeypatch):
    def install(*durations):
        model = FakeModel(durations)
        monkeypatch.setattr(tts_server, "_model", model)
        return model

    return install


TEXT = "How can I help you today?"  # 25 chars -> limit ~3.55s


def test_normal_audio_is_returned_without_retry(fake):
    model = fake(1.3)
    wav, _ = tts_server._generate(TEXT, "en", None, None)
    assert model.calls == 1
    assert wav.shape[-1] == int(1.3 * SR)


def test_looping_audio_is_regenerated_and_shorter_kept(fake):
    model = fake(4.2, 1.25)
    wav, _ = tts_server._generate(TEXT, "en", None, None)
    assert model.calls == 2
    assert wav.shape[-1] == int(1.25 * SR)


def test_retry_keeps_first_when_retry_is_longer(fake):
    fake(4.2, 5.0)
    wav, _ = tts_server._generate(TEXT, "en", None, None)
    assert wav.shape[-1] == int(4.2 * SR)


def test_slow_cloned_voice_is_not_regenerated(fake):
    model = fake(2.6)
    tts_server._generate(TEXT, "en", None, None)
    assert model.calls == 1


def test_short_text_has_duration_floor(fake):
    model = fake(0.72)
    tts_server._generate("Hi!", "en", None, None)
    assert model.calls == 1


class FakeTurbo:
    sr = SR

    def __init__(self, reject=False):
        self.reject = reject
        self.calls = 0
        self.conds = None

    def prepare_conditionals(self, path):
        assert not self.reject, "Audio prompt must be longer than 5 seconds!"
        self.conds = path

    def generate(self, text, temperature):
        self.calls += 1
        return torch.zeros(1, int(1.3 * SR))


@pytest.fixture
def engines(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(tts_server, "_SPEAKER_CACHE", tmp_path)
    monkeypatch.setattr(tts_server, "_speaker_conds", tts_server.OrderedDict())
    monkeypatch.setattr(tts_server, "_turbo_rejected", set())
    monkeypatch.setattr(FakeModel, "prepare_conditionals", lambda self, path: None, raising=False)

    def install(reject=False):
        turbo = FakeTurbo(reject)
        monkeypatch.setattr(tts_server, "_turbo", turbo)
        return fake(1.3), turbo

    return install


def test_english_cloned_voice_uses_turbo(engines):
    multi, turbo = engines()
    tts_server._generate(TEXT, "en", "voice", b"RIFF")
    assert (turbo.calls, multi.calls) == (1, 0)


@pytest.mark.parametrize("language,key", [("fr", "voice"), ("en", None)])
def test_other_languages_and_default_voice_use_multilingual(engines, language, key):
    multi, turbo = engines()
    tts_server._generate(TEXT, language, key, b"RIFF" if key else None)
    assert (turbo.calls, multi.calls) == (0, 1)


def test_reference_too_short_for_turbo_falls_back(engines):
    multi, turbo = engines(reject=True)
    tts_server._generate(TEXT, "en", "voice", b"RIFF")
    assert (turbo.calls, multi.calls) == (0, 1)
    assert "voice" in tts_server._turbo_rejected
