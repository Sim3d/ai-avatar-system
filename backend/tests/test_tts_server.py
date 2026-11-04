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


TEXT = "How can I help you today?"  # 25 chars -> limit ~2.2s


def test_normal_audio_is_returned_without_retry(fake):
    model = fake(1.3)
    wav = tts_server._generate(TEXT, "en", None, None)
    assert model.calls == 1
    assert wav.shape[-1] == int(1.3 * SR)


def test_looping_audio_is_regenerated_and_shorter_kept(fake):
    model = fake(2.52, 1.25)
    wav = tts_server._generate(TEXT, "en", None, None)
    assert model.calls == 2
    assert wav.shape[-1] == int(1.25 * SR)


def test_retry_keeps_first_when_retry_is_longer(fake):
    fake(2.52, 3.0)
    wav = tts_server._generate(TEXT, "en", None, None)
    assert wav.shape[-1] == int(2.52 * SR)


def test_short_text_has_duration_floor(fake):
    model = fake(0.72)
    tts_server._generate("Hi!", "en", None, None)
    assert model.calls == 1
