"""GET /api/v1/avatars/{id}/intro — spoken avatar introduction, cached in storage."""

from pathlib import Path

import pytest
from httpx import AsyncClient

from app.models import Avatar
from app.services import avatar_intro
from app.services.tts import SynthResult


class _FakeStorage:
    def __init__(self):
        self.files: dict[str, bytes] = {}

    async def download_file(self, key: str) -> bytes:
        if key not in self.files:
            raise FileNotFoundError(key)
        return self.files[key]

    async def upload_file(self, data: bytes, key: str, content_type: str = "", metadata=None):
        self.files[key] = data
        return key


class _FakeTTS:
    def __init__(self, voice_cloned: bool = False):
        self.calls: list[dict] = []
        self.voice_cloned = voice_cloned

    async def synthesize(self, text, output_path, speaker_wav=None, language="en", voice_id=None):
        self.calls.append({"text": text, "speaker_wav": speaker_wav, "voice_id": voice_id})
        Path(output_path).write_bytes(b"RIFF-fake-" + text.encode())
        return SynthResult(
            output_path=output_path,
            engine="chatterbox",
            fallback=False,
            voice_cloned=self.voice_cloned,
        )


@pytest.fixture
def fakes(monkeypatch):
    storage, tts = _FakeStorage(), _FakeTTS()
    monkeypatch.setattr(avatar_intro, "storage_service", storage)
    monkeypatch.setattr(avatar_intro, "tts_service", tts)
    return storage, tts


async def _make_avatar(db_session, user, metadata):
    avatar = Avatar(
        user_id=user.id,
        name="Intro Avatar",
        image_url="http://x/i.jpg",
        s3_key="avatars/x/image.jpg",
        status="ready",
        avatar_metadata=metadata,
    )
    db_session.add(avatar)
    await db_session.commit()
    await db_session.refresh(avatar)
    return avatar


@pytest.mark.asyncio
async def test_intro_is_synthesized_once_then_served_from_cache(
    client: AsyncClient, db_session, test_user, auth_headers, fakes
):
    storage, tts = fakes
    avatar = await _make_avatar(db_session, test_user, {"description": "Ask me about Revit."})

    first = await client.get(f"/api/v1/avatars/{avatar.id}/intro", headers=auth_headers)
    second = await client.get(f"/api/v1/avatars/{avatar.id}/intro", headers=auth_headers)

    assert first.status_code == 200
    assert first.headers["content-type"] == "audio/wav"
    assert second.content == first.content
    assert [call["text"] for call in tts.calls] == ["Ask me about Revit."]
    assert len(storage.files) == 1


@pytest.mark.asyncio
async def test_intro_returns_404_without_description(
    client: AsyncClient, db_session, test_user, auth_headers, fakes
):
    avatar = await _make_avatar(db_session, test_user, {"role": "Support"})

    resp = await client.get(f"/api/v1/avatars/{avatar.id}/intro", headers=auth_headers)

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_intro_cache_key_changes_with_description_and_voice():
    base = avatar_intro.intro_key("a", "Hello there.", None)

    assert avatar_intro.intro_key("a", "Hello there!", None) != base
    assert avatar_intro.intro_key("a", "Hello there.", "voice-1") != base


@pytest.mark.asyncio
async def test_metadata_accepts_description(
    client: AsyncClient, db_session, test_user, auth_headers, fakes
):
    avatar = await _make_avatar(db_session, test_user, {})

    resp = await client.patch(
        f"/api/v1/avatars/{avatar.id}/metadata",
        json={"description": "I help with BIM coordination."},
        headers=auth_headers,
    )

    assert resp.status_code == 200
    assert resp.json()["avatar_metadata"]["description"] == "I help with BIM coordination."
