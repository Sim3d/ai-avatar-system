"""
Warm-up contract: the client is told when a session's avatar/voice warm-up
starts and ends (it locks input meanwhile), concurrent warms of one avatar
share a single GPU job, and the boot prewarm never takes the app down.
"""

import asyncio

import pytest

from app import websocket as ws_module
from app.services.animator import AvatarAnimator
from app.websocket import ConnectionManager


class _FakeWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, message):
        self.sent.append(message)


def _wire(manager: ConnectionManager, image="/tmp/a.jpg", wav=None):
    ws = _FakeWS()
    manager.active_connections["s1"] = ws
    manager._send_locks["s1"] = asyncio.Lock()
    manager.session_data["s1"] = {"avatar_image_local": image, "voice_wav": wav}
    return ws


async def _drain(manager: ConnectionManager):
    await asyncio.gather(*list(manager._warm_tasks))


@pytest.mark.asyncio
async def test_warmup_sends_start_then_done(monkeypatch):
    m = ConnectionManager()
    ws = _wire(m)
    calls = []

    async def fake_warm(path):
        calls.append(path)

    monkeypatch.setattr(ws_module.avatar_animator, "is_avatar_warm", lambda p: False)
    monkeypatch.setattr(ws_module.avatar_animator, "warm_avatar", fake_warm)

    m._start_warmup("s1")
    await _drain(m)

    assert calls == ["/tmp/a.jpg"]
    assert ws.sent == [
        {"type": "warmup", "status": "start"},
        {"type": "warmup", "status": "done"},
    ]


@pytest.mark.asyncio
async def test_warmup_done_sent_even_if_warm_fails(monkeypatch):
    m = ConnectionManager()
    ws = _wire(m)

    async def boom(path):
        raise RuntimeError("gpu gone")

    monkeypatch.setattr(ws_module.avatar_animator, "is_avatar_warm", lambda p: False)
    monkeypatch.setattr(ws_module.avatar_animator, "warm_avatar", boom)

    m._start_warmup("s1")
    await asyncio.gather(*list(m._warm_tasks), return_exceptions=True)

    assert ws.sent[-1] == {"type": "warmup", "status": "done"}


@pytest.mark.asyncio
async def test_no_warmup_event_when_already_warm(monkeypatch):
    m = ConnectionManager()
    ws = _wire(m)
    monkeypatch.setattr(ws_module.avatar_animator, "is_avatar_warm", lambda p: True)

    m._start_warmup("s1")

    assert not m._warm_tasks
    assert ws.sent == []


@pytest.mark.asyncio
async def test_concurrent_warm_avatar_runs_once(monkeypatch, tmp_path):
    animator = AvatarAnimator()
    runs = 0
    release = asyncio.Event()

    async def fake_once(path, key):
        nonlocal runs
        runs += 1
        await release.wait()
        animator._warmed_avatars.add(key)

    monkeypatch.setattr(animator, "_warm_avatar_once", fake_once)
    img = str(tmp_path / "a.jpg")

    first = asyncio.create_task(animator.warm_avatar(img))
    second = asyncio.create_task(animator.warm_avatar(img))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(first, second)
    await animator.warm_avatar(img)

    assert runs == 1
    assert animator.is_avatar_warm(img)


@pytest.mark.asyncio
async def test_prewarm_recent_avatars_never_raises(monkeypatch):
    m = ConnectionManager()

    class _Broken:
        async def __aenter__(self):
            raise RuntimeError("db down")

        async def __aexit__(self, *a):
            return False

    import app.database as database

    monkeypatch.setattr(database, "AsyncSessionLocal", lambda: _Broken())
    await m.prewarm_recent_avatars()
