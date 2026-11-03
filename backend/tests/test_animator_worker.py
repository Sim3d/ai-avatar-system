"""
Regression test for issue #5 — missing musetalk_worker.py.

musetalk_worker.py is our custom persistent-worker driver, not part of the
upstream MuseTalk repo that setup_musetalk.sh clones. It's gitignored under
models/, so a fresh clone never had it and MuseTalk silently fell back to the
static-image engine for every user. We now ship a tracked copy at
backend/musetalk_worker.py and resolve to it when the clone lacks the script.
"""

import asyncio
import json
from pathlib import Path

from app.services.animator import AvatarAnimator


def test_tracked_worker_script_is_shipped():
    """The tracked worker must exist in the repo (outside gitignored models/)."""
    tracked = Path(__file__).resolve().parent.parent / "musetalk_worker.py"
    assert tracked.is_file(), "backend/musetalk_worker.py is missing — MuseTalk will not start"


def test_resolve_worker_prefers_clone_then_falls_back(tmp_path):
    animator = AvatarAnimator()

    # Clone has its own copy → use it.
    clone = tmp_path / "MuseTalk"
    (clone / "scripts").mkdir(parents=True)
    in_clone = clone / "scripts" / "musetalk_worker.py"
    in_clone.write_text("# worker")
    assert animator._resolve_worker_script(clone) == in_clone

    # Clone lacks the script → fall back to the tracked backend copy.
    empty_clone = tmp_path / "EmptyClone"
    (empty_clone / "scripts").mkdir(parents=True)
    resolved = animator._resolve_worker_script(empty_clone)
    assert resolved.name == "musetalk_worker.py"
    assert resolved.is_file()
    # It's the tracked backend copy, not anything under the empty clone.
    assert "EmptyClone" not in str(resolved)


class _FakeStdin:
    def __init__(self):
        self.jobs = []

    def write(self, data: bytes):
        self.jobs.append(json.loads(data.decode()))

    async def drain(self):
        pass


class _FakeStdout:
    def __init__(self, stdin: _FakeStdin, stale: list):
        self._stdin = stdin
        self._lines = [json.dumps(s).encode() + b"\n" for s in stale]

    async def readline(self):
        if self._lines:
            return self._lines.pop(0)
        job = self._stdin.jobs[-1]
        return json.dumps({"id": job["id"], "status": "ok", "output": job["output"]}).encode() + b"\n"


class _FakeProc:
    def __init__(self, stale):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(self.stdin, stale)

    def kill(self):
        pass


def test_worker_infer_discards_stale_replies(tmp_path):
    """A reply left behind by a cancelled turn must not be taken as this turn's result."""
    animator = AvatarAnimator()
    proc = _FakeProc(stale=[{"id": "old", "status": "error", "msg": "previous turn failed"}])

    async def fake_ensure():
        return proc

    animator._ensure_worker = fake_ensure  # type: ignore[method-assign]
    out = tmp_path / "out.mp4"
    result = asyncio.run(
        animator._worker_infer(str(tmp_path / "a.jpg"), str(tmp_path / "a.wav"), str(out), None)
    )
    assert result == str(out)
