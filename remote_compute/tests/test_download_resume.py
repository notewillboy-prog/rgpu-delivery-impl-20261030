"""Download .part resume: keep bytes on disconnect, verify prefix, retry from 0 on corruption."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from client.connection import ConnectionLost
from client.runner import JobRunner
from common.models import OutputEntry


class FakeOutputConn:
    def __init__(self, data: bytes, *, fail_at: int | None = None) -> None:
        self.data = data
        self.fail_at = fail_at
        self.offsets: list[int] = []
        self._chunk = b""
        self._eof = False

    def send_json(self, message):
        offset = int(message["offset"])
        self.offsets.append(offset)
        if self.fail_at is not None and offset >= self.fail_at:
            raise ConnectionLost("simulated disconnect")
        self._chunk = self.data[offset:offset + 4]
        self._eof = offset + len(self._chunk) >= len(self.data)

    def wait_for(self, _kinds, on_message=None):
        if on_message is not None:
            on_message({"kind": "job_progress"})
        return {
            "kind": "output_chunk",
            "length": len(self._chunk),
            "sha256": hashlib.sha256(self._chunk).hexdigest(),
            "eof": self._eof,
        }

    def recv_blob(self):
        return self._chunk


def _runner(data: bytes, *, fail_at: int | None = None):
    runner = object.__new__(JobRunner)
    runner.job_id = "job-20261030-000000-abcdef"
    runner.conn = FakeOutputConn(data, fail_at=fail_at)
    runner.callbacks = SimpleNamespace(log=lambda _message: None)
    runner._handle_passive = lambda _message: None
    return runner


def _entry(data: bytes) -> OutputEntry:
    return OutputEntry(path="out.bin", size=len(data), sha256=hashlib.sha256(data).hexdigest())


def test_resumes_from_partial_file(tmp_path: Path) -> None:
    data = b"0123456789abcdef"
    entry = _entry(data)
    target = tmp_path / "out.bin"
    (tmp_path / "out.bin.part").write_bytes(data[:5])
    runner = _runner(data)
    record = runner._download_one(entry, tmp_path, lambda _n: None)
    assert runner.conn.offsets[0] == 5
    assert target.read_bytes() == data
    assert record["verified"] is True
    assert not (tmp_path / "out.bin.part").exists()


def test_disconnect_keeps_part_file(tmp_path: Path) -> None:
    data = b"abcdefghijklmnop"
    entry = _entry(data)
    (tmp_path / "out.bin.part").write_bytes(data[:4])
    runner = _runner(data, fail_at=4)
    with pytest.raises(ConnectionLost):
        runner._download_one(entry, tmp_path, lambda _n: None)
    assert (tmp_path / "out.bin.part").read_bytes() == data[:4]


def test_corrupt_prefix_is_retried_from_zero(tmp_path: Path) -> None:
    data = b"0123456789abcdef"
    entry = _entry(data)
    (tmp_path / "out.bin.part").write_bytes(b"XXXXX")
    runner = _runner(data)
    runner._download_one(entry, tmp_path, lambda _n: None)
    assert runner.conn.offsets[0] == 5
    assert 0 in runner.conn.offsets
    assert (tmp_path / "out.bin").read_bytes() == data

