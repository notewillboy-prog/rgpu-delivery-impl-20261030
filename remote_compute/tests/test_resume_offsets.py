"""Resume offsets: uploader starts at ``received`` and whole-file retry resets to 0."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from client.connection import RemoteError
from client.uploader import upload_file
from common import protocol
from common.models import FileEntry


class FakeConn:
    def __init__(self, initial: bytes = b"", *, fail_done_once: bool = False) -> None:
        self.file = bytearray(initial)
        self.offsets: list[int] = []
        self.fail_done_once = fail_done_once
        self._last = ""

    def send_chunk(self, header, data):
        assert header["kind"] == protocol.KIND_FILE_CHUNK
        offset = int(header["offset"])
        self.offsets.append(offset)
        if offset > len(self.file):
            raise RemoteError(protocol.ERR_BAD_REQUEST, f"offset {offset} > {len(self.file)}")
        self.file[offset:offset + len(data)] = data
        self._last = "chunk"

    def send_json(self, message):
        assert message["kind"] == protocol.KIND_FILE_DONE
        self._last = "done"

    def wait_for(self, _kinds):
        if self._last == "chunk":
            return {"kind": protocol.KIND_FILE_OK, "offset": len(self.file)}
        if self._last == "done":
            if self.fail_done_once:
                self.fail_done_once = False
                raise RemoteError(protocol.ERR_HASH_MISMATCH, "simulated whole-file mismatch")
            actual = hashlib.sha256(self.file).hexdigest()
            return {"kind": protocol.KIND_FILE_OK, "done": True, "sha256": actual}
        raise AssertionError("no request before wait_for")


def _entry(path: Path, rel: str) -> FileEntry:
    data = path.read_bytes()
    return FileEntry(path=rel, size=len(data), sha256=hashlib.sha256(data).hexdigest())


def test_upload_resumes_at_received_offset(tmp_path: Path) -> None:
    payload = b"0123456789abcdef"
    local = tmp_path / "f.bin"
    local.write_bytes(payload)
    conn = FakeConn(initial=payload[:5])
    entry = _entry(local, "f.bin")
    upload_file(conn, entry, tmp_path, chunk_size=4, start_offsets={"f.bin": 5})
    assert conn.offsets[0] == 5
    assert bytes(conn.file) == payload


def test_whole_file_retry_resets_to_zero(tmp_path: Path) -> None:
    payload = b"abcdefghijklmnop"
    local = tmp_path / "f.bin"
    local.write_bytes(payload)
    conn = FakeConn(initial=payload[:4], fail_done_once=True)
    entry = _entry(local, "f.bin")
    upload_file(conn, entry, tmp_path, chunk_size=4, start_offsets={"f.bin": 4})
    assert conn.offsets[0] == 4
    assert 0 in conn.offsets[1:], conn.offsets
    assert bytes(conn.file) == payload


def test_start_offset_is_clamped(tmp_path: Path) -> None:
    payload = b"xyz"
    local = tmp_path / "f.bin"
    local.write_bytes(payload)
    conn = FakeConn(initial=payload)
    entry = _entry(local, "f.bin")
    upload_file(conn, entry, tmp_path, chunk_size=2, start_offsets={"f.bin": 999})
    assert conn.offsets == []
    assert bytes(conn.file) == payload
