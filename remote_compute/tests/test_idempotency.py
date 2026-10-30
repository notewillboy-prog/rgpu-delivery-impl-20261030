"""Server-side idempotency index + wire-level duplicate submit tests."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import helpers
import rawflow
from client import packager
from server.idempotency import IdempotencyConflict, IdempotencyError, IdempotencyIndex


def _fp(seed: str = "a") -> str:
    return seed * 64


def test_record_lookup_and_conflict(tmp_path: Path) -> None:
    index = IdempotencyIndex(tmp_path / "idem")
    assert index.lookup("k1") is None
    index.record("k1", "job-20261030-000000-abcdef", _fp())
    record = index.lookup("k1")
    assert record and record["job_id"] == "job-20261030-000000-abcdef"
    with pytest.raises(IdempotencyConflict):
        with index.guard("k1"):
            existing = index.lookup("k1")
            if existing["spec_fingerprint"] != _fp("b"):
                raise IdempotencyConflict()


def test_get_or_create_only_once_concurrent(tmp_path: Path) -> None:
    index = IdempotencyIndex(tmp_path / "idem")
    created: list[str] = []
    lock = threading.Lock()

    def create() -> str:
        with lock:
            created.append("x")
        return "job-20261030-000000-abcdef"

    results = []

    def worker() -> None:
        results.append(index.get_or_create("same", _fp(), create))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(created) == 1
    assert {reused for _job, reused in results} == {False, True}
    assert len({job for job, _reused in results}) == 1


def test_get_or_create_rejects_different_spec(tmp_path: Path) -> None:
    index = IdempotencyIndex(tmp_path / "idem")
    index.get_or_create("k", _fp(), lambda: "job-20261030-000000-abcdef")
    with pytest.raises(IdempotencyConflict):
        index.get_or_create("k", _fp("b"), lambda: "job-20261030-000000-999999")


def test_corrupt_or_tampered_record_is_rejected(tmp_path: Path) -> None:
    index = IdempotencyIndex(tmp_path / "idem")
    path = index.record_path("k")
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(IdempotencyError):
        index.lookup("k")
    path.write_text(json.dumps({"key": "other", "job_id": "job-x", "spec_fingerprint": _fp()}), encoding="utf-8")
    with pytest.raises(IdempotencyError):
        index.lookup("k")


def test_key_validation_and_path_is_hashed(tmp_path: Path) -> None:
    index = IdempotencyIndex(tmp_path / "idem")
    with pytest.raises(ValueError):
        index.record_path("")
    with pytest.raises(ValueError):
        index.record_path("a" * 201)
    with pytest.raises(ValueError):
        index.record_path("../escape")
    path = index.record_path("normal-key")
    assert path.parent == index.root
    assert path.name.endswith(".json") and len(path.stem) == 64


def _server(tmp_path: Path):
    return helpers.ServerProcess(
        port=helpers.find_free_port(),
        token="idem-token",
        data_dir=tmp_path / "server",
        ready_timeout=20.0,
    ).start()


def _spec(project: Path, key: str):
    spec = packager.make_job_spec(
        project,
        ["python", "train.py", "--epochs", "1", "--data", "data.csv",
         "--out", "outputs", "--model", "model.json"],
        timeout_s=60,
        meta={"idempotency_key": key},
    )
    return spec


def test_wire_duplicate_submit_reuses_same_job(tmp_path: Path) -> None:
    server = _server(tmp_path)
    project = helpers.copy_example_project(tmp_path / "proj")
    spec = _spec(project, "dup-key")
    try:
        client, _hello = rawflow.connect(f"127.0.0.1:{server.port}", token="idem-token")
        try:
            first = rawflow.submit(client, spec)
        finally:
            client.close()
        client, _hello = rawflow.connect(f"127.0.0.1:{server.port}", token="idem-token")
        try:
            second = rawflow.submit(client, spec)
            assert second["job_id"] == first["job_id"]
            assert second["reused"] is True
            assert second["state"] == "uploading"
        finally:
            client.close()
        jobs = [p.name for p in (tmp_path / "server" / "jobs").iterdir() if p.is_dir()]
        assert len(jobs) == 1
    finally:
        server.stop(cleanup_data_dir=True)


def test_wire_different_spec_same_key_is_conflict(tmp_path: Path) -> None:
    server = _server(tmp_path)
    project = helpers.copy_example_project(tmp_path / "proj")
    spec = _spec(project, "conflict-key")
    changed = _spec(project, "conflict-key")
    changed.job_name = "different-name"
    try:
        client, _hello = rawflow.connect(f"127.0.0.1:{server.port}", token="idem-token")
        try:
            rawflow.submit(client, spec)
            client.send_json({"kind": "job_submit", "spec": changed.to_dict()})
            error = client.recv_json()
            assert error.get("kind") == "error"
            assert error.get("code") == "CONFLICT"
        finally:
            client.close()
    finally:
        server.stop(cleanup_data_dir=True)
