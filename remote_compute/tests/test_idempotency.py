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
from common.models import EnvSpec, JobSpec, Resources
from server.idempotency import IdempotencyConflict, IdempotencyError, IdempotencyIndex
from server.jobstore import JobStore, canonical_spec_fingerprint


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

    def create(job_id: str) -> None:
        with lock:
            created.append(job_id)

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
    index.get_or_create("k", _fp(), lambda job_id: None)
    with pytest.raises(IdempotencyConflict):
        index.get_or_create("k", _fp("b"), lambda job_id: None)


def test_crash_after_create_before_commit_does_not_duplicate(tmp_path: Path, monkeypatch) -> None:
    """Synthetic P0-2: record fails after create; retry must reuse the id."""
    index = IdempotencyIndex(tmp_path / "idem")
    created: list[str] = []

    def create(job_id: str) -> None:
        created.append(job_id)

    original_record = index.record
    calls = {"n": 0}

    def flaky_record(key, job_id, spec_fingerprint, *, status="committed"):
        calls["n"] += 1
        # First record is the reservation (must succeed).  The second record is
        # the commit after create; fail it, simulating a crash in that window.
        if calls["n"] == 2:
            raise RuntimeError("simulated crash before commit")
        return original_record(key, job_id, spec_fingerprint, status=status)

    monkeypatch.setattr(index, "record", flaky_record)
    with pytest.raises(RuntimeError, match="simulated crash"):
        index.get_or_create("crash-key", _fp(), create)
    assert len(created) == 1
    reserved = index.lookup("crash-key")
    assert reserved is not None and reserved["status"] == "reserved"
    assert reserved["job_id"] == created[0]

    monkeypatch.setattr(index, "record", original_record)
    job_id, reused = index.get_or_create("crash-key", _fp(), create)
    assert job_id == created[0]
    assert reused is False
    # Retry calls create with the same id (idempotent repair), never a new id.
    assert created == [job_id, job_id]
    committed = index.lookup("crash-key")
    assert committed is not None and committed["status"] == "committed"
    assert committed["job_id"] == job_id
    assert len({p.stem for p in index.root.glob("*.json")}) == 1


def test_create_before_commit_crash_leaves_one_job_on_disk(tmp_path: Path, monkeypatch) -> None:
    """P0-2 end-to-end at store level: retry must not create a second job dir."""
    store = JobStore(tmp_path / "store")
    index = IdempotencyIndex(tmp_path / "idem")
    spec = JobSpec(
        job_name="idempotent-crash",
        entry_argv=["python", "x.py"],
        env=EnvSpec(),
        resources=Resources(timeout_s=10),
    )
    fingerprint = canonical_spec_fingerprint(spec)
    original_record = index.record
    calls = {"n": 0}

    def flaky_record(key, job_id, spec_fingerprint, *, status="committed"):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash before commit")
        return original_record(key, job_id, spec_fingerprint, status=status)

    def create(job_id: str) -> None:
        store.create_job(spec, job_id=job_id)

    monkeypatch.setattr(index, "record", flaky_record)
    with pytest.raises(RuntimeError):
        index.get_or_create("crash-key", fingerprint, create)
    jobs_after_crash = sorted(p.name for p in store.jobs_dir.iterdir() if p.is_dir())
    assert len(jobs_after_crash) == 1

    monkeypatch.setattr(index, "record", original_record)
    job_id, reused = index.get_or_create("crash-key", fingerprint, create)
    assert reused is False
    assert job_id == jobs_after_crash[0]
    assert sorted(p.name for p in store.jobs_dir.iterdir() if p.is_dir()) == jobs_after_crash
    assert index.lookup("crash-key")["status"] == "committed"


def test_reserved_without_create_is_repaired_with_same_id(tmp_path: Path) -> None:
    """Synthetic P0-2: crash before create; retry creates the reserved id."""
    index = IdempotencyIndex(tmp_path / "idem")
    index.record("reserved-key", "job-20261030-000000-abcdef", _fp(), status="reserved")
    created: list[str] = []

    def create(job_id: str) -> None:
        created.append(job_id)

    job_id, reused, created_here = index.get_or_create_with_state(
        "reserved-key", _fp(), create)
    assert job_id == "job-20261030-000000-abcdef"
    assert reused is False
    # Repairing a prior reservation must not let strict recovery treat the
    # unknown on-disk job as freshly created by this process.
    assert created_here is False
    assert created == [job_id]
    assert index.lookup("reserved-key")["status"] == "committed"


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
