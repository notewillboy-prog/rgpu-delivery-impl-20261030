"""P0-1 regression: strict recovery must not be bypassed by idempotency attach."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import ServerConfig
from common.models import EnvSpec, JobSpec, Resources
from common.protocol import ERR_CONFLICT, ProtocolError
from server.app import ServerApp
from server.jobstore import canonical_spec_fingerprint


def _spec(name: str = "disk-queued") -> JobSpec:
    return JobSpec(
        job_name=name,
        entry_argv=["python", "x.py"],
        env=EnvSpec(),
        resources=Resources(timeout_s=10),
    )


def _app(tmp_path: Path, *, strict: bool = True) -> ServerApp:
    cfg = ServerConfig(
        host="127.0.0.1",
        port=0,
        token="test-token",
        data_dir=str(tmp_path / "server"),
        strict_recovery=strict,
        recover_interrupted=False,
    )
    return ServerApp(cfg, gpus=[])


def test_attach_runtime_refuses_unknown_nonterminal_under_strict_recovery(tmp_path: Path) -> None:
    app = _app(tmp_path)
    job_id, fingerprint = app.store.create_job(_spec())
    # The job exists on disk and is queued/uploading, but this process never
    # registered a runtime (equivalent to an unknown post-restart job).
    assert app.find_runtime(job_id) is None
    with pytest.raises(ProtocolError) as caught:
        app.attach_runtime(job_id, fingerprint)
    assert caught.value.code == ERR_CONFLICT
    assert app.find_runtime(job_id) is None
    # Therefore job_start cannot reach scheduler.submit through this path.


def test_attach_runtime_still_allows_terminal_job_under_strict_recovery(tmp_path: Path) -> None:
    app = _app(tmp_path)
    job_id, fingerprint = app.store.create_job(_spec())
    app.store.update_state(job_id, state="succeeded", exit_code=0)
    runtime = app.attach_runtime(job_id, fingerprint)
    assert runtime.job_id == job_id
    assert runtime.state.value == "succeeded"


def test_strict_recovery_rejects_repaired_reservation(tmp_path: Path) -> None:
    """A prior-process reservation is repaired, but not trusted as new."""
    app = _app(tmp_path)
    spec = _spec("reserved-repair")
    fingerprint = canonical_spec_fingerprint(spec)
    app.idempotency.record(
        "reserved-key", "job-20261030-000000-abcdef", fingerprint, status="reserved"
    )
    job_id, reused, created_here = app.idempotency.get_or_create_with_state(
        "reserved-key",
        fingerprint,
        lambda reserved_id: app.store.create_job(spec, job_id=reserved_id),
    )
    assert job_id == "job-20261030-000000-abcdef"
    assert reused is False
    assert created_here is False
    with pytest.raises(ProtocolError) as caught:
        app.attach_runtime(job_id, fingerprint, created_here=created_here)
    assert caught.value.code == ERR_CONFLICT
    assert app.find_runtime(job_id) is None


def test_idempotency_created_runtime_can_attach_but_unknown_one_cannot(tmp_path: Path) -> None:
    app = _app(tmp_path)
    spec = _spec("idem-strict")
    fingerprint = canonical_spec_fingerprint(spec)
    job_id, reused, created_here = app.idempotency.get_or_create_with_state(
        "strict-key",
        fingerprint,
        lambda reserved_id: app.store.create_job(spec, job_id=reserved_id),
    )
    assert created_here is True
    assert reused is False
    runtime = app.attach_runtime(job_id, fingerprint, created_here=created_here)
    assert runtime.job_id == job_id
    # A fresh app over the same data dir has no runtime for that non-terminal
    # job; strict recovery must refuse the same attach instead of enqueueing it.
    app2 = _app(tmp_path)
    with pytest.raises(ProtocolError) as caught:
        app2.attach_runtime(job_id, fingerprint)
    assert caught.value.code == ERR_CONFLICT
    assert app2.find_runtime(job_id) is None
