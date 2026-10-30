"""job_status exposes updated_at/log_tail/error/gpus for stale and diagnostics."""

from __future__ import annotations

from pathlib import Path

from common.models import EnvSpec, JobSpec, Resources
from server.jobstore import JobStore
from server.runtime import JobRuntime, status_payload_from_state


def test_status_payload_has_diagnostic_fields(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    spec = JobSpec(job_name="status", entry_argv=["python", "x.py"], env=EnvSpec(),
                   resources=Resources(timeout_s=10))
    job_id, _fp = store.create_job(spec)
    runtime = JobRuntime(store, job_id, spec, store.load_state(job_id))
    runtime.note = "boom"
    runtime.gpus = [0]
    runtime.persist()
    state = store.load_state(job_id)
    assert state["updated_at"] >= state["created_at"]
    payload = status_payload_from_state(state)
    assert payload["updated_at"] == state["updated_at"]
    assert payload["error"] == "boom"
    assert payload["gpus"] == [0]
    assert payload["log_tail"] == []
