"""Startup recovery: non-terminal jobs become FAILED, terminal jobs are untouched."""

from __future__ import annotations

from pathlib import Path

from common.models import EnvSpec, JobSpec, Resources
from server.jobstore import JobStore


def _spec(name: str) -> JobSpec:
    return JobSpec(job_name=name, entry_argv=["python", "x.py"], env=EnvSpec(),
                   resources=Resources(timeout_s=10))


def test_recover_interrupted_marks_only_nonterminal(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    running, _fp = store.create_job(_spec("running"))
    done, _fp2 = store.create_job(_spec("done"))
    store.update_state(done, state="succeeded", exit_code=0)
    recovered = store.recover_interrupted("test restart")
    assert recovered == [running]
    state = store.load_state(running)
    assert state["state"] == "failed"
    assert "test restart" in state["error"]
    assert state["finished_at"] is not None
    assert state["updated_at"] >= state["created_at"]
    assert store.load_state(done)["state"] == "succeeded"
    assert store.recover_interrupted() == []
