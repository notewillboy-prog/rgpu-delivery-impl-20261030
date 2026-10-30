"""Opt-in client receipt wiring: first send is allowed, retry is refused."""

from __future__ import annotations

import json
from pathlib import Path

from client.durable_receipt import DurableReceipt
from client.runner import Callbacks, JobRunner
from common.config import ClientConfig
from common.models import EnvSpec, JobSpec, Resources


class FakeConn:
    def __init__(self) -> None:
        self.config = ClientConfig(host="127.0.0.1", port=1234, chunk_size=1024)
        self.sent: list[dict] = []

    def send_json(self, message):
        self.sent.append(message)

    def wait_for(self, _kinds, on_message=None):
        kind = self.sent[-1].get("kind")
        if kind == "job_submit":
            return {"kind": "job_accepted", "job_id": "job-20261030-000000-abcdef",
                    "missing": [], "present": [], "env": {"fingerprint": "f" * 64}}
        if kind == "job_start":
            return {"kind": "job_started", "job_id": "job-20261030-000000-abcdef",
                    "env": {"state": "ready"}, "attached": False}
        raise AssertionError(f"unexpected wait_for after {kind}")


def _spec() -> JobSpec:
    return JobSpec(
        job_name="receipt-test",
        entry_argv=["python", "train.py"],
        env=EnvSpec(),
        resources=Resources(timeout_s=30),
        outputs=["outputs/**"],
    )


def test_receipt_allows_one_submit_and_refuses_retry(tmp_path: Path) -> None:
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    conn = FakeConn()
    runner = JobRunner(conn, _spec(), tmp_path, callbacks=Callbacks(),
                       receipt_dir=str(receipt_dir))
    first = runner.run(watch=False)
    assert first.ok is True
    assert first.state == "started"
    receipt = DurableReceipt.for_request(receipt_dir, runner.receipt_request_id)
    current = json.loads(receipt.path.read_text(encoding="utf-8"))
    assert current["job_id"] == "job-20261030-000000-abcdef"
    assert current["state"] == "tracking"

    conn2 = FakeConn()
    runner2 = JobRunner(conn2, _spec(), tmp_path, callbacks=Callbacks(),
                        receipt_dir=str(receipt_dir))
    second = runner2.run(watch=False)
    assert second.ok is False
    assert second.client_exit == 4
    assert conn2.sent == []
