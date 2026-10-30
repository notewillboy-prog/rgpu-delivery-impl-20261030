"""run.py orchestration with a fake prepared store/worker (no torch, no GPU)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT.parent / "server" / "rgpu_adapter"
sys.path.insert(0, str(ADAPTER))
from run import Deps, execute  # noqa: E402
import weights_validate  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_zip(path: Path, *, family: str = "locate", parent_fp: str = "a" * 64,
               parent_id: str = "model-parent", classes: dict | None = None,
               corrupt_checksum: bool = False) -> None:
    classes = classes or {"cat": "a cat"}
    weights = {"model.safetensors": b"weights-bytes"}
    weight_hashes = {name: _sha(data) for name, data in weights.items()}
    weight_fp = _sha(json.dumps(weight_hashes, sort_keys=True).encode())
    manifest = {
        "model_family": family,
        "artifact_type": "complete_merged_model",
        "standalone": True,
        "quality_status": "needs_review",
        "parent_model_fingerprint": parent_fp,
        "parent_model_version_id": parent_id,
        "weight_fingerprint": weight_fp,
        "weight_hashes": weight_hashes,
    }
    files = {
        "model/model.safetensors": next(iter(weights.values())),
        "adapter/adapter.bin": b"adapter",
        "evaluation.json": json.dumps({"ok": True}).encode(),
        "class_prompts.json": json.dumps(classes).encode(),
        "training_config.json": json.dumps({"steps": 1}).encode(),
        "result_manifest.json": json.dumps(manifest).encode(),
    }
    checksums = {name: _sha(data) for name, data in files.items()}
    if corrupt_checksum:
        checksums["evaluation.json"] = "0" * 64
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("file_checksums.json", json.dumps(checksums))


class FakeStore:
    def __init__(self, root: Path, job: dict) -> None:
        self.root = root
        self.job_data = job
        self.root.mkdir(parents=True, exist_ok=True)

    def job(self, _job_id: str) -> dict:
        return self.job_data


class FakeDeps:
    def __init__(self, root: Path, *, fail: bool = False, corrupt: bool = False) -> None:
        self.root = root
        self.fail = fail
        self.corrupt = corrupt
        self.job = {"id": "job-20261030-000000-abcdef", "state": "QUEUED", "artifacts": {}}
        self.jobs_dir = root / "prepared_runtime" / "store" / "jobs" / self.job["id"]
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        (root / "outputs").mkdir(parents=True, exist_ok=True)

    def prepare(self, root: Path, sha: str) -> dict:
        return {
            "schema": 1, "phase": "prepared_only", "family": "locate",
            "request_id": "req-1", "parent_fingerprint": "a" * 64,
            "bundle_manifest_sha256": sha, "remote_parent_model_id": "model-parent",
            "job_id": self.job["id"], "job_state": "QUEUED",
        }

    def store(self, root: Path) -> FakeStore:
        return FakeStore(root / "prepared_runtime" / "store", self.job)

    def load_job(self, store: FakeStore, _job_id: str) -> dict:
        return self.job

    def run_job(self, _store: FakeStore, _job: dict, _config: dict) -> None:
        if self.fail:
            self.job["state"] = "FAILED"
            self.job["error"] = "fake worker failure"
            return
        weights = self.jobs_dir / "weights.zip"
        _write_zip(weights, corrupt_checksum=self.corrupt)
        self.job["state"] = "COMPLETED"
        self.job["artifacts"] = {
            "weights.zip": {"path": str(weights), "sha256": _sha(weights.read_bytes()), "size": weights.stat().st_size}
        }
        (self.jobs_dir / "events.jsonl").write_text('{"stage":"COMPLETED"}\n', encoding="utf-8")

    def base_path(self, _root: Path, _prepared: dict) -> Path:
        return self.root

    def classes(self, _root: Path) -> dict:
        return {"cat": "a cat"}

    def artifacts(self, _store: FakeStore, job: dict) -> dict:
        return job.get("artifacts") or {}

    def family_package(self, _family: str) -> str:
        return "training_service"

    def settings(self, _family: str = "locate") -> dict:
        return {"root": self.root, "base": self.root, "gpu": "0"}


def _deps(root: Path, **kwargs) -> Deps:
    fake = FakeDeps(root, **kwargs)
    return Deps(
        prepare=fake.prepare, store=fake.store, load_job=fake.load_job,
        run_job=fake.run_job, base_path=fake.base_path, classes=fake.classes,
        artifacts=fake.artifacts, family_package=fake.family_package,
        settings=fake.settings,
    )


def test_execute_success_writes_receipt_and_validates_zip(tmp_path: Path) -> None:
    result = execute(tmp_path, "b" * 64, deps=_deps(tmp_path))
    assert result["ok"] is True, result
    assert result["phase"] == "executed"
    assert result["validation"]["ok"] is True
    assert result["quality_status"] == "needs_review"
    receipt = json.loads((tmp_path / "outputs" / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["job_id"] == result["job_id"]
    assert (tmp_path / "outputs" / "logs" / "events.jsonl").is_file()


def test_second_execute_is_refused(tmp_path: Path) -> None:
    assert execute(tmp_path, "b" * 64, deps=_deps(tmp_path))["ok"] is True
    second = execute(tmp_path, "b" * 64, deps=_deps(tmp_path))
    assert second["ok"] is False
    assert second["exit_code"] == 5


def test_worker_failure_writes_failure_receipt(tmp_path: Path) -> None:
    result = execute(tmp_path, "b" * 64, deps=_deps(tmp_path, fail=True))
    assert result["ok"] is False
    assert result["exit_code"] == 4
    failure = json.loads((tmp_path / "outputs" / "run_failure.json").read_text(encoding="utf-8"))
    assert failure["phase"] == "run_failed"


def test_validation_failure_exit_code(tmp_path: Path) -> None:
    result = execute(tmp_path, "b" * 64, deps=_deps(tmp_path, corrupt=True))
    assert result["ok"] is False
    assert result["exit_code"] == 3
    assert "checksum mismatch" in str(result.get("error"))
