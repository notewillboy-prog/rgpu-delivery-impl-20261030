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
from run import Deps, RealDeps, execute  # noqa: E402
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
    def __init__(self, root: Path, *, fail: bool = False, corrupt: bool = False,
                 classes: dict | None = None) -> None:
        self.root = root
        self.fail = fail
        self.corrupt = corrupt
        self.expected_classes = classes or {"cat": "a cat"}
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
        _write_zip(weights, classes=self.expected_classes, corrupt_checksum=self.corrupt)
        self.job["state"] = "COMPLETED"
        self.job["artifacts"] = {
            "weights.zip": {"path": str(weights), "sha256": _sha(weights.read_bytes()), "size": weights.stat().st_size}
        }
        (self.jobs_dir / "events.jsonl").write_text('{"stage":"COMPLETED"}\n', encoding="utf-8")

    def base_path(self, _root: Path, _prepared: dict) -> Path:
        return self.root

    def classes(self, _root: Path) -> dict:
        return dict(self.expected_classes)

    def parent_binding(self, _root: Path, _prepared: dict) -> dict:
        return {
            "local_parent_model_id": "local-parent",
            "remote_parent_model_id": "model-parent",
            "parent_model_ids": {"local-parent": "model-parent"},
        }

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
        settings=fake.settings, parent_binding=fake.parent_binding,
    )


def test_execute_success_writes_receipt_and_validates_zip(tmp_path: Path) -> None:
    (tmp_path / "bundle_mode.json").write_text(
        json.dumps({"schema": 1, "mode": "train", "entry_script": "run.py"}), encoding="utf-8"
    )
    result = execute(tmp_path, "b" * 64, deps=_deps(tmp_path))
    assert result["ok"] is True, result
    assert result["phase"] == "executed"
    assert result["validation"]["ok"] is True
    assert result["quality_status"] == "needs_review"
    receipt = json.loads((tmp_path / "outputs" / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["job_id"] == result["job_id"]
    assert receipt["mode"] == "train"
    assert receipt["entry_script"] == "run.py"
    assert receipt["entry"][5] == "run.py"
    assert receipt["entry"][-2:] == ["--bundle-mode", "train"]
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


def test_execute_accepts_new_class_not_only_parent_classes(tmp_path: Path) -> None:
    """P1-3: worker merges class_prompts; validator must compare final map."""
    result = execute(tmp_path, "b" * 64, deps=_deps(tmp_path, classes={"car": "car"}))
    assert result["ok"] is True, result
    assert result["validation"]["class_prompts"] == {"car": "car"}
    receipt = json.loads((tmp_path / "outputs" / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["expected_classes"] == {"car": "car"}
    assert receipt["parent_model_ids"] == {"local-parent": "model-parent"}


@pytest.mark.parametrize("family", ["locate", "sam3"])
def test_real_deps_classes_merges_parent_and_new_classes(tmp_path: Path, family: str) -> None:
    request = {
        "schema": 1,
        "family": family,
        "parent_fingerprint": "a" * 64,
        "parent_classes": {},
        "training": {
            "class_prompts": [{"class_name": "car", "prompt": "car"}],
        },
    }
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    deps = RealDeps(tmp_path)
    assert deps.classes(tmp_path) == {"car": "car"}

    request["parent_classes"] = {"fish": "fish"}
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    assert deps.classes(tmp_path) == {"fish": "fish", "car": "car"}

    request["training"]["class_prompts"] = [{"class_name": "fish", "prompt": "fish"}]
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    assert deps.classes(tmp_path) == {"fish": "fish"}

    request["training"]["class_prompts"] = [{"class_name": "fish", "prompt": "OTHER"}]
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts with parent"):
        deps.classes(tmp_path)


@pytest.mark.parametrize("family", ["locate", "sam3"])
def test_real_deps_classes_replay_mode_merges_old_and_new(
    tmp_path: Path, family: str
) -> None:
    request = {
        "schema": 1,
        "family": family,
        "parent_fingerprint": "a" * 64,
        "parent_classes": {"fish": "fish"},
        "training": {
            "training_mode": "replay",
            "replay_class_prompts": [{"class_name": "fish", "prompt": "fish"}],
            "class_prompts": [{"class_name": "car", "prompt": "car"}],
        },
    }
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    assert RealDeps(tmp_path).classes(tmp_path) == {"fish": "fish", "car": "car"}


def test_real_deps_classes_omitted_mode_merges_replay_prompts(tmp_path: Path) -> None:
    """The worker may infer replay; don't false-reject its old classes."""
    request = {
        "family": "locate",
        "parent_classes": {"fish": "fish"},
        "training": {
            "replay_class_prompts": [{"class_name": "fish", "prompt": "fish"}],
            "class_prompts": [{"class_name": "car", "prompt": "car"}],
        },
    }
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    assert RealDeps(tmp_path).classes(tmp_path) == {"fish": "fish", "car": "car"}


def test_real_deps_classes_rejects_duplicate_class_prompt(tmp_path: Path) -> None:
    request = {
        "family": "locate",
        "parent_classes": {},
        "training": {
            "class_prompts": [
                {"class_name": "car", "prompt": "car"},
                {"class_name": "car", "prompt": "vehicle"},
            ],
        },
    }
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate class"):
        RealDeps(tmp_path).classes(tmp_path)


def test_real_deps_classes_new_only_rejects_replay_prompts(tmp_path: Path) -> None:
    request = {
        "family": "locate",
        "parent_classes": {},
        "training": {
            "training_mode": "new_only",
            "replay_class_prompts": [{"class_name": "fish", "prompt": "fish"}],
            "class_prompts": [{"class_name": "car", "prompt": "car"}],
        },
    }
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="must not carry replay"):
        RealDeps(tmp_path).classes(tmp_path)


def test_real_deps_parent_binding_is_explicit_local_to_remote(tmp_path: Path) -> None:
    request = {
        "family": "locate",
        "parent_model": {"path": "inputs/parent.zip", "sha256": "c" * 64, "id": "local-parent"},
        "parent_fingerprint": "a" * 64,
    }
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    binding = RealDeps(tmp_path).parent_binding(
        tmp_path,
        {"remote_parent_model_id": "model-parent", "parent_fingerprint": "a" * 64},
    )
    assert binding == {
        "local_parent_model_id": "local-parent",
        "remote_parent_model_id": "model-parent",
        "parent_model_ids": {"local-parent": "model-parent"},
    }


def test_real_deps_parent_binding_rejects_missing_local_id(tmp_path: Path) -> None:
    request = {
        "family": "locate",
        "parent_model": {"path": "inputs/parent.zip", "sha256": "c" * 64},
        "parent_fingerprint": "a" * 64,
    }
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="local parent id is missing"):
        RealDeps(tmp_path).parent_binding(
            tmp_path,
            {"remote_parent_model_id": "model-parent", "parent_fingerprint": "a" * 64},
        )


def test_expected_parent_fingerprint_missing_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "weights.zip"
    _write_zip(archive, parent_fp=None)
    with pytest.raises(weights_validate.ValidationError, match="parent_model_fingerprint is missing"):
        weights_validate.validate_weights_zip(
            archive, family="locate", expected_parent_fingerprint="a" * 64
        )


def test_empty_expected_parent_fingerprint_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "weights.zip"
    _write_zip(archive, parent_fp="a" * 64)
    with pytest.raises(weights_validate.ValidationError, match="not a SHA-256"):
        weights_validate.validate_weights_zip(
            archive, family="locate", expected_parent_fingerprint=""
        )


def test_parent_id_mapping_mismatch_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "weights.zip"
    _write_zip(archive, parent_id="other-parent")
    with pytest.raises(weights_validate.ValidationError, match="not in trusted mapping"):
        weights_validate.validate_weights_zip(
            archive,
            family="locate",
            expected_parent_fingerprint="a" * 64,
            expected_parent_model_ids={"local-parent": "model-parent"},
        )
