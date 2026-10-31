"""Regression tests for the sealed bundle mode and parent-class provenance."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
ADAPTER = ROOT / "server" / "rgpu_adapter"
for item in (str(TOOLS), str(ADAPTER)):
    if item not in sys.path:
        sys.path.insert(0, item)

import bundle_mode  # noqa: E402
import parent_identity  # noqa: E402
import prepare as adapter_prepare  # noqa: E402
import run as adapter_run  # noqa: E402


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _metadata(**overrides) -> dict:
    value = {
        "schema": 1,
        "family": "locate",
        "archive_sha256": "a" * 64,
        "approved_base_fingerprint": "b" * 64,
        "parent_fingerprint": "b" * 64,
        "classes": {},
        "source": {"kind": "archive_metadata_scan", "detail": "scanned config"},
        "class_evidence": [],
    }
    value.update(overrides)
    return value


def test_mode_records_bind_exact_entry_script():
    prepare_record = bundle_mode.build_record("prepare_only")
    train_record = bundle_mode.build_record("train")
    assert prepare_record == {"schema": 1, "mode": "prepare_only", "entry_script": "prepare.py"}
    assert train_record == {"schema": 1, "mode": "train", "entry_script": "run.py"}


def test_marker_is_required_when_caller_names_a_mode(tmp_path: Path):
    with pytest.raises(ValueError, match="marker is missing"):
        bundle_mode.enforce(tmp_path, "train", "train")


def test_train_entry_rejects_prepare_only_marker(tmp_path: Path):
    _write(tmp_path / "bundle_mode.json", bundle_mode.build_record("prepare_only"))
    result = adapter_run.execute(tmp_path, "0" * 64)
    assert result["ok"] is False
    assert result["phase"] == "mode_rejected"
    assert "prepare_only" in result["error"]
    assert not (tmp_path / "outputs").exists()


def test_prepare_entry_rejects_train_marker(tmp_path: Path):
    _write(tmp_path / "bundle_mode.json", bundle_mode.build_record("train"))
    with pytest.raises(ValueError, match="train"):
        adapter_prepare.prepare(tmp_path, expected_manifest_sha="0" * 64,
                                requested_mode="prepare_only")


def test_prepare_only_marker_rejects_requested_train_mode(tmp_path: Path):
    _write(tmp_path / "bundle_mode.json", bundle_mode.build_record("prepare_only"))
    with pytest.raises(ValueError, match="mismatch"):
        bundle_mode.enforce(tmp_path, "prepare_only", "train")


def test_explicit_empty_parent_class_map_is_legal(tmp_path: Path):
    path = tmp_path / "meta.json"
    _write(path, _metadata(classes={}))
    loaded = parent_identity.load(path, family="locate", archive_sha256="a" * 64)
    assert loaded["classes"] == {}


def test_nonempty_parent_classes_require_evidence(tmp_path: Path):
    path = tmp_path / "meta.json"
    _write(path, _metadata(classes={"fish": "fish"}))
    with pytest.raises(ValueError, match="class_evidence"):
        parent_identity.load(path, family="locate", archive_sha256="a" * 64)


def test_fabricated_parent_classes_with_evidence_are_still_bound(tmp_path: Path):
    path = tmp_path / "meta.json"
    evidence = {"source": "server/runtime/weight_index", "sha256": "c" * 64,
                "classes": {"pothole": "pothole"}}
    _write(path, _metadata(classes={"pothole": "pothole"}, class_evidence=[evidence]))
    loaded = parent_identity.load(path, family="locate", archive_sha256="a" * 64)
    assert loaded["classes"] == {"pothole": "pothole"}


def test_parent_metadata_archive_mismatch_is_rejected(tmp_path: Path):
    path = tmp_path / "meta.json"
    _write(path, _metadata())
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        parent_identity.load(path, family="locate", archive_sha256="d" * 64)


def test_parent_metadata_without_source_is_rejected(tmp_path: Path):
    path = tmp_path / "meta.json"
    _write(path, _metadata(source={"kind": "manual"}))
    with pytest.raises(ValueError, match="source.detail"):
        parent_identity.load(path, family="locate", archive_sha256="a" * 64)


def test_empty_classes_must_not_carry_evidence(tmp_path: Path):
    path = tmp_path / "meta.json"
    evidence = {"source": "x", "sha256": "c" * 64, "classes": {}}
    _write(path, _metadata(classes={}, class_evidence=[evidence]))
    with pytest.raises(ValueError, match="must not carry"):
        parent_identity.load(path, family="locate", archive_sha256="a" * 64)


def test_parent_content_id_is_accepted_and_bound(tmp_path: Path):
    import hashlib
    payload = tmp_path / "parent.zip"
    payload.write_bytes(b"parent-archive")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    entry = {"path": "parent.zip", "sha256": digest, "id": "sha256:" + digest}
    assert adapter_prepare.checked_file(tmp_path, entry, allow_id=True) == payload
    with pytest.raises(ValueError, match="content-address"):
        adapter_prepare.checked_file(
            tmp_path, {"path": "parent.zip", "sha256": digest, "id": "sha256:" + "0" * 64},
            allow_id=True,
        )


def test_parent_content_id_is_not_accepted_for_non_parent_inputs(tmp_path: Path):
    import hashlib
    payload = tmp_path / "dataset.zip"
    payload.write_bytes(b"dataset")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="identity"):
        adapter_prepare.checked_file(
            tmp_path, {"path": "dataset.zip", "sha256": digest, "id": "sha256:" + digest}
        )
