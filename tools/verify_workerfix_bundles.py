#!/usr/bin/env python3
"""Read-only proof that a fresh bundle pins the worker, mode, and entry."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PACKAGES = {"locate": "training_service", "sam3": "sam3_training_service"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify(family: str, data_root: Path, patched_root: Path, adapter_root: Path) -> dict:
    sys.path.insert(0, str(adapter_root))
    import bundle_mode  # noqa: PLC0415

    package = PACKAGES[family]
    bundle = data_root / "bundles" / family
    receipt_path = data_root / "bundles" / f"{family}.bundle_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_path = bundle / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    worker = bundle / "server" / package / "worker.py"
    source = patched_root / package / "worker.py"
    request = json.loads((bundle / "request.json").read_text(encoding="utf-8"))
    marker_path = bundle / "bundle_mode.json"
    marker = bundle_mode.read_record(bundle) if marker_path.exists() else None
    mode = receipt.get("mode")
    entry = receipt.get("entry") or []
    entry_script = receipt.get("entry_script")
    mode_ok = (
        isinstance(marker, dict)
        and marker.get("mode") == mode
        and marker.get("entry_script") == entry_script
        and mode in bundle_mode.MODES
        and entry_script == bundle_mode.MODES.get(mode)
        and isinstance(entry, list)
        and len(entry) >= 5
        and entry[0] == "python"
        and entry[5] == entry_script
        and entry[-2:] == ["--bundle-mode", mode]
    )
    checks = {
        "bundle_exists": bundle.is_dir(),
        "receipt_family": receipt.get("family") == family,
        "manifest_sha_matches_receipt": sha256(manifest_path) == receipt.get("manifest_sha256"),
        "manifest_binds_mode_marker": manifest.get("bundle_mode.json") == sha256(marker_path),
        "mode_matches_entry": mode_ok,
        "entry_script_sha_matches_receipt": sha256(bundle / entry_script) == receipt.get("entry_script_sha256") if entry_script else False,
        "prepare_sha_matches_receipt": sha256(bundle / "prepare.py") == receipt.get("scripts", {}).get("prepare.py"),
        "run_sha_matches_receipt": sha256(bundle / "run.py") == receipt.get("scripts", {}).get("run.py"),
        "validator_sha_matches_receipt": sha256(
            bundle / "weights_validate.py") == receipt.get("scripts", {}).get("weights_validate.py"),
        "mode_helper_sha_matches_receipt": sha256(
            bundle / "bundle_mode.py") == receipt.get("scripts", {}).get("bundle_mode.py"),
        "manifest_binds_worker": manifest.get(f"server/{package}/worker.py") == sha256(worker),
        "worker_matches_patched_source": sha256(worker) == sha256(source),
        "receipt_worker_sha": receipt.get("worker", {}).get("bundle_sha256") == sha256(worker),
        "worker_source_sha": receipt.get("worker", {}).get("source_sha256") == sha256(source),
        "worker_has_two_binding_fields": worker.read_text(encoding="utf-8").count(
            "parent_model_fingerprint") == 2,
        "request_parent_model_id": request.get("parent_model_id") == receipt.get("parent_model_id"),
        "request_parent_model_object_id": request.get("parent_model", {}).get("id") == receipt.get("parent_model_id"),
        "request_parent_fingerprint": request.get("parent_fingerprint") == receipt.get("parent_fingerprint"),
        "request_approved_base_fingerprint": request.get("approved_base_fingerprint") == receipt.get("approved_base_fingerprint"),
        "parent_metadata_bound": bool(receipt.get("parent_metadata", {}).get("source"))
            and bool(receipt.get("parent_metadata", {}).get("archive_sha256")),
        "parent_class_evidence": bool(receipt.get("parent_classes"))
            == bool(receipt.get("parent_metadata", {}).get("class_evidence")),
        "not_consumed": not any((bundle / name).exists() for name in (
            "prepare.events.jsonl", "prepared_runtime", "outputs")),
    }
    dataset = receipt.get("dataset")
    if dataset:
        checks.update({
            "receipt_dataset_sha": sha256(bundle / "inputs" / "dataset.zip") == dataset.get("sha256"),
            "request_dataset_sha": request.get("dataset", {}).get("sha256") == dataset.get("sha256"),
            "request_parent_classes": request.get("parent_classes") == receipt.get("parent_classes"),
            "request_new_class": any(
                prompt.get("class_name") == receipt.get("new_class", {}).get("name")
                and prompt.get("prompt") == receipt.get("new_class", {}).get("prompt")
                for prompt in request.get("training", {}).get("class_prompts", [])
            ),
        })
    requirements = receipt.get("requirements") or {}
    if requirements:
        checks["requirements_sha"] = sha256(bundle / "requirements.txt") == requirements.get("sha256")
    return {
        "family": family,
        "bundle": str(bundle.resolve()),
        "receipt": str(receipt_path.resolve()),
        "manifest_sha256": receipt.get("manifest_sha256"),
        "mode": mode,
        "entry_script": entry_script,
        "entry": entry,
        "worker_sha256": sha256(worker),
        "checks": checks,
        "ok": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--patched-root", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--only", action="append", choices=tuple(PACKAGES), default=None)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    results = [verify(family, args.data_root.resolve(), args.patched_root.resolve(),
                      args.adapter_root.resolve())
               for family in (args.only or list(PACKAGES))]
    report = {"ok": all(item["ok"] for item in results), "results": results}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
