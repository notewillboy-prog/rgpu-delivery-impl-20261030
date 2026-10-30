#!/usr/bin/env python3
"""Read-only proof that a fresh bundle pins the independently patched worker."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PACKAGES = {"locate": "training_service", "sam3": "sam3_training_service"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify(family: str, data_root: Path, patched_root: Path) -> dict:
    package = PACKAGES[family]
    bundle = data_root / "bundles" / family
    receipt_path = data_root / "bundles" / f"{family}.bundle_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_path = bundle / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    worker = bundle / "server" / package / "worker.py"
    source = patched_root / package / "worker.py"
    request = json.loads((bundle / "request.json").read_text(encoding="utf-8"))
    checks = {
        "bundle_exists": bundle.is_dir(),
        "receipt_family": receipt.get("family") == family,
        "manifest_sha_matches_receipt": sha256(manifest_path) == receipt.get("manifest_sha256"),
        "manifest_binds_worker": manifest.get(f"server/{package}/worker.py") == sha256(worker),
        "worker_matches_patched_source": sha256(worker) == sha256(source),
        "receipt_worker_sha": receipt.get("worker", {}).get("bundle_sha256") == sha256(worker),
        "worker_source_sha": receipt.get("worker", {}).get("source_sha256") == sha256(source),
        "worker_has_two_binding_fields": worker.read_text(encoding="utf-8").count(
            "parent_model_fingerprint") == 2,
        "run_sha_matches_receipt": sha256(bundle / "run.py") == receipt.get("scripts", {}).get("run.py"),
        "validator_sha_matches_receipt": sha256(
            bundle / "weights_validate.py") == receipt.get("scripts", {}).get("weights_validate.py"),
        "request_parent_model_id": request.get("parent_model_id") == receipt.get("parent_model_id"),
        "request_parent_model_object_id": request.get("parent_model", {}).get("id") == receipt.get("parent_model_id"),
        "not_consumed": not any((bundle / name).exists() for name in (
            "prepare.events.jsonl", "prepared_runtime", "outputs")),
    }
    return {
        "family": family,
        "bundle": str(bundle.resolve()),
        "receipt": str(receipt_path.resolve()),
        "manifest_sha256": receipt.get("manifest_sha256"),
        "worker_sha256": sha256(worker),
        "checks": checks,
        "ok": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--patched-root", type=Path, required=True)
    parser.add_argument("--only", action="append", choices=tuple(PACKAGES), default=None)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    results = [verify(family, args.data_root.resolve(), args.patched_root.resolve())
               for family in (args.only or list(PACKAGES))]
    report = {"ok": all(item["ok"] for item in results), "results": results}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
