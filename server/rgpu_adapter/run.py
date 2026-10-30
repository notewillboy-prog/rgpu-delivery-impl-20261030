"""Execute a prepared RGPU training bundle (the missing ``run.py``).

``prepare.py`` deliberately stops after registering the parent model and
dataset.  This adapter is the next step: it loads that prepared store, invokes
the family-specific synchronous worker, validates the exported ``weights.zip``
and writes a durable execution receipt.  It never downloads or claims model
quality; ``quality_status`` stays ``needs_review``.

Usage (inside an already-prepared bundle)::

    python -I -B -X utf8 run.py --manifest-sha256 <bundle_manifest_sha256>

Exit codes: 0 success, 2 prepare/runtime error, 3 artifact validation error,
4 worker failed/timeout, 5 previous execution receipt exists (refuse replay).

The module is importable and takes a ``Deps`` object so the orchestration can
be unit-tested without torch/transformers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from . import weights_validate
except ImportError:  # script execution: ``python run.py``
    import weights_validate  # type: ignore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _write_failure(root: Path, phase: str, exc: BaseException, prepared: Optional[dict] = None) -> dict:
    payload = {
        "schema": 1,
        "phase": phase,
        "ok": False,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "created_at": time.time(),
        "prepared": prepared or {},
        "quality_status": "not_trained",
    }
    _atomic_json(root / "outputs" / "run_failure.json", payload)
    return payload


@dataclass
class Deps:
    """Injectable seam; the real implementation is :class:`RealDeps`."""

    prepare: Any
    store: Any
    load_job: Any
    run_job: Any
    base_path: Any
    classes: Any
    artifacts: Any
    family_package: Any
    settings: Any


class RealDeps:
    """Lazy imports so tests and ``--help`` do not need torch installed."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.server = root / "server"
        self.prepare_module = None

    def _prepare_module(self):
        if self.prepare_module is None:
            add = [str(Path(__file__).resolve().parent), str(self.server), str(self.server / "vendor_sam3")]
            for item in reversed(add):
                if item not in sys.path:
                    sys.path.insert(0, item)
            self.prepare_module = importlib.import_module("prepare")
        return self.prepare_module

    def _modules(self, family: str):
        package = self.family_package(family)
        for item in (str(self.server), str(self.server / "vendor_sam3")):
            if item not in sys.path:
                sys.path.insert(0, item)
        core = importlib.import_module(package + ".core")
        worker = importlib.import_module(package + ".worker")
        assets = importlib.import_module(package + ".assets")
        return package, core, worker, assets

    def prepare(self, root: Path, manifest_sha256: str) -> dict:
        return self._prepare_module().prepare(root=root, expected_manifest_sha=manifest_sha256)

    def family_package(self, family: str) -> str:
        mapping = {"locate": "training_service", "sam3": "sam3_training_service"}
        try:
            return mapping[family]
        except KeyError as exc:
            raise ValueError(f"unknown family: {family!r}") from exc

    def store(self, root: Path):
        prepared = json.loads((root / "outputs" / "prepared.json").read_text(encoding="utf-8"))
        _package, core, _worker, _assets = self._modules(prepared["family"])
        return core.Store(root / "prepared_runtime" / "store")

    def load_job(self, store, job_id: str) -> dict:
        return store.job(job_id)

    def run_job(self, store, job: dict, config: dict) -> None:
        package = self.family_package(job.get("_family") or self._family_from_job(store, job))
        _package, _core, worker, _assets = self._modules(package)
        worker.run_job(store, job, config)

    def _family_from_job(self, store, job: dict) -> str:
        # The prepared manifest lives next to the job; fall back to locate only
        # when the family marker is absent (prepare.py always writes it).
        try:
            root = store.root / "jobs" / job["id"]
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            return str(manifest.get("family") or "locate")
        except (OSError, ValueError):
            return "locate"

    def base_path(self, root: Path, prepared: dict) -> Path:
        _package, _core, _worker, assets = self._modules(prepared["family"])
        return assets.find_root(root / "prepared_runtime" / "approved_base", "config.json")

    def classes(self, root: Path) -> dict:
        request = json.loads((root / "request.json").read_text(encoding="utf-8"))
        classes = request.get("parent_classes")
        if not isinstance(classes, dict):
            raise ValueError("request.parent_classes is missing")
        return classes

    def artifacts(self, store, job: dict) -> dict:
        value = job.get("artifacts")
        if isinstance(value, dict) and value:
            return value
        directory = store.root / "jobs" / job["id"]
        result = {}
        for name in ("weights.zip", "evaluation.json", "result_manifest.json", "events.jsonl"):
            path = directory / name
            if path.is_file():
                result[name] = {"path": str(path), "sha256": _sha256(path), "size": path.stat().st_size}
        return result

    def settings(self) -> dict:
        package = "training_service"
        for item in (str(self.server), str(self.server / "vendor_sam3")):
            if item not in sys.path:
                sys.path.insert(0, item)
        return importlib.import_module(package + ".core").settings()


def _copy_logs(store, job: dict, root: Path) -> None:
    directory = store.root / "jobs" / job["id"]
    target = root / "outputs" / "logs"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("events.jsonl", "failure.log"):
        source = directory / name
        if source.is_file():
            shutil.copy2(source, target / name)


def execute(
    root: str | os.PathLike[str],
    manifest_sha256: str,
    deps: Optional[Deps] = None,
    *,
    validate=weights_validate.validate_weights_zip,
    write_receipt: bool = True,
) -> dict:
    """Run one prepared job and return a receipt-like dict (never raises for job failure)."""
    root = Path(root).expanduser().resolve()
    outputs = root / "outputs"
    receipt_path = outputs / "run_receipt.json"
    if receipt_path.exists():
        return {
            "ok": False,
            "phase": "already_attempted",
            "exit_code": 5,
            "error": f"execution receipt already exists: {receipt_path}",
        }
    deps = deps or Deps(
        prepare=RealDeps(root).prepare,
        store=RealDeps(root).store,
        load_job=RealDeps(root).load_job,
        run_job=RealDeps(root).run_job,
        base_path=RealDeps(root).base_path,
        classes=RealDeps(root).classes,
        artifacts=RealDeps(root).artifacts,
        family_package=RealDeps(root).family_package,
        settings=RealDeps(root).settings,
    )
    prepared: Optional[dict] = None
    try:
        prepared = deps.prepare(root, manifest_sha256)
        if prepared.get("phase") != "prepared_only":
            raise ValueError(f"prepare did not stop at prepared_only: {prepared.get('phase')!r}")
        store = deps.store(root)
        job = deps.load_job(store, prepared["job_id"])
        if job.get("state") not in ("QUEUED", "VALIDATING"):
            raise ValueError(f"prepared job is not runnable: {job.get('state')!r}")
        family = str(prepared["family"])
        package = deps.family_package(family)
        config = deps.settings()
        config["root"] = store.root
        config["base"] = deps.base_path(root, prepared)
        config["gpu"] = os.environ.get("INCREMENTAL_GPU") or os.environ.get("CUDA_VISIBLE_DEVICES") or config.get("gpu", "0")
        job["_family"] = family
        physical = str(config["gpu"]).split(",")[0].strip()
        if physical not in {"0", "1"}:
            raise ValueError(f"GPU 必须映射到物理 0/1，当前 {config['gpu']!r}")
        previous_gpu = os.environ.get("INCREMENTAL_GPU")
        os.environ["INCREMENTAL_GPU"] = physical
        try:
            deps.run_job(store, job, config)
        finally:
            if previous_gpu is None:
                os.environ.pop("INCREMENTAL_GPU", None)
            else:
                os.environ["INCREMENTAL_GPU"] = previous_gpu
        job = deps.load_job(store, prepared["job_id"])
        _copy_logs(store, job, root)
        if job.get("state") != "COMPLETED":
            raise RuntimeError(f"training worker ended in state {job.get('state')!r}: {job.get('error')}")
        artifacts = deps.artifacts(store, job)
        weights = artifacts.get("weights.zip")
        if not weights:
            raise ValueError("worker did not publish weights.zip")
        weights_path = Path(str(weights.get("path") or ""))
        if not weights_path.is_file():
            raise ValueError("weights.zip path is missing")
        expected_classes = deps.classes(root)
        validation = validate(
            weights_path,
            family=family,
            expected_parent_fingerprint=str(prepared["parent_fingerprint"]),
            expected_parent_model_id=str(prepared["remote_parent_model_id"]),
            expected_classes=set(expected_classes),
        )
        receipt = {
            "schema": 1,
            "phase": "executed",
            "ok": True,
            "family": family,
            "request_id": prepared["request_id"],
            "job_id": prepared["job_id"],
            "job_state": job["state"],
            "bundle_manifest_sha256": prepared["bundle_manifest_sha256"],
            "parent_fingerprint": prepared["parent_fingerprint"],
            "remote_parent_model_id": prepared["remote_parent_model_id"],
            "weights": {
                "path": str(weights_path),
                "sha256": weights.get("sha256") or _sha256(weights_path),
                "size": weights_path.stat().st_size,
            },
            "validation": validation,
            "quality_status": "needs_review",
            "automatic_quality_approval": False,
            "created_at": time.time(),
        }
        if write_receipt:
            _atomic_json(receipt_path, receipt)
        return receipt
    except weights_validate.ValidationError as exc:
        failure = _write_failure(root, "validation_failed", exc, prepared)
        failure["exit_code"] = 3
        return failure
    except FileExistsError as exc:
        failure = _write_failure(root, "already_attempted", exc, prepared)
        failure["exit_code"] = 5
        return failure
    except Exception as exc:  # noqa: BLE001 - CLI boundary must return a receipt
        failure = _write_failure(root, "run_failed", exc, prepared)
        failure["exit_code"] = 2 if prepared is None else 4
        return failure


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Execute a prepared RGPU training bundle")
    parser.add_argument("--manifest-sha256", required=True,
                        help="verified bundle_manifest.json SHA-256 from the sender")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent),
                        help="bundle root (default: this script's directory)")
    args = parser.parse_args(argv)
    result = execute(args.root, args.manifest_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else int(result.get("exit_code", 2))


if __name__ == "__main__":
    sys.exit(main())
