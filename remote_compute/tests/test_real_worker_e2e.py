"""Offline end-to-end binding: REAL worker packaging -> new run.py validator.

The GPU is replaced with a restricted fake engine, but the worker source itself is
the patched copy under ``vendor/rgpu_worker_patch`` and its own ZIP packaging code
is executed verbatim.  No test code edits the resulting ZIP or adds the parent
fingerprint by hand.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "rgpu_worker_patch"
UPSTREAM = ROOT.parents[1] / "server"
FAMILIES = {"locate": "training_service", "sam3": "sam3_training_service"}
PARENT_FP = "1" * 64
WRONG_FP = "2" * 64
LOCAL_ID = "local-parent-model-1"
REMOTE_ID = "remote-parent-model-1"


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_run_module():
    path = ROOT / "server" / "rgpu_adapter" / "run.py"
    spec = importlib.util.spec_from_file_location("rgpu_run_e2e", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _model_fingerprint(path):
    root = Path(path)
    files = sorted(root.glob("*.safetensors")) if root.is_dir() else []
    if not files:
        # Parent archive probe in the real service fingerprints all model files;
        # this restricted harness only needs the trusted parent identity.
        return PARENT_FP, {}
    hashes = {item.name: _sha(item) for item in files}
    fingerprint = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return fingerprint, hashes


class _DiskUsage:
    def __init__(self, free):
        self.free = free


class _FakeModel:
    def state_dict(self):
        return {"weight": "fake"}


class _FakeEngine:
    def __init__(self, path, parameter, seed):
        self.path = Path(path)
        self.parameter = parameter
        self.seed = seed
        self.model = _FakeModel()

    def evaluate(self, rows, output, callback, stage):
        names = sorted({str(row["class_name"]) for row in rows})
        return {name: {"f1": 0.5, "invalid_boxes": 0} for name in names}

    def train(self, manifest, directory, callback):
        _write_json(Path(directory) / "training.json", {"fake": True})
        return {"steps": 1, "training_mode": manifest.get("training_mode", "new_only")}

    def export(self, destination, base):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text("{}", encoding="utf-8")
        (destination / "model.safetensors").write_bytes(b"fake-weights")

    def close(self):
        pass


class _FakeCursor:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row


class _FakeConnection:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=()):
        if sql.startswith("SELECT data"):
            return _FakeCursor({"data": json.dumps(self.store.job_data)})
        if sql.startswith("INSERT INTO objects"):
            self.store.inserted.append((params[0], params[1], json.loads(params[2])))
        return _FakeCursor(None)


class _FakeStore:
    def __init__(self, root: Path, parent: dict, job: dict):
        self.root = Path(root)
        self.parent = parent
        self.job_data = job
        self.inserted = []

    def get(self, collection, identifier):
        assert collection == "models"
        assert identifier == self.parent["id"]
        return self.parent

    def job(self, job_id):
        assert job_id == self.job_data["id"]
        return dict(self.job_data)

    def update_job(self, job_id, **changes):
        assert job_id == self.job_data["id"]
        self.job_data.update(changes)

    def connect(self):
        return _FakeConnection(self)


def _install_worker_modules(family: str):
    """Install the real patched worker with restricted dependency stubs."""
    package = FAMILIES[family]
    worker_path = VENDOR / package / "worker.py"
    if not worker_path.is_file():
        raise AssertionError(f"patched worker copy is missing: {worker_path}")

    saved = {name: sys.modules.get(name) for name in (
        package, package + ".assets", package + ".core", package + ".engine",
        "workbench_server", "workbench_server.gpu_lock",
    )}

    pkg = types.ModuleType(package)
    pkg.__path__ = [str(worker_path.parent)]
    pkg.__package__ = package
    sys.modules[package] = pkg

    assets = types.ModuleType(package + ".assets")
    assets.model_fingerprint = _model_fingerprint
    sys.modules[assets.__name__] = assets

    core = types.ModuleType(package + ".core")
    core.digest = _sha
    core.identifier = lambda prefix: prefix + "_fake_0001"
    core.now = lambda: "2026-10-30T00:00:00+00:00"
    core.settings = lambda: {}
    core.write_json = _write_json
    core.Store = _FakeStore
    sys.modules[core.__name__] = core

    engine = types.ModuleType(package + ".engine")
    engine.LocateEngine = _FakeEngine
    engine.Sam3Engine = _FakeEngine
    sys.modules[engine.__name__] = engine

    wb = types.ModuleType("workbench_server")
    wb.__path__ = []
    sys.modules["workbench_server"] = wb
    gpu_lock = types.ModuleType("workbench_server.gpu_lock")
    gpu_lock.leased = lambda function: function
    sys.modules["workbench_server.gpu_lock"] = gpu_lock

    spec = importlib.util.spec_from_file_location(package + ".worker", worker_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, saved


def _restore_modules(saved):
    for name, value in saved.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


def _make_case(tmp_path: Path, family: str, classes: dict[str, str],
               parent_classes: dict[str, str], scenario: str):
    job_id = f"job-e2e-{family}-{scenario}"
    root = tmp_path / job_id
    job_dir = root / "jobs" / job_id
    job_dir.mkdir(parents=True)
    parent_dir = root / "parent"
    parent_dir.mkdir()
    (parent_dir / "parent.bin").write_bytes(b"parent")
    image = root / "image.jpg"
    label = root / "label.txt"
    image.write_bytes(b"image")
    label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    rows = [{
        "class_name": next(iter(classes)),
        "image": str(image),
        "label": str(label),
        "image_sha256": _sha(image),
        "label_sha256": _sha(label),
    }]
    parent = {"id": REMOTE_ID, "path": str(parent_dir), "fingerprint": PARENT_FP}
    request = {
        "name": f"e2e-{family}-{scenario}",
        "seed": 20261030,
        "training_mode": "new_only",
    }
    request["image_tokens" if family == "locate" else "resolution"] = 1024
    manifest = {
        "family": family,
        "parent_model_id": REMOTE_ID,
        "parent_fingerprint": PARENT_FP,
        "training_mode": "new_only",
        "old_knowledge_status": "not_evaluated",
        "evaluated_old_classes": [],
        "new_classes": list(classes),
        "old_classes": list(parent_classes),
        "classes": classes,
        "warnings": [],
        "next_replay_assets": [],
        "request": request,
        "sets": {
            "new_train": rows,
            "old_train": [],
            "eval": rows,
        },
    }
    _write_json(job_dir / "manifest.json", manifest)
    job = {"id": job_id, "state": "RUNNING", "cancel_requested": False}
    return root, job_dir, parent, job, classes


@pytest.mark.parametrize("family", ["locate", "sam3"])
@pytest.mark.parametrize("scenario,parent_classes,classes", [
    ("empty_parent_new_class", {}, {"pothole": "pothole"}),
    ("existing_parent_new_class", {"fish": "fish"}, {"fish": "fish", "pothole": "pothole"}),
    ("same_class", {"fish": "fish"}, {"fish": "fish"}),
])
def test_real_worker_zip_binds_parent_and_validates(
    tmp_path: Path, monkeypatch, family: str, scenario: str,
    parent_classes: dict[str, str], classes: dict[str, str],
) -> None:
    root, job_dir, parent, job, expected_classes = _make_case(
        tmp_path, family, classes, parent_classes, scenario)
    store = _FakeStore(root, parent, job)
    module, saved = _install_worker_modules(family)
    try:
        # The real worker insists on 24 GiB free; this is a host-space policy,
        # not a packaging behavior, so the harness supplies a restricted value.
        monkeypatch.setattr(module.shutil, "disk_usage", lambda _: _DiskUsage(100 * 1024**3))
        module.run_job(store, job, {"base": str(root / "base")})
    finally:
        _restore_modules(saved)

    archive = job_dir / "weights.zip"
    assert archive.is_file(), (job_dir / "failure.log").read_text(encoding="utf-8")
    with __import__("zipfile").ZipFile(archive) as value:
        manifest = json.loads(value.read("result_manifest.json"))
        checksums = json.loads(value.read("file_checksums.json"))
        assert "result_manifest.json" in checksums
        assert checksums["result_manifest.json"] == _sha(
            job_dir / "result_manifest.json")
    assert manifest["parent_model_fingerprint"] == PARENT_FP
    assert manifest["parent_model_version_id"] == REMOTE_ID
    assert any(
        value["parent_model_fingerprint"] == PARENT_FP
        for _, _, value in store.inserted if isinstance(value, dict)
    ), "external version metadata is missing parent_model_fingerprint"

    run = _load_run_module()
    result = run.weights_validate.validate_weights_zip(
        archive,
        family=family,
        expected_parent_fingerprint=PARENT_FP,
        expected_parent_model_id=REMOTE_ID,
        expected_parent_model_ids={LOCAL_ID: REMOTE_ID},
        expected_classes=expected_classes,
    )
    assert result["ok"] is True
    assert result["parent_fingerprint_in_manifest"] is True


@pytest.mark.parametrize("family", ["locate", "sam3"])
def test_real_worker_zip_rejects_wrong_parent_fingerprint(
    tmp_path: Path, monkeypatch, family: str
) -> None:
    classes = {"pothole": "pothole"}
    root, job_dir, parent, job, _ = _make_case(
        tmp_path, family, classes, {}, "wrong_parent")
    store = _FakeStore(root, parent, job)
    module, saved = _install_worker_modules(family)
    try:
        monkeypatch.setattr(module.shutil, "disk_usage", lambda _: _DiskUsage(100 * 1024**3))
        module.run_job(store, job, {"base": str(root / "base")})
    finally:
        _restore_modules(saved)

    archive = job_dir / "weights.zip"
    assert archive.is_file()
    run = _load_run_module()
    with pytest.raises(run.weights_validate.ValidationError) as caught:
        run.weights_validate.validate_weights_zip(
            archive,
            family=family,
            expected_parent_fingerprint=WRONG_FP,
            expected_parent_model_id=REMOTE_ID,
            expected_classes=classes,
        )
    assert "parent_model_fingerprint mismatch" in str(caught.value)


def test_patched_worker_source_is_the_only_new_field_change() -> None:
    """The vendor copy must differ from upstream only by the binding fields."""
    if not (UPSTREAM / "training_service" / "worker.py").is_file():
        pytest.skip("upstream project checkout is not available")
    for package in FAMILIES.values():
        source = (VENDOR / package / "worker.py").read_text(encoding="utf-8")
        original = (UPSTREAM / package / "worker.py").read_text(encoding="utf-8")
        assert source.count("parent_model_fingerprint") == 2
        assert "parent_model_fingerprint" not in original
        assert source.replace(
            "             'parent_model_fingerprint':parent['fingerprint'],\n"
            "             'training_mode'",
            "             'training_mode'",
        ).replace(
            "            'parent_model_fingerprint':parent['fingerprint'],\n"
            "            'training_mode'",
            "            'training_mode'",
        ) == original
