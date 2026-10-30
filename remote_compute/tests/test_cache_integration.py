"""WeightCache wiring inside ServerApp: cache fill bypasses upload, state records it."""

from __future__ import annotations

import hashlib
from pathlib import Path

from common.config import ServerConfig
from common.models import EnvSpec, FileEntry, JobSpec, Resources
from server.app import ServerApp


def _app(tmp_path: Path, quota: int = 2 * 1024 ** 3) -> ServerApp:
    config = ServerConfig(data_dir=str(tmp_path / "data"), weight_cache_bytes=quota)
    return ServerApp(config, gpus=[])


def _spec(path: str, payload: bytes) -> JobSpec:
    return JobSpec(
        job_name="cache-test",
        entry_argv=["python", "train.py"],
        inputs=[FileEntry(path=path, size=len(payload), sha256=hashlib.sha256(payload).hexdigest())],
        env=EnvSpec(),
        resources=Resources(timeout_s=30),
        outputs=["outputs/**"],
    )


def test_cache_materializes_missing_input_and_records_upload(tmp_path: Path) -> None:
    payload = b"model-bytes" * 1000
    spec = _spec("weights/model.bin", payload)
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    app = _app(tmp_path)
    runtime = app.create_job(spec)
    present, missing = app.store.present_missing(runtime.job_id, spec)
    assert present == [] and len(missing) == 1

    assert app.weight_cache is not None
    assert app.weight_cache.put(source, spec.inputs[0].sha256, len(payload)) is True
    present, missing, hits = app.cache_missing_inputs(runtime.job_id, spec, present, missing)
    assert present == ["weights/model.bin"]
    assert missing == []
    assert hits == ["weights/model.bin"]
    target = app.store.input_path(runtime.job_id, "weights/model.bin")
    assert target.read_bytes() == payload
    state = app.store.load_state(runtime.job_id)
    assert state["uploads"]["weights/model.bin"]["done"] is True


def test_cache_hit_is_idempotent_and_keeps_cached_object(tmp_path: Path) -> None:
    payload = b"same-bytes" * 100
    spec = _spec("data.bin", payload)
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    app = _app(tmp_path)
    app.weight_cache.put(source, spec.inputs[0].sha256, len(payload))
    runtime = app.create_job(spec)
    present, missing = [], [spec.inputs[0]]
    present, missing, hits = app.cache_missing_inputs(runtime.job_id, spec, present, missing)
    assert hits == ["data.bin"]
    present, missing = app.store.present_missing(runtime.job_id, spec)
    assert present == ["data.bin"] and missing == []
    present, missing, hits = app.cache_missing_inputs(runtime.job_id, spec, present, missing)
    assert hits == []  # already present; no second materialization


def test_disabled_cache_is_transparent(tmp_path: Path) -> None:
    payload = b"x"
    spec = _spec("a.bin", payload)
    config = ServerConfig(data_dir=str(tmp_path / "data"), weight_cache_bytes=0)
    app = ServerApp(config, gpus=[])
    runtime = app.create_job(spec)
    present, missing = app.store.present_missing(runtime.job_id, spec)
    updated_present, updated_missing, hits = app.cache_missing_inputs(
        runtime.job_id, spec, present, missing
    )
    assert updated_present == [] and updated_missing == missing and hits == []
