"""Regression checks for CUDA evidence, artifact validation, and SSH cleanup."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gpu_roundtrip", ROOT / "tools" / "gpu_roundtrip.py")
roundtrip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roundtrip)


def make_result(directory, *, parent="parent", runs=1, epochs=128, cached=False):
    directory.mkdir(parents=True, exist_ok=True)
    metrics = {
        "device": "cuda:0", "tensor_device": "cuda:0", "parameter_device": "cuda:0",
        "peak_memory_allocated_bytes": 4096, "cuda_elapsed_ms": 100,
        "parent_checkpoint_sha256": parent, "dataset_sha256": "dataset",
        "runs": runs, "epochs_total": epochs, "start_loss": 1 / runs,
        "final_loss": 1 / (runs + 1), "pid": 100 + runs, "cuda_visible_devices": "0",
    }
    model = {"w": 1.5, "b": 0.8, "runs": runs, "epochs_total": epochs}
    outputs = []
    for filename, data in [("metrics.json", metrics), ("model.json", model)]:
        path = directory / filename
        path.write_text(json.dumps(data), encoding="utf-8")
        outputs.append({"path": "outputs/" + filename, "local_path": str(path),
                        "size": path.stat().st_size, "sha256": roundtrip.sha256_file(path), "verified": True})
    result = SimpleNamespace(ok=True, state="succeeded", downloaded=outputs,
                             env={"state": "cached" if cached else "built"})
    result.to_dict = lambda: {"ok": True, "outputs": outputs, "env": result.env}
    return result


def update_metrics(result, **changes):
    output = result.downloaded[0]
    path = Path(output["local_path"])
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")
    output.update(size=path.stat().st_size, sha256=roundtrip.sha256_file(path))


def test_valid_cuda_result(tmp_path):
    result = make_result(tmp_path)
    artifacts, metrics = roundtrip.verify_result(result, "parent", "dataset", 1, 128)
    assert set(artifacts) == {"model.json", "metrics.json"}
    assert metrics["final_loss"] < metrics["start_loss"]


@pytest.mark.parametrize("changes", [
    {"device": "cpu"}, {"tensor_device": "cpu"}, {"parameter_device": "cpu"},
    {"peak_memory_allocated_bytes": 0}, {"cuda_elapsed_ms": 0},
    {"parent_checkpoint_sha256": "another-parent"}, {"dataset_sha256": "another-data"},
    {"runs": 7}, {"final_loss": float("nan")}, {"final_loss": 2},
])
def test_rejects_invalid_cuda_evidence(tmp_path, changes):
    result = make_result(tmp_path)
    update_metrics(result, **changes)
    with pytest.raises(ValueError):
        roundtrip.verify_result(result, "parent", "dataset", 1, 128)


def test_rejects_changed_download(tmp_path):
    result = make_result(tmp_path)
    Path(result.downloaded[1]["local_path"]).write_text("corrupted")
    with pytest.raises(ValueError, match="checksum"):
        roundtrip.verify_result(result, "parent", "dataset", 1, 128)


def test_gpu_observation_matches_device_pid_and_allocated_memory():
    metrics = {"pid": 101, "cuda_visible_devices": "1"}
    sample = {"gpus": ["0, GPU-first, GPU name, 24000, 0, 0", "1, GPU-second, GPU name, 24000, 100, 5"],
              "processes": ["GPU-second, 101, python, 100"]}
    assert roundtrip.gpu_process_observed(metrics, [sample])
    assert not roundtrip.gpu_process_observed({**metrics, "pid": 102}, [sample])
    assert not roundtrip.gpu_process_observed({**metrics, "cuda_visible_devices": "0"}, [sample])
    sample["processes"] = ["GPU-second, 101, python, 0"]
    assert not roundtrip.gpu_process_observed(metrics, [sample])


@pytest.mark.parametrize("fail_submit", [False, True])
def test_runner_api_order_and_cleanup(tmp_path, monkeypatch, fail_submit):
    calls = []
    tunnel = SimpleNamespace(terminate=lambda: calls.append("tunnel-stop"), wait=lambda timeout: 0)
    monkeypatch.setattr(roundtrip, "deploy", lambda args, run_id: {"root": "/isolated/session"})
    monkeypatch.setattr(roundtrip, "open_tunnel", lambda *args: (tunnel, 12345))

    def fake_remote(args, root, command, request=None):
        calls.append(command)
        if command == "start":
            return {"port": 45678, "python_version": "3.10"}
        if command == "snapshot":
            return {"gpus": ["0, GPU-test, GPU name, 24000, 100, 5"],
                    "processes": ["GPU-test, 101, python, 100", "GPU-test, 102, python, 100"]}
        return {"stopped": True} if command == "stop" else {}

    def fake_submit(config, options, callbacks):
        assert isinstance(config, roundtrip.ClientConfig)
        assert isinstance(options, roundtrip.SubmitOptions)
        assert options.gpu
        if fail_submit:
            raise RuntimeError("submission failed")
        number = 1 if options.job_name.endswith("-1") else 2
        parent = roundtrip.sha256_file(Path(options.project_dir) / "model.json")
        result = make_result(Path(options.download_dir), parent=parent, runs=number,
                             epochs=number * 128, cached=number == 2)
        update_metrics(result, dataset_sha256=roundtrip.sha256_file(Path(options.project_dir) / "data.csv"))
        return result

    monkeypatch.setattr(roundtrip, "remote", fake_remote)
    monkeypatch.setattr(roundtrip, "submit_project", fake_submit)
    args = SimpleNamespace(work_dir=tmp_path, remote_python="/venv/bin/python", steps=128, timeout=30)
    if fail_submit:
        with pytest.raises(RuntimeError, match="submission failed"):
            roundtrip.run(args)
    else:
        roundtrip.run(args)
    report = json.loads(next(tmp_path.glob("*/report.json")).read_text(encoding="utf-8"))
    assert report["server_cleanup"]["stopped"]
    assert report["tunnel_stopped"]
    assert "stop" in calls and "tunnel-stop" in calls
    assert report["status"] == ("failed" if fail_submit else "passed")
