"""Run two incremental CUDA jobs through RGPU over an isolated SSH tunnel."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "remote_compute"))

from client.runner import Callbacks, SubmitOptions, submit_project
from common.config import ClientConfig
from common.hashing import sha256_file


INSTALL = r'''
import base64, hashlib, io, json, pathlib, re, sys, tarfile
request = json.load(sys.stdin)
if not re.fullmatch(r"[a-zA-Z0-9_-]+", request["run_id"]):
    raise ValueError("Invalid run identifier")
payload = base64.b64decode(request["archive"], validate=True)
if hashlib.sha256(payload).hexdigest() != request["sha256"]:
    raise ValueError("Deployment checksum mismatch")
root = pathlib.Path.home() / ".rgpu-roundtrip" / request["run_id"]
root.mkdir(parents=True, exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
    for member in archive.getmembers():
        if not (member.isfile() or member.isdir()):
            raise ValueError("Unexpected archive member")
        if not (root / member.name).resolve().is_relative_to(root.resolve()):
            raise ValueError("Invalid deployment path")
    archive.extractall(root)
print(json.dumps({"root": str(root), "sha256": request["sha256"]}))
'''


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ssh(args, command, request=None, timeout=30):
    process = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
         "-o", "ClearAllForwardings=yes", args.ssh_host, shlex.join(command)],
        input=json.dumps(request).encode() if request is not None else None,
        capture_output=True, timeout=timeout,
    )
    if process.returncode:
        raise RuntimeError("SSH operation failed\n" + process.stderr.decode("utf-8", "replace")[-4000:])
    return json.loads(process.stdout.decode("utf-8"))


def deploy(args, run_id):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for package in ("common", "server"):
            source = ROOT / "remote_compute" / package
            for path in sorted(source.rglob("*.py")):
                archive.add(path, arcname=path.relative_to(ROOT).as_posix())
        archive.add(ROOT / "tools" / "remote_gpu_session.py", arcname="remote_gpu_session.py")
    payload = buffer.getvalue()
    request = {"run_id": run_id, "archive": base64.b64encode(payload).decode(),
               "sha256": hashlib.sha256(payload).hexdigest()}
    return ssh(args, [args.remote_python, "-B", "-c", INSTALL], request, timeout=60)


def remote(args, root, command, request=None):
    return ssh(args, [args.remote_python, "-B", root + "/remote_gpu_session.py", command], request)


def open_tunnel(args, remote_port, log_path):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with log_path.open("wb") as stderr:
        process = subprocess.Popen(
            ["ssh", "-N", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
             "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
             "-o", "ServerAliveCountMax=3", "-L",
             f"127.0.0.1:{port}:127.0.0.1:{remote_port}", args.ssh_host],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    try:
        for _ in range(150):
            if process.poll() is not None:
                raise RuntimeError(log_path.read_text(errors="replace"))
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return process, port
            except OSError:
                time.sleep(0.1)
        raise TimeoutError("SSH tunnel startup timed out")
    except BaseException:
        process.terminate()
        process.wait(timeout=10)
        raise


def verify_result(result, expected_parent, expected_dataset, expected_runs, expected_epochs):
    if not result.ok or result.state != "succeeded":
        raise ValueError(f"RGPU job failed: {result.to_dict()}")
    artifacts = {}
    for item in result.downloaded:
        path = Path(item["local_path"])
        if (not item.get("verified") or path.stat().st_size != item["size"]
                or sha256_file(path) != item["sha256"]):
            raise ValueError("Downloaded artifact checksum verification failed")
        artifacts[Path(item["path"]).name] = path
    if set(artifacts) != {"model.json", "metrics.json"}:
        raise ValueError("Expected model.json and metrics.json in job outputs")
    metrics = json.loads(artifacts["metrics.json"].read_text(encoding="utf-8"))
    model = json.loads(artifacts["model.json"].read_text(encoding="utf-8"))
    if any(not str(metrics[key]).startswith("cuda:") for key in ("device", "tensor_device", "parameter_device")):
        raise ValueError("Training tensors and parameters must be on CUDA")
    if metrics["peak_memory_allocated_bytes"] <= 0 or metrics["cuda_elapsed_ms"] <= 0:
        raise ValueError("CUDA allocation and timing evidence are required")
    if metrics["parent_checkpoint_sha256"] != expected_parent or metrics["dataset_sha256"] != expected_dataset:
        raise ValueError("Task inputs do not match the submitted parent and dataset")
    if model["runs"] != expected_runs or model["epochs_total"] != expected_epochs:
        raise ValueError("Incremental checkpoint counters do not match")
    if metrics["runs"] != model["runs"] or metrics["epochs_total"] != model["epochs_total"]:
        raise ValueError("Checkpoint and metrics counters disagree")
    if (not all(math.isfinite(float(metrics[key])) for key in ("start_loss", "final_loss"))
            or not 0 <= metrics["final_loss"] < metrics["start_loss"]):
        raise ValueError("The CUDA training loss did not improve")
    return artifacts, metrics


def gpu_process_observed(metrics, samples):
    physical = str(metrics["cuda_visible_devices"])
    for sample in samples:
        gpu_uuids = {line.split(",", 2)[1].strip() for line in sample.get("gpus", [])
                     if line.split(",", 1)[0].strip() == physical}
        for line in sample.get("processes", []):
            values = [value.strip() for value in line.split(",")]
            if (len(values) == 4 and values[0] in gpu_uuids and values[1] == str(metrics["pid"])
                    and values[3].isdigit() and int(values[3]) > 0):
                return True
    return False


def run(args):
    run_id = time.strftime("gpu-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    directory = args.work_dir.resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    report_path = directory / "report.json"
    report = {"run_id": run_id, "status": "running", "rounds": [], "checks": {}}
    write_json(report_path, report)
    remote_root, tunnel = None, None
    stop_monitor = threading.Event()
    monitor = None
    samples = []
    errors = []

    def collect_samples():
        while not stop_monitor.is_set():
            try:
                samples.append(remote(args, remote_root, "snapshot"))
            except Exception as error:
                errors.append(str(error))
            stop_monitor.wait(0.5)

    try:
        print("[roundtrip] deploying dedicated RGPU server", flush=True)
        deployment = deploy(args, run_id)
        remote_root = deployment["root"]
        report["deployment"] = deployment
        report["gpu_before"] = remote(args, remote_root, "snapshot")
        token = secrets.token_urlsafe(32)
        session = remote(args, remote_root, "start", {"token": token})
        report["session"] = session
        tunnel, local_port = open_tunnel(args, session["port"], directory / "tunnel.stderr.log")
        config = ClientConfig(host="127.0.0.1", port=local_port, token=token, io_timeout=args.timeout)
        monitor = threading.Thread(target=collect_samples, daemon=True)
        monitor.start()
        project = directory / "project"
        project.mkdir()
        shutil.copy2(ROOT / "remote_compute" / "examples" / "gpu_train" / "train.py", project / "train.py")
        for filename in ("model.json", "data.csv"):
            shutil.copy2(ROOT / "remote_compute" / "examples" / "incremental_train" / filename, project / filename)
        receipts = directory / "receipts"
        receipts.mkdir()
        expected_dataset = sha256_file(project / "data.csv")
        base = json.loads((project / "model.json").read_text(encoding="utf-8"))
        previous_metrics = None
        callbacks = Callbacks(log=lambda value: print("[rgpu] " + value, flush=True),
                              event=lambda stream, value: print(value, end="", flush=True))
        for round_number in (1, 2):
            print(f"[roundtrip] submitting CUDA round {round_number}", flush=True)
            expected_parent = sha256_file(project / "model.json")
            options = SubmitOptions(
                project_dir=str(project),
                entry=shlex.join(["python", "train.py", "--runtime-python", args.remote_python,
                                  "--steps", str(args.steps), "--step-delay", "0.02"]),
                job_name=f"cuda-roundtrip-{round_number}", gpu=True, timeout_s=args.timeout,
                download_dir=str(directory / f"round-{round_number}"),
                python_version=session["python_version"], incremental=False,
                receipt_dir=str(receipts), receipt_request_id=f"{run_id}-{round_number}",
            )
            result = submit_project(config, options, callbacks=callbacks)
            report["rounds"].append({"result": result.to_dict()})
            write_json(report_path, report)
            artifacts, metrics = verify_result(result, expected_parent, expected_dataset,
                                               base["runs"] + round_number,
                                               base["epochs_total"] + args.steps * round_number)
            if previous_metrics and not math.isclose(previous_metrics["final_loss"], metrics["start_loss"], rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError("Round two did not continue from round one's final loss")
            if round_number == 2 and result.env.get("state") != "cached":
                raise ValueError("The second job did not reuse the prepared environment")
            report["rounds"][-1]["metrics"] = metrics
            previous_metrics = metrics
            shutil.copy2(artifacts["model.json"], project / "model.json")
            write_json(report_path, report)
        stop_monitor.set()
        monitor.join(timeout=40)
        if monitor.is_alive():
            raise RuntimeError("GPU observation thread did not stop")
        for item in report["rounds"]:
            if not gpu_process_observed(item["metrics"], samples):
                raise ValueError("nvidia-smi did not observe the CUDA training PID")
        report["checks"] = {"two_cuda_jobs": True, "download_hashes": True,
                            "checkpoint_continuity": True, "loss_improved": True,
                            "environment_reused": True, "gpu_pids_observed": True}
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        stop_monitor.set()
        if monitor is not None:
            monitor.join(timeout=40)
        report["gpu_samples"] = samples
        report["gpu_sample_errors"] = errors
        cleanup_errors = []
        if remote_root is not None:
            for command, key in [("stop", "server_cleanup"), ("logs", "server_logs"), ("snapshot", "gpu_after")]:
                try:
                    report[key] = remote(args, remote_root, command)
                except Exception as error:
                    cleanup_errors.append(f"{command}: {error}")
        if tunnel is not None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=10)
                report["tunnel_stopped"] = True
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait(timeout=10)
                report["tunnel_stopped"] = True
        report["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            report["status"] = "failed"
        write_json(report_path, report)
        print(f"[roundtrip] {report['status']} report={report_path}", flush=True)
    if report["status"] != "passed":
        raise RuntimeError("Round trip cleanup failed")
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", required=True, help="SSH alias with key-based access")
    parser.add_argument("--remote-python", required=True, help="Absolute remote Python path with CUDA PyTorch")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "_rgpu_data" / "gpu_roundtrip")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)
    if args.ssh_host.startswith("-") or not args.remote_python.startswith("/") or args.steps < 1 or args.timeout < 1:
        parser.error("Supply a valid SSH host, absolute remote Python, and positive steps/timeout")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
