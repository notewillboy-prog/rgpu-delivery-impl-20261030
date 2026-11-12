"""Manage the dedicated RGPU process created by gpu_roundtrip.py."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "session.json"


def snapshot():
    queries = {
        "gpus": "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "processes": "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    }
    result = {"time": time.time(), "hostname": socket.gethostname()}
    for key, query in queries.items():
        process = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10, check=True)
        result[key] = process.stdout.splitlines()
    return result


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        if stat[0] == "Z":
            return None
        return stat[19]
    except FileNotFoundError:
        return None


def stop():
    if not STATE.exists():
        return {"stopped": True, "reason": "session not started"}
    state = json.loads(STATE.read_text())
    pid = state["pid"]
    if process_identity(pid) is None:
        return {"stopped": True, "pid": pid}
    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if (process_identity(pid) != state["identity"] or b"server.main" not in command
            or str(ROOT / "server_data").encode() not in command):
        raise RuntimeError("Server process identity changed")
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if process_identity(pid) is None:
            return {"stopped": True, "pid": pid}
        time.sleep(0.1)
    if os.getpgid(pid) != pid or process_identity(pid) != state["identity"]:
        raise RuntimeError("Server process group identity changed")
    os.killpg(pid, signal.SIGKILL)
    for _ in range(30):
        if process_identity(pid) is None:
            return {"stopped": True, "pid": pid, "forced": True}
        time.sleep(0.1)
    raise RuntimeError("Dedicated server did not stop")


def start():
    request = json.load(sys.stdin)
    if STATE.exists():
        raise RuntimeError("Session already exists")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("The selected remote Python has no CUDA PyTorch")
    if shutil.disk_usage(ROOT).free < 128 * 1024 * 1024:
        raise RuntimeError("Insufficient free disk space for the round trip")
    env = dict(os.environ, RGPU_TOKEN=request["token"], PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env["PYTHONPATH"] = str(ROOT / "remote_compute")
    for key, directory in [("TMPDIR", "tmp"), ("PIP_CACHE_DIR", "pip-cache"), ("XDG_CACHE_HOME", "cache")]:
        (ROOT / directory).mkdir(exist_ok=True)
        env[key] = str(ROOT / directory)
    command = [sys.executable, "-B", "-m", "server.main", "serve", "--host", "127.0.0.1",
               "--port", "0", "--data-dir", str(ROOT / "server_data"),
               "--strict-gpu", "--strict-recovery", "--max-parallel", "1"]
    with (ROOT / "server.stdout.log").open("w") as stdout, (ROOT / "server.stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, cwd=ROOT / "remote_compute", env=env,
                                   stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
    state = {"pid": process.pid, "identity": process_identity(process.pid)}
    STATE.write_text(json.dumps(state), encoding="utf-8")
    try:
        for _ in range(200):
            ready = (ROOT / "server.stdout.log").read_text()
            match = re.search(r"RGPU_SERVER_READY host=127\.0\.0\.1 port=(\d+)", ready)
            if match:
                return {**state, "port": int(match.group(1)), "root": str(ROOT),
                        "python": sys.executable, "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                        "torch_version": torch.__version__, "cuda_version": torch.version.cuda}
            if process.poll() is not None:
                raise RuntimeError((ROOT / "server.stderr.log").read_text()[-4000:])
            time.sleep(0.1)
        raise TimeoutError("Dedicated server startup timed out")
    except BaseException:
        stop()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start", "stop", "snapshot", "logs"])
    args = parser.parse_args()
    if args.command == "logs":
        result = {path.name: path.read_text(errors="replace")[-30000:]
                  for path in ROOT.glob("server.*.log")}
    else:
        result = {"start": start, "stop": stop, "snapshot": snapshot}[args.command]()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
