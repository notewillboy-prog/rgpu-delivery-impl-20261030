"""Train a linear model on CUDA and return a verifiable JSON checkpoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", help="Existing remote Python with CUDA PyTorch")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--step-delay", type=float, default=0)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.step_delay < 0:
        parser.error("steps must be positive and step-delay must be nonnegative")
    if args.runtime_python:
        runtime = str(Path(args.runtime_python).resolve(strict=True))
        os.execv(runtime, [runtime, "-B", str(Path(__file__).resolve()),
                          "--steps", str(args.steps), "--step-delay", str(args.step_delay)])

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this task")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("The RGPU scheduler must assign a CUDA device")
    device = torch.device("cuda:0")
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint_path = Path("model.json")
    checkpoint_bytes = checkpoint_path.read_bytes()
    parent = json.loads(checkpoint_bytes)
    dataset_path = Path("data.csv")
    with dataset_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    x = torch.tensor([[float(row["x"])] for row in rows], device=device)
    y = torch.tensor([[float(row["y"])] for row in rows], device=device)
    model = torch.nn.Linear(1, 1, device=device)
    with torch.no_grad():
        model.weight.fill_(float(parent["w"]))
        model.bias.fill_(float(parent["b"]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_function = torch.nn.MSELoss()
    start_loss = loss_function(model(x), y).item()
    torch.cuda.reset_peak_memory_stats(device)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(x), y)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 32 == 0 or step + 1 == args.steps:
            print(f"[cuda-train] step={step + 1}/{args.steps} loss={loss.item():.8f}", flush=True)
        if args.step_delay:
            time.sleep(args.step_delay)
    end.record()
    torch.cuda.synchronize(device)
    final_loss = loss_function(model(x), y).item()
    checkpoint = {
        "w": model.weight.detach().cpu().item(),
        "b": model.bias.detach().cpu().item(),
        "runs": int(parent.get("runs", 0)) + 1,
        "epochs_total": int(parent.get("epochs_total", 0)) + args.steps,
    }
    metrics = {
        "device": str(device),
        "tensor_device": str(x.device),
        "parameter_device": str(model.weight.device),
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "pid": os.getpid(),
        "samples": len(rows),
        "steps": args.steps,
        "start_loss": start_loss,
        "final_loss": final_loss,
        "runs": checkpoint["runs"],
        "epochs_total": checkpoint["epochs_total"],
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_elapsed_ms": begin.elapsed_time(end),
        "step_delay_seconds": args.step_delay,
        "parent_checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    }
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    for filename, data in [("model.json", checkpoint), ("metrics.json", metrics)]:
        temporary = output / (filename + ".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / filename)
    print(f"[cuda-train] device={device} loss={start_loss:.8f}->{final_loss:.8f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
