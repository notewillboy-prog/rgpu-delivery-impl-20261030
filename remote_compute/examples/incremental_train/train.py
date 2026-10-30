#!/usr/bin/env python3
"""增量训练示例（纯 Python 标准库，不依赖任何第三方包）。

用途：作为 RGPU 客户端 / 服务端端到端验收的「真实训练任务」。
它刻意保持极简：单特征线性回归，全批量梯度下降，但具备增量训练的全部语义：

1. 若 ``--model`` 指向的模型文件存在，则**加载旧权重继续训练**（增量）；
2. 每次运行把更新后的权重写回 ``--model``（便于本地连续两次运行验证增量），
   同时按约定把产物写到 ``--out/model.json`` 与 ``--out/metrics.json``；
3. ``metrics.json`` 记录 ``start_loss`` / ``final_loss`` / ``improvement`` /
   ``epochs_total`` / ``runs``，能直接证明「第二次训练基于第一次权重、loss 继续下降」。

训练日志逐 epoch 打印并 flush，作为远端流式日志（``job_event``）的验证据点。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

DEFAULT_DATA = "data.csv"
DEFAULT_MODEL = "model.json"
DEFAULT_OUT = "outputs"
DEFAULT_EPOCHS = 3
DEFAULT_LR = 0.05


def log(message: str) -> None:
    """逐行 flush：服务端才能实时把日志流给客户端。"""
    print(message, flush=True)


def load_dataset(path: str) -> Tuple[List[float], List[float]]:
    """读取两列 CSV（x,y），忽略空行；首行若无法解析为数字则视为表头。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据集不存在: {path}")
    xs: List[float] = []
    ys: List[float] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.replace(";", ",").split(",")]
            if len(parts) < 2:
                skipped += 1
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                if lineno == 1:  # 表头
                    continue
                skipped += 1
                continue
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError(f"数据集为空或格式错误（需要两列 x,y）: {path}")
    if skipped:
        log(f"[train] 警告: 跳过 {skipped} 行无法解析的数据")
    return xs, ys


def load_model(path: str) -> Tuple[Dict[str, float], bool]:
    """读取基座模型；不存在则返回初始权重（w=0,b=0）。"""
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("模型文件必须是 JSON 对象")
        except (OSError, ValueError) as exc:
            raise ValueError(f"基座模型无法读取 {path}: {exc}") from exc
        model = {
            "w": float(data.get("w", 0.0)),
            "b": float(data.get("b", 0.0)),
            "runs": int(data.get("runs", 0)),
            "epochs_total": int(data.get("epochs_total", 0)),
        }
        return model, True
    return {"w": 0.0, "b": 0.0, "runs": 0, "epochs_total": 0}, False


def mse_loss(xs: List[float], ys: List[float], w: float, b: float) -> float:
    total = 0.0
    for x, y in zip(xs, ys):
        diff = w * x + b - y
        total += diff * diff
    return total / len(xs)


def train_epoch(
    xs: List[float], ys: List[float], w: float, b: float, lr: float
) -> Tuple[float, float]:
    """一个全批量梯度下降 epoch（MSE 的解析梯度），返回 (w, b)。"""
    n = len(xs)
    grad_w = 0.0
    grad_b = 0.0
    for x, y in zip(xs, ys):
        diff = w * x + b - y
        grad_w += diff * x
        grad_b += diff
    scale = 2.0 / n
    return w - lr * scale * grad_w, b - lr * scale * grad_b


def write_json(path: str, payload: Dict[str, object]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="纯标准库增量训练示例（线性回归 + 基座模型继续训练）"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"本次训练轮数（默认 {DEFAULT_EPOCHS}）")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"学习率（默认 {DEFAULT_LR}）")
    parser.add_argument("--data", default=DEFAULT_DATA, help=f"数据集 CSV（默认 {DEFAULT_DATA}）")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"基座模型 JSON（存在则继续训练，默认 {DEFAULT_MODEL}）")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"产物目录（默认 {DEFAULT_OUT}）")
    parser.add_argument("--simulate-seconds", type=float, default=0.0,
                        help="模拟耗时（秒，按 epoch 均摊 sleep），用于演示流式日志")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs <= 0:
        log("[train] 错误: --epochs 必须大于 0")
        return 2
    if args.lr <= 0:
        log("[train] 错误: --lr 必须大于 0")
        return 2
    if args.simulate_seconds < 0:
        log("[train] 错误: --simulate-seconds 不能为负")
        return 2

    log("[train] ===== 增量训练 demo（纯标准库）=====")
    log(
        f"[train] 参数: epochs={args.epochs} lr={args.lr} "
        f"data={args.data} model={args.model} out={args.out}"
        + (f" simulate_seconds={args.simulate_seconds}" if args.simulate_seconds else "")
    )

    try:
        xs, ys = load_dataset(args.data)
    except (OSError, ValueError) as exc:
        log(f"[train] 错误: {exc}")
        return 1
    log(f"[train] 数据集: {args.data}  rows={len(xs)}  "
        f"x∈[{min(xs):.3f}, {max(xs):.3f}]  y∈[{min(ys):.3f}, {max(ys):.3f}]")

    try:
        model, resumed = load_model(args.model)
    except ValueError as exc:
        log(f"[train] 错误: {exc}")
        return 1
    w, b = model["w"], model["b"]
    if resumed:
        log(
            f"[train] 加载基座模型 {args.model}: runs={model['runs']} "
            f"epochs_total={model['epochs_total']} w={w:.4f} b={b:.4f} → 继续增量训练"
        )
    else:
        log(f"[train] 未找到基座模型 {args.model}，初始化为 w={w:.4f} b={b:.4f}")

    start_loss = mse_loss(xs, ys, w, b)
    log(f"[train] 起始 loss={start_loss:.6f}")

    history: List[float] = []
    sleep_per_epoch = (args.simulate_seconds / args.epochs) if args.simulate_seconds else 0.0
    for epoch in range(1, args.epochs + 1):
        if sleep_per_epoch > 0:
            time.sleep(sleep_per_epoch)
        w, b = train_epoch(xs, ys, w, b, args.lr)
        current_loss = mse_loss(xs, ys, w, b)
        history.append(current_loss)
        log(
            f"[train] epoch {epoch}/{args.epochs} "
            f"loss={current_loss:.6f} w={w:.6f} b={b:.6f}"
        )

    final_loss = history[-1]
    improvement = start_loss - final_loss
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    runs = model["runs"] + 1
    epochs_total = model["epochs_total"] + args.epochs

    model_payload: Dict[str, object] = {
        "w": round(w, 8),
        "b": round(b, 8),
        "runs": runs,
        "epochs_total": epochs_total,
        "final_loss": round(final_loss, 8),
        "updated_at": now,
        "demo": "incremental_train",
    }
    metrics_payload: Dict[str, object] = {
        "mode": "incremental_training",
        "base_model_loaded": resumed,
        "base_model": args.model,
        "dataset": args.data,
        "rows": len(xs),
        "epochs_this_run": args.epochs,
        "epochs_total": epochs_total,
        "runs": runs,
        "lr": args.lr,
        "start_loss": round(start_loss, 8),
        "final_loss": round(final_loss, 8),
        "improvement": round(improvement, 8),
        "history": [round(value, 8) for value in history],
        "weights": {"w": round(w, 8), "b": round(b, 8)},
        "model_file": os.path.join(args.out, "model.json"),
        "updated_at": now,
    }

    model_out = os.path.join(args.out, "model.json")
    metrics_out = os.path.join(args.out, "metrics.json")
    write_json(model_out, model_payload)
    write_json(metrics_out, metrics_payload)
    log(f"[train] 写出 {model_out} (runs={runs}, epochs_total={epochs_total})")
    log(
        f"[train] 写出 {metrics_out} (start_loss={start_loss:.6f} "
        f"final_loss={final_loss:.6f} improvement={improvement:+.6f})"
    )

    # 就地更新 --model：本地连续运行两次即可看到第二次基于第一次权重继续训练。
    # 注意：远端作业运行在独立的 inputs 目录里，不会影响本地项目文件。
    if args.model and os.path.abspath(args.model) != os.path.abspath(model_out):
        write_json(args.model, model_payload)
        log(f"[train] 就地更新基座模型 {args.model}（下次运行可继续增量）")

    log("[train] 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
