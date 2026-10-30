"""RGPU 客户端命令行入口。

契约见 ``docs/PROTOCOL.md`` 第 6 节（已冻结）::

    python -m client.main submit <PROJECT_DIR> --entry "python train.py ..." \\
        --server 127.0.0.1:8765 --token <token> [--requirements ...] [--env-mode venv|docker] \\
        [--include GLOB ...] [--exclude GLOB ...] [--outputs GLOB] [--gpu|--no-gpu] \\
        [--timeout 600] [--name demo] [--download-dir DIR] [--watch|--no-watch] [--json]
    python -m client.main status <JOB_ID> --server ... --token ...
    python -m client.main gui [--config <path>]
    python -m client.main config --server ... --token ... [--save]

退出码：0 任务成功；1 任务失败/超时；2 参数或配置错误；3 连接/网络错误；
4 服务端拒绝（鉴权、协议、环境构建失败等）。

``--json`` 时 stdout 只有一行 JSON（机器可读），人类可读日志全部走 stderr。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import protocol
from common.config import (
    DEFAULT_PORT,
    ClientConfig,
    load_client_config,
    parse_host_port,
    save_client_config,
)

from . import __version__
from .connection import ClientError
from .runner import (
    DEFAULT_DOWNLOAD_DIR,
    Callbacks,
    RunResult,
    SubmitOptions,
    query_job_status,
    submit_project,
)


class UsageError(Exception):
    """参数/配置错误 → 退出码 2。"""


# ---------------------------------------------------------------------------
# 终端打印
# ---------------------------------------------------------------------------
class CliPrinter:
    """把 runner 回调转成终端输出（真实时、节流进度）。"""

    def __init__(self, stream=None, prefix: str = "[rgpu]"):
        self.stream = stream or sys.stdout
        self.prefix = prefix
        self._last_pct = -100
        self._last_text = ""
        self._at_line_start = True

    def _ensure_newline(self) -> None:
        if not self._at_line_start:
            self.stream.write("\n")
            self._at_line_start = True

    def log(self, message: str) -> None:
        self._ensure_newline()
        self.stream.write(f"{self.prefix} {message}\n")
        self.stream.flush()

    def event(self, _stream: str, data: str) -> None:
        """远端日志原样输出（保持训练的流式体验）。"""
        if not data:
            return
        self.stream.write(data)
        self.stream.flush()
        self._at_line_start = data.endswith("\n")

    def progress(self, fraction: float, text: str) -> None:
        try:
            pct = int(round(max(0.0, min(1.0, float(fraction))) * 100))
        except (TypeError, ValueError):
            return
        text_changed = text != self._last_text
        if "%" in text:
            # 上传/下载百分比：每 10% 打印一次，避免逐分片刷屏
            should_print = pct >= self._last_pct + 10 or pct == 100 or not self._last_text
        else:
            # 阶段提示（连接/提交/环境就绪…）：文字变化就打印
            should_print = text_changed or pct >= self._last_pct + 10 or not self._last_text
        if not should_print:
            return
        self._last_pct = pct
        self._last_text = text
        self._ensure_newline()
        self.stream.write(f"{self.prefix} {text}\n")
        self.stream.flush()

    def job_id(self, job_id: str) -> None:
        self.log(f"作业 ID: {job_id}")

    def print_result(self, result: RunResult, json_mode: bool) -> None:
        if json_mode:
            # 机器可读结果固定写 stdout；人类可读日志走 self.stream（stderr）
            sys.stdout.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            sys.stdout.flush()
            return
        self._ensure_newline()
        if result.ok:
            self.stream.write(
                f"{self.prefix} 成功: job_id={result.job_id} "
                f"state={result.state} 用时 {result.duration_s:.1f}s\n"
            )
        else:
            error = result.error or {}
            self.stream.write(
                f"{self.prefix} 失败: job_id={result.job_id or '-'} "
                f"state={result.state} client_exit={result.client_exit} "
                f"error={error.get('code', '?')} {error.get('message', '')}\n"
            )
            if result.log_tail:
                self.stream.write(f"{self.prefix} 远端日志末尾:\n")
                for line in result.log_tail[-20:]:
                    self.stream.write(f"  {line}\n")
        self.stream.flush()


def _emit_usage_error(message: str, json_mode: bool) -> None:
    if json_mode:
        payload = {
            "ok": False,
            "state": "error",
            "client_exit": 2,
            "error": {"code": protocol.ERR_BAD_REQUEST, "message": message},
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    else:
        sys.stderr.write(f"错误: {message}\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------
def _resolve_config(args: argparse.Namespace) -> ClientConfig:
    """命令行参数 > 配置文件 > 默认值。"""
    config_path = getattr(args, "config", None)
    try:
        config = load_client_config(config_path)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc

    server = getattr(args, "server", None) or os.environ.get("RGPU_SERVER")
    if server:
        try:
            config.host, config.port = parse_host_port(server, DEFAULT_PORT)
        except (TypeError, ValueError) as exc:
            raise UsageError(f"--server 格式错误（应为 host:port）: {server}") from exc

    token = getattr(args, "token", None) or os.environ.get("RGPU_TOKEN")
    if token:
        config.token = str(token)
    return config


def _require_token(config: ClientConfig) -> None:
    if not config.token:
        raise UsageError(
            "缺少 token：请加 --token <token>，或用 "
            "`python -m client.main config --server <host:port> --token <token> --save` 保存配置"
        )


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------
def cmd_submit(args: argparse.Namespace) -> int:
    json_mode = bool(args.json)
    stream = sys.stderr if json_mode else sys.stdout
    printer = CliPrinter(stream)
    try:
        config = _resolve_config(args)
        _require_token(config)
        project_dir = str(Path(args.project_dir).expanduser())
        if not Path(project_dir).is_dir():
            raise UsageError(f"项目目录不存在或不是目录: {project_dir}")
        receipt_dir = getattr(args, "receipt_dir", None)
        if receipt_dir and not Path(receipt_dir).is_dir():
            raise UsageError(f"--receipt-dir 必须是已存在的目录: {receipt_dir}")
        if getattr(args, "request_id", None) and not receipt_dir:
            raise UsageError("--request-id 必须与 --receipt-dir 一起使用")
        options = SubmitOptions(
            project_dir=project_dir,
            entry=args.entry,
            job_name=args.name or "",
            requirements=args.requirements,
            env_mode=args.env_mode,
            include=list(args.include or []),
            exclude=list(args.exclude or []),
            outputs=list(args.outputs) if args.outputs else ["outputs/**"],
            gpu=bool(args.gpu),
            gpu_count=1,
            timeout_s=int(args.timeout),
            download_dir=args.download_dir or DEFAULT_DOWNLOAD_DIR,
            watch=bool(args.watch),
            incremental=args.incremental,
            python_version=args.python_version,
            chunk_size=args.chunk_size,
            receipt_dir=getattr(args, "receipt_dir", None),
            receipt_request_id=getattr(args, "request_id", None),
        )
    except UsageError as exc:
        _emit_usage_error(str(exc), json_mode)
        return 2

    callbacks = Callbacks(
        log=printer.log,
        event=printer.event,
        progress=printer.progress,
        job_id=printer.job_id,
    )
    printer.log(f"服务器 {config.address}，项目 {project_dir}")
    result = submit_project(config, options, callbacks)
    printer.print_result(result, json_mode)
    return int(result.client_exit)


def cmd_status(args: argparse.Namespace) -> int:
    json_mode = bool(args.json)
    try:
        config = _resolve_config(args)
        _require_token(config)
        if not args.job_id:
            raise UsageError("job_id 不能为空")
    except UsageError as exc:
        _emit_usage_error(str(exc), json_mode)
        return 2

    try:
        status = query_job_status(config, args.job_id)
    except ClientError as exc:
        payload = {
            "ok": False,
            "job_id": args.job_id,
            "state": "error",
            "client_exit": int(exc.exit_code),
            "error": {"code": exc.code, "message": str(exc)},
        }
        if json_mode:
            sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        else:
            sys.stderr.write(f"错误: {exc}\n")
        return int(exc.exit_code)

    state = str(status.get("state") or "")
    if state == "succeeded":
        client_exit = 0
    elif state in ("failed", "timeout", "cancelled"):
        client_exit = 1
    elif state in ("queued", "uploading", "env_building", "running", "started"):
        client_exit = 0
    else:
        client_exit = 4

    if json_mode:
        data: Dict[str, Any] = dict(status)
        data["ok"] = state == "succeeded"
        data["client_exit"] = client_exit
        sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
        return client_exit

    outputs = status.get("outputs") or []
    print(f"job_id      : {status.get('job_id', args.job_id)}")
    print(f"state       : {state}")
    print(f"progress    : {status.get('progress', 0)}")
    print(f"outputs     : {len(outputs)} 个")
    for item in outputs:
        if isinstance(item, dict):
            print(f"  - {item.get('path')} ({item.get('size')} B)")
    print(f"created_at  : {status.get('created_at')}")
    print(f"started_at  : {status.get('started_at')}")
    print(f"finished_at : {status.get('finished_at')}")
    return client_exit


def cmd_config(args: argparse.Namespace) -> int:
    try:
        config = _resolve_config(args)
        if args.save:
            path = save_client_config(config, args.config)
            print(f"配置已保存: {path}")
    except UsageError as exc:
        sys.stderr.write(f"错误: {exc}\n")
        return 2
    except OSError as exc:
        sys.stderr.write(f"错误: 无法写入配置文件: {exc}\n")
        return 2
    data = config.to_dict()
    data["config_path"] = str(
        Path(args.config).expanduser() if args.config else _default_config_path()
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def _default_config_path() -> Path:
    from common.config import default_client_config_path

    return default_client_config_path()


def cmd_gui(args: argparse.Namespace) -> int:
    from . import gui

    return int(gui.main(config_path=args.config))


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--server", metavar="HOST:PORT", default=None, help="服务端地址，如 127.0.0.1:8765")
    p.add_argument("--token", metavar="TOKEN", default=None, help="鉴权 token")
    p.add_argument("--config", metavar="PATH", default=None, help="客户端配置文件路径")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m client.main",
        description="RGPU 远程算力调度客户端（提交任务 / 查状态 / 一键增量训练 GUI）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "退出码: 0 成功 | 1 任务失败/超时 | 2 参数配置错误 | "
            "3 连接错误 | 4 服务端拒绝\n"
            "示例: python -m client.main submit examples/incremental_train "
            '--entry "python train.py --epochs 3 --data data.csv" '
            "--server 127.0.0.1:8765 --token <token>"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"rgpu-client {__version__} (protocol {protocol.PROTOCOL_VERSION})",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # submit
    p = sub.add_parser("submit", help="提交项目目录到远端执行", description="提交任务并跟踪到结束")
    p.add_argument("project_dir", metavar="PROJECT_DIR", help="本地项目目录")
    p.add_argument(
        "--entry",
        metavar="CMD",
        required=True,
        help='入口命令（argv 形式，不加 shell），如 "python train.py --epochs 3 --data data.csv"',
    )
    p.add_argument("--requirements", metavar="PATH", default=None, help="依赖文件（默认自动识别项目下 requirements.txt）")
    p.add_argument("--env-mode", choices=("venv", "docker"), default="venv", help="远端环境模式（默认 venv）")
    p.add_argument("--include", action="append", default=[], metavar="GLOB", help="只上传匹配的文件（可多次）")
    p.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="排除匹配的文件/目录（可多次）")
    p.add_argument("--outputs", action="append", default=None, metavar="GLOB", help='产物匹配模式（默认 "outputs/**"）')
    p.add_argument("--gpu", dest="gpu", action="store_true", default=False, help="请求 GPU（默认 --no-gpu）")
    p.add_argument("--no-gpu", dest="gpu", action="store_false", help="不请求 GPU")
    p.add_argument("--timeout", type=int, default=600, metavar="SECONDS", help="远端任务超时（默认 600s）")
    p.add_argument("--name", metavar="NAME", default="", help="作业名（默认取目录名）")
    p.add_argument("--download-dir", metavar="DIR", default=None, help=f"产物下载目录（默认 {DEFAULT_DOWNLOAD_DIR}）")
    p.add_argument("--watch", dest="watch", action="store_true", default=True, help="跟踪日志与产物（默认）")
    p.add_argument("--no-watch", dest="watch", action="store_false", help="上传并启动后立即返回，不跟踪")
    p.add_argument("--json", action="store_true", help="stdout 输出机器可读 JSON（日志走 stderr）")
    p.add_argument(
        "--incremental",
        dest="incremental",
        action="store_true",
        default=None,
        help="强制按增量训练填写 meta（默认自动识别 model.json + data.csv）",
    )
    p.add_argument(
        "--no-incremental",
        dest="incremental",
        action="store_false",
        help="不填增量训练 meta",
    )
    p.add_argument("--python-version", metavar="X.Y", default=None, help="环境指纹里的 Python 版本（默认本机版本）")
    p.add_argument("--chunk-size", type=int, default=None, metavar="BYTES", help="上传分片大小（默认采用服务端建议值）")
    p.add_argument("--receipt-dir", metavar="DIR", default=None,
                   help="启用客户端持久投递账本（目录必须已存在；禁止自动重投）")
    p.add_argument("--request-id", metavar="ID", default=None,
                   help="账本请求 ID（仅与 --receipt-dir 同用；默认由 spec 派生）")
    _add_common(p)

    # status
    p = sub.add_parser("status", help="查询作业状态", description="查询作业状态")
    p.add_argument("job_id", metavar="JOB_ID")
    p.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    _add_common(p)

    # gui
    p = sub.add_parser("gui", help="打开一键增量训练窗口", description="tkinter 一键增量训练 GUI")
    p.add_argument("--config", metavar="PATH", default=None, help="客户端配置文件路径")
    p.set_defaults(server=None, token=None)

    # config
    p = sub.add_parser("config", help="查看/保存客户端配置", description="查看/保存客户端配置")
    _add_common(p)
    p.add_argument("--save", action="store_true", help="写入配置文件")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "submit": cmd_submit,
        "status": cmd_status,
        "gui": cmd_gui,
        "config": cmd_config,
    }
    handler = handlers.get(args.command)
    if handler is None:  # pragma: no cover - argparse 已保证
        parser.error(f"未知子命令: {args.command}")
        return 2
    try:
        return int(handler(args))
    except KeyboardInterrupt:
        sys.stderr.write("\n已中断\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
