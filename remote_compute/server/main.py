"""服务端 CLI 入口（``docs/PROTOCOL.md`` 第 6 节）。

用法::

    python -m server.main serve --host 0.0.0.0 --port 8765 --token <token> --data-dir <dir>
                              [--max-parallel 1] [--python-bin python3]
                              [--enable-docker] [--idle-timeout 300]
    python -m server.main gpus          # 打印探测到的 GPU（JSON）
    python -m server.main token         # 生成 token

约定：

- token 必填（``--token`` 或环境变量 ``RGPU_TOKEN``）；缺失时报错并以退出码 2 退出。
- 起服后必须向 stdout 打印一行（测试用来判断就绪）::

      RGPU_SERVER_READY host=<host> port=<port> data_dir=<dir>

- 其余运行日志走 stderr，避免污染 stdout 上的就绪行/JSON。
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from typing import List, Optional

from common.config import (
    DEFAULT_PORT,
    DEFAULT_SERVER_HOST,
    generate_token,
    server_config_from_args,
)
from common.protocol import PROTOCOL_VERSION

from . import __version__
from .app import ServerApp
from .scheduler import detect_gpus

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m server.main",
        description="RGPU 远程算力调度服务端（Linux；Windows 可 CPU 模式自测）",
    )
    parser.add_argument("--version", action="version",
                        version=f"rgpu-server {__version__} (protocol {PROTOCOL_VERSION})")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="启动服务端")
    serve.add_argument("--host", default=DEFAULT_SERVER_HOST,
                       help=f"监听地址（默认 {DEFAULT_SERVER_HOST}）")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT,
                       help=f"监听端口（默认 {DEFAULT_PORT}）")
    serve.add_argument("--token", default="",
                       help="鉴权 token（也可用环境变量 RGPU_TOKEN）")
    serve.add_argument("--data-dir", default="",
                       help="数据目录（默认 RGPU_DATA_DIR 或 ~/.rgpu-server）")
    serve.add_argument("--max-parallel", type=int, default=1,
                       help="最大并行作业数（默认 1）")
    serve.add_argument("--python-bin", default=sys.executable,
                       help="用于创建 venv 的解释器（默认当前解释器；Linux 可用 python3）")
    serve.add_argument("--enable-docker", action="store_true",
                       help="启用 docker 环境模式（需要 docker + nvidia-container-toolkit）")
    serve.add_argument("--strict-gpu", action="store_true",
                       help="GPU 作业必须获得空闲且满足显存要求的 GPU，失败时不使用 CPU")
    serve.add_argument("--strict-recovery", action="store_true",
                       help="重启后拒绝自动恢复归属不明的未完成作业，需人工核查")
    serve.add_argument("--min-gpu-free-mb", type=int, default=16384,
                       help="严格 GPU 模式的每卡最低空闲显存 MiB（默认 16384）")
    serve.add_argument("--weight-cache-bytes", type=int, default=0,
                       help="跨作业权重/数据内容寻址缓存配额（字节；0=禁用）")
    serve.add_argument("--idle-timeout", type=int, default=300,
                       help="会话空闲超时秒数（默认 300）")
    serve.set_defaults(func=cmd_serve)

    gpus = subparsers.add_parser("gpus", help="打印探测到的 GPU（JSON）")
    gpus.set_defaults(func=cmd_gpus)

    token = subparsers.add_parser("token", help="生成一个随机 token")
    token.add_argument("--bytes", type=int, default=24, dest="nbytes",
                       help="随机字节数（默认 24 → 32 字符）")
    token.set_defaults(func=cmd_token)
    return parser


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        stream=sys.stderr,
        force=True,
    )


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        config = server_config_from_args(args)
    except (AttributeError, TypeError) as exc:  # pragma: no cover - 参数映射兜底
        print(f"错误：参数解析失败: {exc}", file=sys.stderr)
        return 2
    if not str(config.token or "").strip():
        print(
            "错误：必须提供 token（--token <token> 或环境变量 RGPU_TOKEN）。\n"
            "      可以用 `python -m server.main token` 生成一个。",
            file=sys.stderr,
        )
        return 2
    _setup_logging()
    logging.getLogger("rgpu.server").info(
        "rgpu-server %s 启动（protocol %s）", __version__, PROTOCOL_VERSION
    )
    try:
        app = ServerApp(config)
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    def _handle_signal(signum, _frame):  # pragma: no cover - 信号路径
        logging.getLogger("rgpu.server").info("收到信号 %s，退出", signum)
        app.shutdown()

    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handle_signal)
        except (OSError, ValueError):  # pragma: no cover - 非主线程/平台限制
            pass

    try:
        app.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        app.shutdown()
    return 0


def cmd_gpus(_args: argparse.Namespace) -> int:
    gpus = detect_gpus(logger=lambda message: print(message, file=sys.stderr))
    payload: List[dict] = [gpu.to_dict() for gpu in gpus]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_token(args: argparse.Namespace) -> int:
    print(generate_token(int(getattr(args, "nbytes", 24) or 24)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:  # pragma: no cover - argparse required=True 已拦住
        parser.print_help()
        return 2
    return int(func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
