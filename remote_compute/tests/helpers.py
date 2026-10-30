"""测试公共辅助：路径常量、临时目录、服务端子进程、原始 socket 客户端。

设计约束（实测结论，见 ``开发流程.md``）：

- 只依赖标准库与 ``common/``；**不 import** ``server`` / ``client`` 包，
  这样阶段 A 在 server/client 尚未落地时也能跑。
- 临时目录放在 ``<项目根>/_rgpu_data/tests/``。系统 temp 目录在 DSH 沙箱下
  **只能建目录、写文件被拒**（``PermissionError`` + ``[sandbox: file access denied]``），
  因此不用 ``tempfile.gettempdir()``；``_rgpu_data/`` 已在 ``.gitignore`` 中。
- 服务端子进程用项目 venv 的 ``python -m server.main serve ...`` 启动，
  从 stdout 等待 ``RGPU_SERVER_READY``；超时信息里带完整服务端日志，便于定位。
- 所有子进程统一 UTF-8 输出（``PYTHONIOENCODING`` / ``PYTHONUTF8``），避免 Windows GBK 乱码。
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import unittest
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 允许 ``python tests/test_xxx.py`` 直接运行时也能 import common
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.is_file():  # 兜底：venv 缺失时用当前解释器
    PYTHON = Path(sys.executable)
PY = str(PYTHON)

SERVER_AVAILABLE = (PROJECT_ROOT / "server" / "main.py").is_file()
CLIENT_AVAILABLE = (PROJECT_ROOT / "client" / "main.py").is_file()
EXAMPLE_PROJECT = PROJECT_ROOT / "examples" / "incremental_train"
EXAMPLE_AVAILABLE = (EXAMPLE_PROJECT / "train.py").is_file()

TMP_ROOT = PROJECT_ROOT / "_rgpu_data" / "tests"

from common.hashing import sha256_file  # noqa: E402
from common.protocol import (  # noqa: E402
    KIND_ERROR,
    KIND_HELLO,
    FrameReader,
    FrameWriter,
)


# ---------------------------------------------------------------------------
# 临时目录 / 环境变量
# ---------------------------------------------------------------------------
def make_temp_dir(prefix: str = "case-") -> Path:
    """在项目工作区内创建唯一临时目录（沙箱下唯一可写位置）。"""
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / f"{prefix}{uuid.uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def cleanup_dir(path: Optional[Path]) -> None:
    """尽力删除临时目录（不抛异常，避免掩盖真正的失败原因）。"""
    if path is None:
        return
    shutil.rmtree(path, ignore_errors=True)


def child_env(**overrides: Any) -> Dict[str, str]:
    """构造子进程环境：强制 UTF-8，PYTHONPATH 指向项目根。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return env


def run_capture(cmd: Iterable[str], *, cwd: Optional[Path] = None, timeout: float = 120.0,
                env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    """统一以 UTF-8 捕获子进程输出。"""
    return subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd or PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=child_env() if env is None else env,
    )


# ---------------------------------------------------------------------------
# 端口 / 轮询
# ---------------------------------------------------------------------------
def find_free_port() -> int:
    """让系统分配一个当前空闲的本地端口（避免固定端口冲突）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_port(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    last: Optional[BaseException] = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError as exc:  # 连接被拒 = 尚未监听
            last = exc
            time.sleep(0.05)
    raise TimeoutError(f"{host}:{port} 在 {timeout}s 内未监听: {last}")


def wait_until(predicate: Callable[[], Any], timeout: float = 5.0,
               interval: float = 0.05, message: str = "条件未满足") -> Any:
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise TimeoutError(f"{message}（等待 {timeout}s）")


# ---------------------------------------------------------------------------
# 服务端子进程
# ---------------------------------------------------------------------------
class ServerProcess:
    """启动 ``python -m server.main serve`` 并等待 ``RGPU_SERVER_READY``。

    用法::

        server = ServerProcess(token="e2e-token").start()
        try:
            ...
        finally:
            server.stop()

    或 ``with ServerProcess() as server: ...``。
    """

    READY_MARKER = "RGPU_SERVER_READY"

    def __init__(self, *, port: Optional[int] = None, token: str = "e2e-token",
                 data_dir: Optional[Path] = None, extra_args: Iterable[str] = (),
                 ready_timeout: float = 20.0, host: str = "127.0.0.1",
                 name: str = "server") -> None:
        self.host = host
        self.port = int(port) if port else find_free_port()
        self.token = token
        self._owns_data_dir = data_dir is None
        self.data_dir = Path(data_dir) if data_dir is not None else make_temp_dir("server-data-")
        self.extra_args = [str(x) for x in extra_args]
        self.ready_timeout = float(ready_timeout)
        self.name = name
        self.command: List[str] = []
        self.ready_line = ""
        self.proc: Optional[subprocess.Popen] = None
        self._log: List[str] = []
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    # -- 生命周期 ----------------------------------------------------------
    def start(self) -> "ServerProcess":
        if self.proc is not None:
            raise RuntimeError("ServerProcess 只能 start 一次")
        self.command = [
            PY, "-m", "server.main", "serve",
            "--host", self.host, "--port", str(self.port),
            "--token", self.token,
            "--data-dir", str(self.data_dir),
            *self.extra_args,
        ]
        self.proc = subprocess.Popen(
            self.command,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env(),
        )
        self._thread = threading.Thread(target=self._pump_output,
                                        name=f"{self.name}-log", daemon=True)
        self._thread.start()
        try:
            self._wait_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def stop(self, cleanup_data_dir: Optional[bool] = None) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=5)
        stream = proc.stdout if proc is not None else None
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
        if cleanup_data_dir is None:
            cleanup_data_dir = self._owns_data_dir
        if cleanup_data_dir:
            cleanup_dir(self.data_dir)

    def __enter__(self) -> "ServerProcess":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    # -- 状态 / 日志 -------------------------------------------------------
    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def log(self) -> str:
        with self._lock:
            return "\n".join(self._log)

    def describe(self) -> str:
        code = None if self.proc is None else self.proc.poll()
        return (
            f"服务端命令: {' '.join(self.command)}\n"
            f"退出码: {code}\n"
            f"数据目录: {self.data_dir}\n"
            f"--- 服务端日志 ---\n{self.log()}"
        )

    # -- 内部 --------------------------------------------------------------
    def _pump_output(self) -> None:
        stream = self.proc.stdout if self.proc is not None else None
        try:
            if stream is not None:
                for line in stream:
                    text = line.rstrip("\r\n")
                    with self._lock:
                        self._log.append(text)
                    self._queue.put(text)
        finally:
            self._queue.put(None)

    def _wait_ready(self) -> None:
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            try:
                line = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError(
                        f"服务端启动即退出（exit={self.proc.returncode}）\n{self.describe()}"
                    )
                continue
            if line is None:  # 日志流结束 = 进程退出
                raise RuntimeError(f"服务端日志流结束，进程退出\n{self.describe()}")
            if self.READY_MARKER in line:
                self.ready_line = line.strip()
                wait_for_port(self.host, self.port, timeout=5.0)
                return
        raise RuntimeError(
            f"等待 {self.READY_MARKER} 超时（{self.ready_timeout}s）\n{self.describe()}"
        )


# ---------------------------------------------------------------------------
# 原始 socket 客户端（协议层测试 / 断线续传测试用）
# ---------------------------------------------------------------------------
class RawClient:
    """不带重连逻辑的最小协议客户端，方便精确控制帧。"""

    def __init__(self, address: str, *, timeout: float = 10.0,
                 token: str = "e2e-token") -> None:
        self.address = address
        host, _, port = address.rpartition(":")
        self.host = host or "127.0.0.1"
        self.port = int(port)
        self.timeout = float(timeout)
        self.token = token
        self.sock: Optional[socket.socket] = None
        self.reader: Optional[FrameReader] = None
        self.writer: Optional[FrameWriter] = None

    def connect(self) -> "RawClient":
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        self.reader = FrameReader(self.sock)
        self.writer = FrameWriter(self.sock)
        return self

    def close(self) -> None:
        sock, self.sock = self.sock, None
        self.reader = None
        self.writer = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    # -- 帧收发 ------------------------------------------------------------
    def send_json(self, obj: Dict[str, Any]) -> None:
        assert self.writer is not None, "not connected"
        self.writer.send_json(obj)

    def send_blob(self, data: bytes) -> None:
        assert self.writer is not None, "not connected"
        self.writer.send_blob(data)

    def recv_json(self) -> Dict[str, Any]:
        assert self.reader is not None, "not connected"
        return self.reader.read_json()

    def recv_blob(self) -> bytes:
        assert self.reader is not None, "not connected"
        return self.reader.read_blob()

    # -- 常用消息 ----------------------------------------------------------
    def hello(self, token: Optional[str] = None, protocol: int = 1,
              role: str = "client") -> Dict[str, Any]:
        self.send_json({
            "kind": KIND_HELLO,
            "protocol": protocol,
            "role": role,
            "client_version": "0.1.0",
            "token": self.token if token is None else token,
        })
        return self.recv_json()

    def expect_error(self, expected_code: str) -> Dict[str, Any]:
        msg = self.recv_json()
        if msg.get("kind") != KIND_ERROR or msg.get("code") != expected_code:
            raise AssertionError(f"期望 error/{expected_code}，实际收到 {msg!r}")
        return msg

    def __enter__(self) -> "RawClient":
        return self.connect()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def connect_raw(server: ServerProcess, **kwargs: Any) -> RawClient:
    client = RawClient(server.address, token=server.token, **kwargs)
    return client.connect()


# ---------------------------------------------------------------------------
# 客户端 CLI / 示例项目
# ---------------------------------------------------------------------------
def client_command(*args: Any) -> List[str]:
    return [PY, "-m", "client.main", *[str(a) for a in args]]


def run_client(*args: Any, timeout: float = 180.0,
               env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    """调用客户端 CLI（``python -m client.main ...``），返回 CompletedProcess。"""
    return run_capture(client_command(*args), timeout=timeout, env=env)


def output_of(proc: subprocess.CompletedProcess) -> str:
    return (proc.stdout or "") + (proc.stderr or "")


def describe_process(proc: subprocess.CompletedProcess) -> str:
    return (
        f"命令: {' '.join(map(str, proc.args))}\n"
        f"退出码: {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )


def copy_example_project(dest: Path) -> Path:
    """把 ``examples/incremental_train`` 复制到临时目录（不带历史产物）。"""
    if not EXAMPLE_AVAILABLE:
        raise unittest.SkipTest(f"示例项目尚未落地: {EXAMPLE_PROJECT}")
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        EXAMPLE_PROJECT,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "outputs", "*.pyc", ".git"),
    )
    return dest


def run_local_train(project_dir: Path, *extra: Any, timeout: float = 120.0,
                    cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """本地直接跑示例训练脚本（生成/更新 model.json），用于增量语义对照。"""
    cmd = [PY, "train.py", "--epochs", "2", "--data", "data.csv",
           "--out", "outputs", "--model", "model.json", *[str(x) for x in extra]]
    return run_capture(cmd, cwd=Path(cwd) if cwd else project_dir, timeout=timeout)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "PROJECT_ROOT",
    "PY",
    "PYTHON",
    "SERVER_AVAILABLE",
    "CLIENT_AVAILABLE",
    "EXAMPLE_PROJECT",
    "EXAMPLE_AVAILABLE",
    "TMP_ROOT",
    "make_temp_dir",
    "cleanup_dir",
    "child_env",
    "run_capture",
    "find_free_port",
    "wait_for_port",
    "wait_until",
    "ServerProcess",
    "RawClient",
    "connect_raw",
    "client_command",
    "run_client",
    "output_of",
    "describe_process",
    "copy_example_project",
    "run_local_train",
    "read_json",
    "sha256_file",
]
