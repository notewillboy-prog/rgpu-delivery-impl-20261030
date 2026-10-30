"""客户端自测：本地假服务端（threading + common.protocol）跑通全链路。

不依赖 ``server/``，只用标准库 + ``common/`` 协议层：

1. ``packager``：忽略规则、POSIX 路径、sha256、include/exclude、requirements、
   增量训练 meta、JobSpec 构建；
2. 成功链路：submit → job_accepted → 上传（**注入一次分片 HASH_MISMATCH 重传** +
   **上传中途断线 → 重连 → resume_job_id 续传**）→ job_start → 流式 job_event →
   job_finished → output_list + 分片下载（sha256 校验）；
3. 失败链路：远端任务失败 → 客户端退出码 1；
4. 鉴权失败 → 退出码 4；连接被拒 → 退出码 3；
5. CLI ``--json`` 输出契约、``status``、``config --save``。

运行（项目根目录）::

    .venv\\Scripts\\python.exe client/selftest.py
"""

from __future__ import annotations

import io
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- 直接脚本运行时的导入引导 ---------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common import protocol  # noqa: E402
from common.config import ClientConfig  # noqa: E402
from common.hashing import sha256_bytes, sha256_file  # noqa: E402
from common.models import JobSpec  # noqa: E402

from client import packager  # noqa: E402
from client import main as client_main  # noqa: E402
from client.runner import (  # noqa: E402
    Callbacks,
    SubmitOptions,
    submit_project,
)

SELFTEST_TOKEN = "selftest-token"
FAULT_FILE = "zz_big.bin"
DROP_THRESHOLD = 128  # zz_big.bin 上传到 128 B 时注入断线
SERVER_CHUNK_SIZE = 64  # job_accepted 里建议的上传分片大小
OUTPUT_CHUNK_SIZE = 32  # 产物下载分片（tests 多分片下载）


def log(message: str) -> None:
    print(message, flush=True)


def check(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    log(f"  [PASS] {message}")


def make_temp_dir(parent: Path, prefix: str) -> Path:
    """在指定父目录下建临时目录（默认权限）。

    注意：不能用 ``tempfile.mkdtemp()`` / ``TemporaryDirectory`` —— 它们以
    ``0o700`` 建目录，在 DSH Windows 沙箱里该目录会变成「后续不可写、不可删」，
    导致任何嵌套文件操作 ``WinError 5``。这里用默认 mode 创建，测试结束后清理。
    """
    parent.mkdir(parents=True, exist_ok=True)
    stamp = f"{prefix}_{os.getpid()}_{int(time.time() * 1000)}"
    for attempt in range(1000):
        candidate = parent / (stamp if attempt == 0 else f"{stamp}_{attempt}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"无法创建临时目录: {parent}/{stamp}")


def cleanup_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# 假服务端
# ---------------------------------------------------------------------------
class FakeServer:
    """按 PROTOCOL v1 实现一个最小的 RGPU 服务端，用于客户端自测。"""

    def __init__(
        self,
        token: str = SELFTEST_TOKEN,
        *,
        base_dir: Path,
        drop_once: bool = False,
        fail_job: bool = False,
        fail_exit_code: int = 3,
        real_exec: bool = False,
    ):
        self.token = token
        self.drop_once = drop_once
        self.fail_job = fail_job
        self.fail_exit_code = fail_exit_code
        #: True 时真正执行 entry_argv（跑真实训练脚本），用于本地端到端验证
        self.real_exec = real_exec

        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.listener.settimeout(0.2)
        self.host, self.port = self.listener.getsockname()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active: Optional[socket.socket] = None
        self._lock = threading.Lock()

        self.data_dir = make_temp_dir(Path(base_dir), "fake-server")
        self.connections = 0
        self.saw_resume = False
        self.hash_fault_triggered = False
        self.dropped = False
        self.job_id = ""
        self.spec: Optional[JobSpec] = None
        self.expected: Dict[str, Tuple[int, str]] = {}
        self.partial: Dict[str, bytearray] = {}
        self.completed: set[str] = set()
        self.output_entries: List[Dict[str, Any]] = []
        self.output_source_root: Optional[Path] = None
        self.finished: Optional[Dict[str, Any]] = None
        self.server_log: List[str] = []
        self.job_created_at = time.time()

    # -- 生命周期 ---------------------------------------------------------
    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    def start(self) -> "FakeServer":
        self._thread = threading.Thread(
            target=self._serve, name="fake-rgpu-server", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            active = self._active
        if active is not None:
            try:
                active.close()
            except OSError:
                pass
        try:
            self.listener.close()
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _log(self, message: str) -> None:
        self.server_log.append(message)

    # -- accept 循环 ------------------------------------------------------
    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _addr = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.connections += 1
            with self._lock:
                self._active = conn
            try:
                self._handle_connection(conn)
            except (ConnectionError, OSError, protocol.ProtocolError) as exc:
                self._log(f"连接 #{self.connections} 结束: {exc}")
            finally:
                with self._lock:
                    self._active = None
                try:
                    conn.close()
                except OSError:
                    pass

    # -- 单连接 -----------------------------------------------------------
    def _handle_connection(self, conn: socket.socket) -> None:
        conn.settimeout(10.0)
        reader = protocol.FrameReader(conn)
        writer = protocol.FrameWriter(conn)

        hello = reader.read_json()
        if hello.get("kind") != protocol.KIND_HELLO:
            raise protocol.ProtocolError(f"首条消息不是 hello: {hello.get('kind')}")
        if int(hello.get("protocol") or 0) != protocol.PROTOCOL_VERSION:
            writer.send_json(
                protocol.make_error(
                    protocol.ERR_BAD_PROTOCOL,
                    f"protocol mismatch: {hello.get('protocol')}",
                )
            )
            return
        if hello.get("token") != self.token:
            writer.send_json(protocol.make_error(protocol.ERR_BAD_AUTH, "token 无效"))
            return
        writer.send_json(
            {
                "kind": protocol.KIND_HELLO_OK,
                "protocol": protocol.PROTOCOL_VERSION,
                "server_id": "srv-selftest",
                "hostname": "selftest-node",
                "platform": "linux",
                "python": "3.12.14",
                "max_frame": protocol.MAX_FRAME_PAYLOAD,
                "gpus": [
                    {
                        "index": 0,
                        "name": "Fake GPU (selftest)",
                        "memory_total_mb": 8192,
                        "memory_free_mb": 8000,
                        "driver": "999.99",
                        "cuda": "12.4",
                    }
                ],
                "max_parallel_jobs": 1,
                "env_cache_dir": str(self.data_dir / "envs"),
            }
        )

        while not self._stop.is_set():
            try:
                msg = reader.read_json()
            except protocol.ProtocolError as exc:
                if exc.code == protocol.ERR_TIMEOUT:
                    return
                raise
            kind = msg.get("kind")
            if kind == protocol.KIND_PING:
                writer.send_json({"kind": protocol.KIND_PONG, "ts": time.time()})
            elif kind == protocol.KIND_BYE:
                return
            elif kind == protocol.KIND_JOB_SUBMIT:
                self._on_job_submit(msg, writer)
            elif kind == protocol.KIND_FILE_CHUNK:
                self._on_file_chunk(msg, reader, writer)
            elif kind == protocol.KIND_FILE_DONE:
                self._on_file_done(msg, writer)
            elif kind == protocol.KIND_JOB_START:
                self._on_job_start(msg, writer)
            elif kind == protocol.KIND_JOB_STATUS:
                self._on_job_status(writer)
            elif kind == protocol.KIND_OUTPUT_LIST:
                self._on_output_list(writer)
            elif kind == protocol.KIND_OUTPUT_CHUNK:
                self._on_output_chunk(msg, writer)
            else:
                writer.send_json(
                    protocol.make_error(
                        protocol.ERR_BAD_REQUEST, f"未知消息 kind={kind!r}"
                    )
                )

    # -- 任务 -------------------------------------------------------------
    def _on_job_submit(self, msg: Dict[str, Any], writer: protocol.FrameWriter) -> None:
        spec = JobSpec.from_dict(msg.get("spec") or {})
        resume = msg.get("resume_job_id")
        if resume and resume == self.job_id and self.spec is not None:
            self.saw_resume = True
            self._log(f"收到 resume_job_id={resume}，续传")
        else:
            self.job_id = f"job-selftest-{self.connections:04d}"
            self.spec = spec
            self.expected = {e.path: (e.size, e.sha256) for e in spec.inputs}
            self.partial = {path: bytearray() for path in self.expected}
            self.completed = set()
            self.output_entries = []
            self.finished = None
            self.job_created_at = time.time()
            self._log(f"新任务 {self.job_id}：{len(spec.inputs)} 个输入文件")
        missing = [
            {"path": path, "size": size, "sha256": digest}
            for path, (size, digest) in self.expected.items()
            if path not in self.completed
        ]
        writer.send_json(
            {
                "kind": protocol.KIND_JOB_ACCEPTED,
                "job_id": self.job_id,
                "missing": missing,
                "present": sorted(self.completed),
                "env": {
                    "fingerprint": spec.env.fingerprint,
                    "reuse": True,
                },
                "chunk_size": SERVER_CHUNK_SIZE,
            }
        )

    def _on_file_chunk(
        self,
        header: Dict[str, Any],
        reader: protocol.FrameReader,
        writer: protocol.FrameWriter,
    ) -> None:
        blob = reader.read_blob()
        path = str(header.get("path"))
        offset = int(header.get("offset") or 0)
        length = int(header.get("length") or 0)
        chunk_sha = str(header.get("sha256") or "")
        if path not in self.expected:
            writer.send_json(
                protocol.make_error(protocol.ERR_NOT_FOUND, f"未知文件 {path}")
            )
            return
        if length != len(blob) or sha256_bytes(blob) != chunk_sha:
            writer.send_json(
                protocol.make_error(protocol.ERR_HASH_MISMATCH, "分片哈希不匹配")
            )
            return
        # 故障注入 1：FAULT_FILE 的第一个分片故意回 HASH_MISMATCH，验证客户端重传
        if path == FAULT_FILE and not self.hash_fault_triggered:
            self.hash_fault_triggered = True
            self._log("注入分片 HASH_MISMATCH（客户端应重发该分片）")
            writer.send_json(
                protocol.make_error(protocol.ERR_HASH_MISMATCH, "注入的故障：分片哈希不匹配")
            )
            return
        buf = self.partial.setdefault(path, bytearray())
        if offset > len(buf):
            writer.send_json(
                protocol.make_error(protocol.ERR_HASH_MISMATCH, f"offset 跳跃: {offset}")
            )
            return
        del buf[offset:]
        buf.extend(blob)
        # 故障注入 2：上传该文件中途直接断开，验证 resume_job_id 续传
        if (
            self.drop_once
            and not self.dropped
            and path == FAULT_FILE
            and len(buf) >= DROP_THRESHOLD
        ):
            self.dropped = True
            self._log("注入断线：上传中途关闭连接")
            raise ConnectionError("injected connection drop")
        writer.send_json(
            {
                "kind": protocol.KIND_FILE_OK,
                "path": path,
                "offset": offset + length,
            }
        )

    def _on_file_done(self, msg: Dict[str, Any], writer: protocol.FrameWriter) -> None:
        path = str(msg.get("path"))
        if path not in self.expected:
            writer.send_json(
                protocol.make_error(protocol.ERR_NOT_FOUND, f"未知文件 {path}")
            )
            return
        size, expected_sha = self.expected[path]
        data = bytes(self.partial.get(path, bytearray()))
        if len(data) != size or sha256_bytes(data) != expected_sha:
            self.partial[path] = bytearray()
            writer.send_json(
                protocol.make_error(
                    protocol.ERR_HASH_MISMATCH,
                    f"整文件哈希不匹配: {path}",
                )
            )
            return
        target = self.data_dir / "jobs" / self.job_id / "inputs" / Path(*path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.completed.add(path)
        self._log(f"文件上传完成: {path} ({size} B)")
        writer.send_json(
            {"kind": protocol.KIND_FILE_OK, "path": path, "done": True}
        )

    def _on_job_start(self, msg: Dict[str, Any], writer: protocol.FrameWriter) -> None:
        if self.completed != set(self.expected):
            writer.send_json(
                protocol.make_error(
                    protocol.ERR_CONFLICT,
                    f"启动时仍有缺失文件: {sorted(set(self.expected) - self.completed)}",
                )
            )
            return
        writer.send_json(
            {
                "kind": protocol.KIND_JOB_PROGRESS,
                "job_id": self.job_id,
                "state": "env_building",
                "progress": 0.4,
                "message": "检测环境指纹，命中缓存",
            }
        )
        writer.send_json(
            {
                "kind": protocol.KIND_JOB_STARTED,
                "job_id": self.job_id,
                "env": {"fingerprint": self.spec.env.fingerprint, "state": "cached"},
            }
        )
        if self.real_exec:
            self._run_real_job(writer)
            return

        if self.fail_job:
            time.sleep(0.02)
            writer.send_json(
                {
                    "kind": protocol.KIND_JOB_FINISHED,
                    "job_id": self.job_id,
                    "state": "failed",
                    "exit_code": int(self.fail_exit_code),
                    "duration_s": 0.1,
                    "outputs": [],
                    "log_tail": [
                        "Traceback (most recent call last):",
                        '  File "train.py", line 1, in <module>',
                        "RuntimeError: 模拟训练失败",
                    ],
                }
            )
            self.finished = {"state": "failed"}
            return

        # 流式 job_event（客户端应实时打印）
        for seq in range(1, 4):
            writer.send_json(
                {
                    "kind": protocol.KIND_JOB_EVENT,
                    "job_id": self.job_id,
                    "seq": seq,
                    "stream": "stdout",
                    "data": f"epoch {seq}/3 loss=0.{100 - seq * 10}\n",
                    "ts": time.time(),
                }
            )
            time.sleep(0.02)
        outputs = self._create_outputs()
        writer.send_json(
            {
                "kind": protocol.KIND_JOB_FINISHED,
                "job_id": self.job_id,
                "state": "succeeded",
                "exit_code": 0,
                "duration_s": 0.3,
                "outputs": outputs,
                "log_tail": ["epoch 3/3 loss=0.70", "done"],
            }
        )
        self.finished = {"state": "succeeded"}
        self._log(f"作业完成，产物 {len(outputs)} 个")

    def _run_real_job(self, writer: protocol.FrameWriter) -> None:
        """真正执行入口命令（把 argv[0] 的 python 换成本解释器），流式回传日志。"""
        assert self.spec is not None
        inputs_dir = self.data_dir / "jobs" / self.job_id / "inputs"
        argv = [str(x) for x in self.spec.entry_argv]
        if argv and argv[0] in ("python", "python3"):
            argv[0] = sys.executable
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        started = time.time()
        tail: List[str] = []
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(inputs_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except OSError as exc:
            writer.send_json(
                {
                    "kind": protocol.KIND_JOB_FINISHED,
                    "job_id": self.job_id,
                    "state": "failed",
                    "exit_code": -1,
                    "duration_s": 0.0,
                    "outputs": [],
                    "log_tail": [f"启动作业进程失败: {exc}"],
                }
            )
            self.finished = {"state": "failed"}
            return
        seq = 0
        assert proc.stdout is not None
        for line in proc.stdout:
            seq += 1
            tail.append(line.rstrip("\r\n"))
            writer.send_json(
                {
                    "kind": protocol.KIND_JOB_EVENT,
                    "job_id": self.job_id,
                    "seq": seq,
                    "stream": "stdout",
                    "data": line,
                    "ts": time.time(),
                }
            )
        exit_code = proc.wait(timeout=120)
        outputs = self._collect_real_outputs(inputs_dir)
        duration = time.time() - started
        state = "succeeded" if exit_code == 0 else "failed"
        writer.send_json(
            {
                "kind": protocol.KIND_JOB_FINISHED,
                "job_id": self.job_id,
                "state": state,
                "exit_code": int(exit_code),
                "duration_s": duration,
                "outputs": outputs,
                "log_tail": tail[-20:],
            }
        )
        self.finished = {"state": state}
        self._log(f"真实作业结束: state={state} exit={exit_code} outputs={len(outputs)}")

    def _collect_real_outputs(self, inputs_dir: Path) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        out_root = inputs_dir / "outputs"
        if not out_root.is_dir():
            return entries
        for path in sorted(out_root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(inputs_dir).as_posix()
            data = path.read_bytes()
            entries.append(
                {
                    "path": rel,
                    "size": len(data),
                    "sha256": sha256_bytes(data),
                    "kind": "file",
                }
            )
        self.output_entries = entries
        self.output_source_root = inputs_dir
        self.job_finished_at = time.time()
        return entries

    def _create_outputs(self) -> List[Dict[str, Any]]:
        job_dir = self.data_dir / "jobs" / self.job_id
        files: Dict[str, bytes] = {
            "outputs/model.json": b'{"w": 1.7, "b": 0.9, "runs": 1}\n',
            "outputs/metrics.json": b'{"final_loss": 0.001, "runs": 1}\n',
            "outputs/sub/note.txt": b"fake incremental training output note\n" * 2,
        }
        entries: List[Dict[str, Any]] = []
        for rel, data in files.items():
            target = job_dir / Path(*rel.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            entries.append(
                {
                    "path": rel,
                    "size": len(data),
                    "sha256": sha256_bytes(data),
                    "kind": "file",
                }
            )
        entries.sort(key=lambda item: item["path"])
        self.output_entries = entries
        self.output_source_root = job_dir
        self.job_finished_at = time.time()
        return entries

    def _on_job_status(self, writer: protocol.FrameWriter) -> None:
        if self.finished is None:
            state, outputs, finished_at = "running", [], None
        else:
            state = str(self.finished.get("state"))
            outputs = self.output_entries
            finished_at = getattr(self, "job_finished_at", None)
        writer.send_json(
            {
                "kind": protocol.KIND_JOB_STATUS_OK,
                "job_id": self.job_id,
                "state": state,
                "progress": 1.0 if self.finished else 0.5,
                "outputs": outputs,
                "created_at": self.job_created_at,
                "started_at": self.job_created_at + 0.01,
                "finished_at": finished_at,
            }
        )

    def _on_output_list(self, writer: protocol.FrameWriter) -> None:
        writer.send_json(
            {
                "kind": protocol.KIND_OUTPUT_LIST_OK,
                "job_id": self.job_id,
                "outputs": self.output_entries,
            }
        )

    def _on_output_chunk(
        self, msg: Dict[str, Any], writer: protocol.FrameWriter
    ) -> None:
        path = str(msg.get("path"))
        offset = int(msg.get("offset") or 0)
        entry = next((e for e in self.output_entries if e["path"] == path), None)
        if entry is None:
            writer.send_json(
                protocol.make_error(protocol.ERR_NOT_FOUND, f"产物不存在: {path}")
            )
            return
        root = self.output_source_root or (self.data_dir / "jobs" / self.job_id)
        source = root / Path(*path.split("/"))
        data = source.read_bytes()
        length = min(OUTPUT_CHUNK_SIZE, len(data) - offset)
        chunk = data[offset : offset + length]
        writer.send_json(
            {
                "kind": protocol.KIND_OUTPUT_CHUNK,
                "job_id": self.job_id,
                "path": path,
                "offset": offset,
                "length": len(chunk),
                "sha256": sha256_bytes(chunk),
                "eof": offset + len(chunk) >= len(data),
            }
        )
        writer.send_blob(chunk)


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------
def build_demo_project(base: Path) -> Path:
    """构造一个带「应忽略文件」的假项目。"""
    project = base / "demo-project"
    project.mkdir(parents=True, exist_ok=True)

    def write(rel: str, data: Any, binary: bool = False) -> None:
        target = project / Path(*rel.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        if binary:
            target.write_bytes(bytes(data))
        else:
            target.write_text(str(data), encoding="utf-8", newline="")

    write(
        "train.py",
        "import json\n\n"
        "print('epoch 1/2 loss=0.9')\n"
        "print('epoch 2/2 loss=0.5')\n"
        "with open('outputs/metrics.json', 'w') as fh:\n"
        "    json.dump({'ok': True}, fh)\n",
    )
    rows = ["x,y"] + [f"{i / 100:.2f},{2 * (i / 100) + 1:.2f}" for i in range(120)]
    write("data.csv", "\n".join(rows) + "\n")
    write("model.json", '{"w": 0.0, "b": 0.0, "runs": 0}\n')
    write("requirements.txt", "# 仅注释：演示环境指纹（无第三方依赖）\n")
    write(FAULT_FILE, bytes(range(256)) + bytes(range(44)), binary=True)

    # 应被默认忽略的内容
    write(".git/config", "[core]\n\trepositoryformatversion = 0\n")
    write(".venv/pyvenv.cfg", "home = /usr/bin\n")
    write("__pycache__/train.cpython-312.pyc", "cache")
    write("outputs/old_model.json", "{}")
    write("debug.log", "should be ignored")
    write("node_modules/pkg/index.js", "console.log(1)")
    return project


def make_config(server: FakeServer, token: Optional[str] = None) -> ClientConfig:
    return ClientConfig(
        host=server.host,
        port=server.port,
        token=token if token is not None else server.token,
        connect_timeout=5.0,
        io_timeout=10.0,
        chunk_size=protocol.DEFAULT_CHUNK_SIZE,
    )


def make_options(
    project: Path, download_dir: Path, *, incremental: bool = True
) -> SubmitOptions:
    return SubmitOptions(
        project_dir=str(project),
        entry="python train.py --epochs 2 --data data.csv",
        incremental=incremental,
        gpu=False,
        timeout_s=60,
        download_dir=str(download_dir),
    )


# ---------------------------------------------------------------------------
# 各阶段
# ---------------------------------------------------------------------------
def test_packager(project: Path) -> None:
    log("[1/7] packager：清单 / 忽略规则 / include-exclude / requirements / meta")
    entries = packager.collect_files(project)
    paths = [entry.path for entry in entries]
    check("train.py" in paths and "data.csv" in paths, "清单包含训练脚本与数据")
    check(all("\\" not in path for path in paths), "路径全部为 POSIX 相对路径")
    for ignored in (".git/config", ".venv/pyvenv.cfg", "__pycache__/train.cpython-312.pyc",
                    "outputs/old_model.json", "debug.log", "node_modules/pkg/index.js"):
        check(ignored not in paths, f"默认忽略 {ignored}")
    entry_map = {entry.path: entry for entry in entries}
    check(entry_map["data.csv"].sha256 == sha256_file(project / "data.csv"),
          "sha256 与实际文件一致")
    check(entry_map["data.csv"].size == (project / "data.csv").stat().st_size,
          "size 与实际文件一致")

    only_py = packager.collect_files(project, include=["*.py"])
    check([e.path for e in only_py] == ["train.py"], "include=['*.py'] 只保留 train.py")
    no_bin = packager.collect_files(project, exclude=["*.bin"])
    check(all(not path.endswith(".bin") for path in (e.path for e in no_bin)),
          "exclude=['*.bin'] 生效")
    no_nested = packager.collect_files(project, exclude=["outputs"])
    check(all(not path.startswith("outputs/") for path in (e.path for e in no_nested)),
          "exclude 目录前缀语义生效")

    requirements, req_path = packager.read_requirements(project)
    check(requirements == [], "requirements.txt 仅注释 → 归一化为空依赖")
    check(req_path is not None and req_path.endswith("requirements.txt"),
          "requirements.txt 被自动识别")

    meta = packager.infer_incremental_meta(project)
    check(
        meta == {
            "mode": "incremental_training",
            "base_model": "model.json",
            "dataset": "data.csv",
        },
        "自动识别增量训练 meta（model.json + data.csv）",
    )

    argv = packager.parse_entry_command('python train.py --epochs 3 --data "data.csv"')
    check(argv == ["python", "train.py", "--epochs", "3", "--data", "data.csv"],
          "入口命令按 argv 解析（不加 shell）")

    spec = packager.make_job_spec(
        project,
        argv,
        gpu=False,
        timeout_s=60,
        meta=None,
    )
    check(len(spec.inputs) == 5, "JobSpec 输入文件数为 5")
    check(len(spec.env.fingerprint) == 64, "env.fingerprint 为 64 位 hex")
    check(spec.meta.get("mode") == "incremental_training", "JobSpec.meta 自动含增量训练语义")
    check(spec.env.fingerprint == spec.env.compute_fingerprint(), "指纹可复算一致")


def test_success_flow(project: Path, work: Path) -> None:
    log("[2/7] 成功链路：上传（HASH_MISMATCH 重传 + 断线重连续传）→ 执行 → 下载产物")
    server = FakeServer(token=SELFTEST_TOKEN, base_dir=work, drop_once=True).start()
    logs: List[str] = []
    events: List[str] = []
    progresses: List[float] = []
    try:
        config = make_config(server)
        result = submit_project(
            config,
            make_options(project, work / "downloads"),
            Callbacks(
                log=logs.append,
                event=lambda _stream, data: events.append(data),
                progress=lambda frac, _text: progresses.append(frac),
            ),
        )
    finally:
        server.stop()

    check(result.ok, f"任务成功（client_exit={result.client_exit}）")
    check(result.client_exit == 0, "成功退出码为 0")
    check(result.state == "succeeded", "state=succeeded")
    check(result.job_id.startswith("job-selftest-"), f"拿到 job_id: {result.job_id}")
    check(server.saw_resume, "断线后携带 resume_job_id 续传（服务端已确认）")
    check(server.hash_fault_triggered, "分片 HASH_MISMATCH 故障被注入且客户端重传成功")
    check(server.connections >= 2, f"发生了重连（连接数 {server.connections} ≥ 2）")
    check(server.completed == set(server.expected), "服务端收到全部输入文件")
    check(
        any("epoch" in data for data in events),
        "客户端收到并输出了流式 job_event 日志",
    )
    check(progresses and max(progresses) >= 0.99, "进度回调推进到 100%")

    downloaded_paths = sorted(item["path"] for item in result.downloaded)
    check(
        downloaded_paths
        == ["outputs/metrics.json", "outputs/model.json", "outputs/sub/note.txt"],
        f"产物清单正确: {downloaded_paths}",
    )
    for record in result.downloaded:
        local = Path(record["local_path"])
        check(local.is_file(), f"产物已落盘: {record['path']}")
        check(sha256_file(local) == record["sha256"], f"产物 sha256 校验通过: {record['path']}")
        server_file = server.data_dir / "jobs" / server.job_id / Path(*record["path"].split("/"))
        check(local.read_bytes() == server_file.read_bytes(),
              f"产物内容与服务端一致: {record['path']}")

    # JobSpec 语义：增量训练 meta 与 requirements/忽略规则
    check(server.spec is not None, "服务端解析了 JobSpec")
    assert server.spec is not None
    check(server.spec.meta.get("mode") == "incremental_training", "服务端看到增量训练 meta")
    check(server.spec.meta.get("base_model") == "model.json", "meta.base_model=model.json")
    check(server.spec.meta.get("dataset") == "data.csv", "meta.dataset=data.csv")
    server_inputs = {
        path.relative_to(server.data_dir / "jobs" / server.job_id / "inputs").as_posix()
        for path in (server.data_dir / "jobs" / server.job_id / "inputs").rglob("*")
        if path.is_file()
    }
    check(server_inputs == {entry.path for entry in server.spec.inputs},
          "服务端落盘文件与客户端清单完全一致")
    check("debug.log" not in server_inputs and ".git/config" not in server_inputs,
          "被忽略的文件没有上传到服务端")

    log("  [INFO] 客户端日志摘要:")
    for line in logs:
        if "重连" in line or "续传" in line or "任务已接受" in line or "任务成功" in line:
            log(f"         {line}")


def test_failure_flow(project: Path, work: Path) -> None:
    log("[3/7] 失败链路：远端任务非零退出 → 客户端退出码 1")
    server = FakeServer(token=SELFTEST_TOKEN, base_dir=work, fail_job=True, fail_exit_code=7).start()
    try:
        result = submit_project(
            make_config(server),
            make_options(project, work / "downloads-failed"),
            Callbacks(),
        )
    finally:
        server.stop()
    check(not result.ok, "任务失败被识别")
    check(result.client_exit == 1, "任务失败退出码为 1")
    check(result.state == "failed", "state=failed")
    check(result.job_exit_code == 7, "保留远端退出码 7")
    check((result.error or {}).get("code") == "JOB_FAILED", "error.code=JOB_FAILED")
    check(any("模拟训练失败" in line for line in result.log_tail), "log_tail 带回错误信息")


def test_auth_failure(project: Path, work: Path) -> None:
    log("[4/7] 鉴权失败：错误 token → 退出码 4")
    server = FakeServer(token="correct-token", base_dir=work).start()
    try:
        result = submit_project(
            make_config(server, token="wrong-token"),
            make_options(project, work / "downloads-auth"),
            Callbacks(),
        )
    finally:
        server.stop()
    check(not result.ok, "鉴权失败被识别")
    check(result.client_exit == 4, "鉴权失败退出码为 4")
    check((result.error or {}).get("code") == protocol.ERR_BAD_AUTH, "error.code=BAD_AUTH")


def test_connection_refused(project: Path, work: Path) -> None:
    log("[5/7] 连接失败：端口未监听 → 退出码 3")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
    probe.close()
    config = ClientConfig(
        host="127.0.0.1", port=free_port, token="x", connect_timeout=1.0, io_timeout=2.0
    )
    result = submit_project(
        config, make_options(project, work / "downloads-refused"), Callbacks()
    )
    check(not result.ok, "连接失败被识别")
    check(result.client_exit == 3, "连接失败退出码为 3")


def test_cli_contract(project: Path, work: Path) -> None:
    log("[6/7] CLI 契约：submit --json / status / config --save")
    server = FakeServer(token="cli-token", base_dir=work).start()
    try:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = client_main.main(
                [
                    "submit",
                    str(project),
                    "--entry",
                    "python train.py --epochs 1 --data data.csv",
                    "--server",
                    f"{server.host}:{server.port}",
                    "--token",
                    "cli-token",
                    "--json",
                    "--timeout",
                    "60",
                    "--download-dir",
                    str(work / "downloads-cli"),
                ]
            )
        check(code == 0, "submit --json 退出码为 0")
        stdout_text = out.getvalue().strip()
        payload = json.loads(stdout_text.splitlines()[-1])
        check(payload["ok"] is True and payload["client_exit"] == 0, "JSON 结果 ok/client_exit 正确")
        check(payload["state"] == "succeeded", "JSON 结果 state=succeeded")
        check(len(payload["outputs"]) == 3, "JSON 结果包含 3 个产物")
        check("epoch 1/3" in err.getvalue(), "远端日志走 stderr（不污染 stdout JSON）")

        out2 = io.StringIO()
        with redirect_stdout(out2):
            status_code = client_main.main(
                [
                    "status",
                    payload["job_id"],
                    "--server",
                    f"{server.host}:{server.port}",
                    "--token",
                    "cli-token",
                    "--json",
                ]
            )
        status_payload = json.loads(out2.getvalue().strip().splitlines()[-1])
        check(status_code == 0, "status 退出码为 0")
        check(status_payload["state"] == "succeeded", "status 返回 succeeded")
    finally:
        server.stop()

    # 参数错误 → 2
    out3 = io.StringIO()
    with redirect_stdout(out3):
        bad_code = client_main.main(
            ["submit", str(project), "--entry", "", "--server", "127.0.0.1:1", "--token", "t", "--json"]
        )
    check(bad_code == 2, "空 entry 参数错误退出码为 2")
    check(json.loads(out3.getvalue().strip().splitlines()[-1])["client_exit"] == 2,
          "JSON 参数错误 client_exit=2")

    # config --save
    config_path = work / "client-config.json"
    code = client_main.main(
        [
            "config",
            "--server",
            "10.0.0.9:9999",
            "--token",
            "saved-token",
            "--config",
            str(config_path),
            "--save",
        ]
    )
    check(code == 0, "config --save 退出码为 0")
    check(config_path.is_file(), "配置文件已写入")
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    check(saved["host"] == "10.0.0.9" and saved["port"] == 9999, "服务器地址已保存")
    check(saved["token"] == "saved-token", "token 已保存")


def test_real_example_execution(work: Path) -> None:
    """真实执行 ``examples/incremental_train``：假服务端跑真训练脚本 + 回传真产物。"""
    log("[7/7] 真实示例：假服务端执行 train.py → 流式 epoch 日志 → 回传真实产物")
    example = _PROJECT_ROOT / "examples" / "incremental_train"
    check(example.is_dir(), f"示例项目存在: {example}")
    project = work / "example-project"
    if project.exists():
        cleanup_dir(project)
    shutil.copytree(
        example,
        project,
        ignore=shutil.ignore_patterns("__pycache__", "outputs", "*.pyc", ".git"),
    )
    check((project / "train.py").is_file(), "示例项目含 train.py")
    check((project / "model.json").is_file(), "示例项目含基座模型 model.json")
    base_model = json.loads((project / "model.json").read_text(encoding="utf-8"))
    check(int(base_model.get("runs", 0)) == 0, "示例基座模型初始 runs=0")

    server = FakeServer(token=SELFTEST_TOKEN, base_dir=work, real_exec=True).start()
    events: List[str] = []
    logs: List[str] = []
    try:
        options = SubmitOptions(
            project_dir=str(project),
            entry="python train.py --epochs 2 --data data.csv --out outputs --model model.json",
            incremental=True,
            timeout_s=120,
            download_dir=str(work / "example-downloads"),
        )
        result = submit_project(
            make_config(server),
            options,
            Callbacks(log=logs.append, event=lambda _stream, data: events.append(data)),
        )
    finally:
        server.stop()

    check(result.ok and result.client_exit == 0, f"真实训练任务成功（client_exit={result.client_exit}）")
    check(result.state == "succeeded", "state=succeeded")
    output_paths = sorted(item["path"] for item in result.downloaded)
    check(
        output_paths == ["outputs/metrics.json", "outputs/model.json"],
        f"回传产物清单正确: {output_paths}",
    )
    joined_events = "".join(events)
    check("epoch 1/2" in joined_events, "客户端收到真实 epoch 1/2 日志")
    check("epoch 2/2" in joined_events, "客户端收到真实 epoch 2/2 日志")
    check("加载基座模型" in joined_events, "远端 train.py 加载了基座模型（增量训练）")

    download_dir = Path(result.download_dir)
    model = json.loads((download_dir / "outputs" / "model.json").read_text(encoding="utf-8"))
    metrics = json.loads((download_dir / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    check(int(model["runs"]) == 1, f"远端产物 model.json runs=1（实际 {model['runs']}）")
    check(int(model["epochs_total"]) == 2, "远端产物 epochs_total=2")
    check(bool(metrics["base_model_loaded"]), "metrics.base_model_loaded=true")
    check(
        float(metrics["final_loss"]) < float(metrics["start_loss"]),
        f"loss 下降: {metrics['start_loss']} → {metrics['final_loss']}",
    )
    local_model = json.loads((project / "model.json").read_text(encoding="utf-8"))
    check(int(local_model["runs"]) == 0, "远端执行不会修改本地项目文件（runs 仍为 0）")
    for record in result.downloaded:
        local = Path(record["local_path"])
        check(
            sha256_file(local) == record["sha256"] and local.stat().st_size == record["size"],
            f"真实产物 sha256/size 校验通过: {record['path']}",
        )


def main() -> int:
    started = time.time()
    log("=" * 72)
    log("RGPU 客户端自测 (client/selftest.py) — 本地假服务端，不依赖 server/")
    log("=" * 72)
    failures: List[str] = []
    work = make_temp_dir(Path(__file__).resolve().parent, "selftest-work")
    log(f"临时工作目录: {work}")
    try:
        project = build_demo_project(work)
        stages = [
            ("packager", lambda: test_packager(project)),
            ("success-flow", lambda: test_success_flow(project, work)),
            ("failure-flow", lambda: test_failure_flow(project, work)),
            ("auth-failure", lambda: test_auth_failure(project, work)),
            ("connection-refused", lambda: test_connection_refused(project, work)),
            ("cli-contract", lambda: test_cli_contract(project, work)),
            ("real-example", lambda: test_real_example_execution(work)),
        ]
        for name, fn in stages:
            try:
                fn()
                log(f"[OK] {name}")
            except Exception as exc:  # noqa: BLE001
                failures.append(name)
                log(f"[FAIL] {name}: {exc}")
                traceback.print_exc()
    finally:
        cleanup_dir(work)
    elapsed = time.time() - started
    log("-" * 72)
    if failures:
        log(f"自测失败: {len(failures)} 个阶段失败 {failures}（用时 {elapsed:.2f}s）")
        return 1
    log(f"自测全部通过（7/7 阶段，用时 {elapsed:.2f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
