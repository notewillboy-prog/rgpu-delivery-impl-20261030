"""任务运行：提交 → 上传 → 启动 → 流式日志 → 下载产物 → 退出码映射。

对应 ``docs/PROTOCOL.md`` 3.2–3.6 与第 4、6 节。运行结果用 ``RunResult`` 表示，
CLI 退出码为 ``client_exit``：

- 0 = 任务成功（产物已下载并校验）
- 1 = 任务失败 / 超时（远端退出码非 0，state=failed/timeout/cancelled）
- 2 = 参数或配置错误（打包、入口命令、本地文件变化等）
- 3 = 连接 / 网络错误（含重连失败）
- 4 = 服务端拒绝（鉴权、协议、环境构建失败、哈希校验失败等）
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from common import protocol
from common.config import ClientConfig
from common.hashing import human_size
from common.models import FileEntry, JobSpec, OutputEntry

from . import packager
from .connection import ClientError, Connection, ConnectionLost, RemoteError
from .durable_receipt import DurableReceipt, ReceiptConflict
from .uploader import upload_files

CLIENT_VERSION = "0.1.0"

EXIT_OK = 0
EXIT_JOB_FAILED = 1
EXIT_USAGE = 2
EXIT_CONNECTION = 3
EXIT_REJECTED = 4

DEFAULT_DOWNLOAD_DIR = "rgpu-outputs"
MAX_DOWNLOAD_CHUNK_RETRIES = 3
MAX_DOWNLOAD_RECONNECTS = 2

_TERMINAL_STATES = {
    "succeeded",
    "failed",
    "timeout",
    "cancelled",
}


# ---------------------------------------------------------------------------
# 回调与选项
# ---------------------------------------------------------------------------
@dataclass
class Callbacks:
    """运行期回调；CLI 打到终端，GUI 通过 queue 回主线程。"""

    log: Callable[[str], None] = lambda _msg: None
    #: 远端日志流：``(stream, data)``
    event: Callable[[str, str], None] = lambda _stream, _data: None
    #: 总体进度：``(fraction 0..1, text)``
    progress: Callable[[float, str], None] = lambda _frac, _text: None
    #: 作业 id 就绪
    job_id: Callable[[str], None] = lambda _job_id: None


@dataclass
class SubmitOptions:
    """``client.main submit`` / GUI 的输入参数。"""

    project_dir: str
    entry: str
    job_name: str = ""
    requirements: Optional[str] = None
    env_mode: str = "venv"
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=lambda: ["outputs/**"])
    gpu: bool = False
    gpu_count: int = 1
    timeout_s: int = 600
    download_dir: str = DEFAULT_DOWNLOAD_DIR
    watch: bool = True
    #: None=自动识别增量训练 meta；True=强制增量 meta；False=不填 meta
    incremental: Optional[bool] = None
    python_version: Optional[str] = None
    chunk_size: Optional[int] = None
    receipt_dir: Optional[str] = None
    receipt_request_id: Optional[str] = None


@dataclass
class RunResult:
    """一次任务提交的最终结果。"""

    ok: bool
    job_id: str = ""
    state: str = ""
    job_exit_code: int = -1
    client_exit: int = EXIT_OK
    duration_s: float = 0.0
    downloaded: List[Dict[str, Any]] = field(default_factory=list)
    download_dir: str = ""
    env: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, str]] = None
    log_tail: List[str] = field(default_factory=list)
    spec_summary: Dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "ok": bool(self.ok),
            "job_id": self.job_id,
            "state": self.state,
            "exit_code": int(self.job_exit_code),
            "client_exit": int(self.client_exit),
            "duration_s": round(float(self.duration_s), 3),
            "download_dir": self.download_dir,
            "outputs": list(self.downloaded),
            "env": dict(self.env),
        }
        if self.error:
            data["error"] = dict(self.error)
        if self.log_tail:
            data["log_tail"] = list(self.log_tail)
        if self.spec_summary:
            data["spec"] = dict(self.spec_summary)
        return data


def _now() -> float:
    return time.time()


def _error_result(
    client_exit: int,
    code: str,
    message: str,
    started_at: float,
    *,
    job_id: str = "",
    state: str = "error",
    spec_summary: Optional[Dict[str, Any]] = None,
) -> RunResult:
    finished = _now()
    return RunResult(
        ok=False,
        job_id=job_id,
        state=state,
        job_exit_code=-1,
        client_exit=int(client_exit),
        duration_s=finished - started_at,
        error={"code": str(code), "message": str(message)},
        spec_summary=spec_summary or {},
        started_at=started_at,
        finished_at=finished,
    )


# ---------------------------------------------------------------------------
# JobRunner
# ---------------------------------------------------------------------------
class JobRunner:
    """单次任务会话（一个 Connection 上完成提交、上传、执行、下载）。"""

    def __init__(
        self,
        conn: Connection,
        spec: JobSpec,
        project_dir: str,
        *,
        chunk_size: Optional[int] = None,
        callbacks: Optional[Callbacks] = None,
        receipt_dir: Optional[str] = None,
        receipt_request_id: Optional[str] = None,
    ):
        self.conn = conn
        self.spec = spec
        self.project_dir = str(project_dir)
        self.callbacks = callbacks or Callbacks()
        self.chunk_size = int(chunk_size or self.conn.config.chunk_size)
        self.job_id = ""
        self.accepted: Dict[str, Any] = {}
        self.env: Dict[str, Any] = {}
        self.receipt: Optional[DurableReceipt] = None
        self.receipt_request_id = ""
        if receipt_dir:
            request_id = receipt_request_id or self._derived_request_id()
            self.receipt = DurableReceipt.for_request(receipt_dir, request_id)
            self.receipt_request_id = request_id

    def _derived_request_id(self) -> str:
        import json
        payload = json.dumps(self.spec.to_dict(), ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:32]

    def _receipt_endpoint(self) -> str:
        return f"tcp://{self.conn.config.host}:{int(self.conn.config.port)}"

    def _receipt_spec_sha(self) -> str:
        import json
        payload = json.dumps(self.spec.to_dict(), ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _receipt_terminal(self, result: RunResult) -> None:
        if self.receipt is None or not self.job_id:
            return
        if result.state not in ("succeeded", "failed", "timeout", "cancelled"):
            return
        if result.ok:
            state = "succeeded"
        elif result.state == "cancelled":
            state = "cancelled"
        elif result.state in ("failed", "timeout"):
            state = "failed"
        else:
            return
        try:
            self.receipt.update_state(self.job_id, state)
        except (ValueError, OSError):
            self.callbacks.log(f"本地投递账本更新失败（不回滚远端任务）: {state}")

    # -- 对外入口 ----------------------------------------------------------
    def run(self, download_dir: str = DEFAULT_DOWNLOAD_DIR, watch: bool = True) -> RunResult:
        started_at = _now()
        try:
            if self.receipt is not None:
                try:
                    allowed = self.receipt.begin_once(
                        self.receipt_request_id,
                        self._receipt_endpoint(),
                        self._receipt_spec_sha(),
                    )
                except (ValueError, OSError, ReceiptConflict) as exc:
                    return _error_result(
                        EXIT_REJECTED, protocol.ERR_CONFLICT,
                        f"本地投递账本拒绝本次提交（禁止自动重投）: {exc}", started_at,
                    )
                if not allowed:
                    return _error_result(
                        EXIT_REJECTED, protocol.ERR_CONFLICT,
                        "本地投递账本已存在同一请求；请用 query/status 核查，禁止自动重投",
                        started_at,
                    )
            self.callbacks.progress(0.08, "提交任务…")
            accepted = self._submit_with_busy_retry()
            self.job_id = str(accepted.get("job_id") or "")
            if not self.job_id:
                raise RemoteError(protocol.ERR_BAD_REQUEST, "job_accepted 缺少 job_id")
            if self.receipt is not None:
                self.receipt.record_accepted(self.job_id)
            self.accepted = accepted
            self.callbacks.job_id(self.job_id)
            self._log_accepted(accepted)

            self._upload_with_resume(accepted)
            if self.receipt is not None:
                try:
                    self.receipt.update_state(self.job_id, "tracking")
                except (ValueError, OSError):
                    pass

            if not watch:
                finished = self._start_job(wait_finished=False)
                if finished is None:
                    result = RunResult(
                        ok=True,
                        job_id=self.job_id,
                        state="started",
                        job_exit_code=-1,
                        client_exit=EXIT_OK,
                        duration_s=_now() - started_at,
                        env=dict(self.env),
                        spec_summary=self._spec_summary(),
                        started_at=started_at,
                        finished_at=_now(),
                    )
                    self._receipt_terminal(result)
                    return result
                result = self._result_from_finished(
                    finished, started_at, download_dir=download_dir, do_download=False
                )
                self._receipt_terminal(result)
                return result

            try:
                finished = self._start_job(wait_finished=True)
            except ConnectionLost:
                self.callbacks.log("执行期间连接中断，正在重连并继续跟踪作业…")
                finished = self._recover_and_follow()
            result = self._result_from_finished(
                finished, started_at, download_dir=download_dir, do_download=True
            )
            self._receipt_terminal(result)
            return result
        except ClientError as exc:
            self.callbacks.progress(0.0, "失败")
            result = _error_result(
                exc.exit_code,
                exc.code,
                str(exc),
                started_at,
                job_id=self.job_id,
                spec_summary=self._spec_summary(),
            )
            self._receipt_terminal(result)
            return result

    # -- 提交 --------------------------------------------------------------
    def _submit_with_busy_retry(self) -> Dict[str, Any]:
        last: Optional[RemoteError] = None
        for attempt in range(4):
            try:
                return self._submit(resume_job_id=None)
            except RemoteError as exc:
                if exc.code == protocol.ERR_BUSY and attempt < 3:
                    last = exc
                    delay = 1.0 + attempt
                    self.callbacks.log(f"服务端繁忙，{delay:.0f}s 后重试（{attempt + 1}/3）…")
                    time.sleep(delay)
                    continue
                raise
        raise last or RemoteError(protocol.ERR_BUSY, "服务端繁忙")

    def _submit(self, resume_job_id: Optional[str]) -> Dict[str, Any]:
        self.conn.send_json(
            {
                "kind": protocol.KIND_JOB_SUBMIT,
                "spec": self.spec.to_dict(),
                "resume_job_id": resume_job_id,
            }
        )
        resp = self.conn.wait_for([protocol.KIND_JOB_ACCEPTED], on_message=self._handle_passive)
        job_id = resp.get("job_id")
        if job_id:
            self.job_id = str(job_id)
        return resp

    def _log_accepted(self, accepted: Dict[str, Any]) -> None:
        missing = accepted.get("missing") or []
        present = accepted.get("present") or []
        env = accepted.get("env") or {}
        fp = str(env.get("fingerprint") or self.spec.env.fingerprint)
        self.callbacks.log(
            f"任务已接受 job_id={self.job_id}  job_name={self.spec.job_name}"
        )
        self.callbacks.log(
            f"环境指纹 {fp[:16]}…（mode={self.spec.env.mode}, "
            f"python={self.spec.env.python_version}, requirements={len(self.spec.env.requirements)} 条）"
        )
        self.callbacks.log(
            f"输入文件 {len(self.spec.inputs)} 个，共 {human_size(self.spec.total_input_bytes())}；"
            f"缺失 {len(missing)}，服务端已有 {len(present)}"
        )
        if present:
            self.callbacks.log("  跳过（服务端已有）: " + ", ".join(str(p) for p in present[:10]))

    # -- 上传（含断线续传） -------------------------------------------------
    def _missing_entries(self, accepted: Dict[str, Any]) -> List[FileEntry]:
        index = self.spec.input_index()
        raw = accepted.get("missing")
        if raw is None:
            raw = list(index)
        paths: List[str] = []
        for item in raw:
            # 协议 3.2 里 missing 是 FileEntry 对象列表；兼容只给路径字符串的实现
            if isinstance(item, dict):
                path = item.get("path")
            else:
                path = item
            if path is not None:
                paths.append(str(path))
        unknown = [p for p in paths if p not in index]
        if unknown:
            raise RemoteError(
                protocol.ERR_BAD_REQUEST,
                f"服务端要求的文件不在本次清单中: {unknown[:5]}",
            )
        return [index[p] for p in paths]

    def _effective_chunk_size(self, accepted: Dict[str, Any]) -> int:
        try:
            chunk = int(accepted.get("chunk_size") or self.chunk_size)
        except (TypeError, ValueError):
            chunk = protocol.DEFAULT_CHUNK_SIZE
        chunk = max(1, min(chunk, protocol.MAX_CHUNK_SIZE))
        if self.conn.max_frame:
            chunk = min(chunk, self.conn.max_frame)
        return chunk

    def _received_offsets(self, accepted: Dict[str, Any]) -> Dict[str, int]:
        raw = accepted.get("received")
        if not isinstance(raw, dict):
            return {}
        index = self.spec.input_index()
        result: Dict[str, int] = {}
        for path, value in raw.items():
            entry = index.get(str(path))
            if entry is None:
                continue
            try:
                offset = int(value)
            except (TypeError, ValueError):
                continue
            if 0 < offset < entry.size:
                result[entry.path] = offset
        return result

    def _upload_with_resume(self, accepted: Dict[str, Any]) -> None:
        while True:
            missing = self._missing_entries(accepted)
            if not missing:
                self.callbacks.progress(0.55, "输入文件已就绪")
                self.callbacks.log("所有输入文件已在服务端就绪，跳过上传")
                return
            total_bytes = sum(e.size for e in missing)
            chunk = self._effective_chunk_size(accepted)
            received = self._received_offsets(accepted)
            self.callbacks.log(
                f"开始上传 {len(missing)} 个文件，共 {human_size(total_bytes)}"
                f"（分片 {chunk} B）"
                + (f"，续传 {len(received)} 个" if received else "")
            )

            def on_progress(done: int, total: int) -> None:
                ratio = (done / total) if total else 1.0
                ratio = max(0.0, min(1.0, ratio))
                self.callbacks.progress(0.15 + 0.40 * ratio, f"上传 {ratio * 100:.0f}%")

            def on_file_start(entry: FileEntry) -> None:
                self.callbacks.log(f"  ↑ {entry.path} ({human_size(entry.size)})")

            try:
                upload_files(
                    self.conn,
                    missing,
                    self.project_dir,
                    chunk_size=chunk,
                    on_progress=on_progress,
                    on_file_start=on_file_start,
                    start_offsets=received,
                )
                self.callbacks.progress(0.55, "上传完成")
                return
            except ConnectionLost:
                self.callbacks.log("上传中断，正在重连并使用 resume_job_id 续传…")
                self._recover_connection()
                accepted = self._resubmit_for_resume()
                continue

    def _resubmit_for_resume(self) -> Dict[str, Any]:
        original_job_id = self.job_id
        if not original_job_id:
            raise RemoteError(protocol.ERR_CONFLICT, '原任务 ID 未知，禁止自动重新提交；请人工核查服务端。')
        try:
            accepted = self._submit(resume_job_id=original_job_id)
            if self.job_id != original_job_id:
                self.job_id = original_job_id
                raise RemoteError(protocol.ERR_CONFLICT, '服务端返回不同的任务 ID；禁止重投，请人工核查。')
            self.callbacks.log(f"已续接作业 {self.job_id}")
            self._log_accepted(accepted)
            return accepted
        except RemoteError as exc:
            if exc.code in (protocol.ERR_NOT_FOUND, protocol.ERR_CONFLICT):
                self.job_id = original_job_id
                raise RemoteError(exc.code, f'服务端无法续接原任务 {original_job_id}（{exc.code}）；禁止自动新建，请人工核查。') from exc
            raise

    # -- 启动与执行 --------------------------------------------------------
    def _start_job(self, wait_finished: bool) -> Optional[Dict[str, Any]]:
        self.callbacks.progress(0.58, "启动远端作业…")
        self.conn.send_json({"kind": protocol.KIND_JOB_START, "job_id": self.job_id})
        self.callbacks.progress(0.60, "等待远端环境就绪…")
        first = self.conn.wait_for(
            [protocol.KIND_JOB_STARTED, protocol.KIND_JOB_FINISHED],
            on_message=self._handle_passive,
        )
        if first.get("kind") == protocol.KIND_JOB_FINISHED:
            return first
        env = first.get("env") or {}
        self.env = dict(env) if isinstance(env, dict) else {}
        state = self.env.get("state") or "ready"
        self.callbacks.log(f"远端环境就绪（{state}），作业开始执行")
        self.callbacks.progress(0.62, "执行中…")
        if not wait_finished:
            return None
        finished = self.conn.wait_for(
            [protocol.KIND_JOB_FINISHED], on_message=self._handle_passive
        )
        return finished

    def _handle_passive(self, msg: Dict[str, Any]) -> None:
        """执行期间的异步消息（日志流 / 环境进度）。"""
        kind = msg.get("kind")
        if kind == protocol.KIND_JOB_EVENT:
            stream = str(msg.get("stream") or "stdout")
            data = msg.get("data") or ""
            if data:
                self.callbacks.event(stream, str(data))
            return
        if kind == protocol.KIND_JOB_PROGRESS:
            state = str(msg.get("state") or "")
            message = str(msg.get("message") or "")
            try:
                progress = float(msg.get("progress") or 0.0)
            except (TypeError, ValueError):
                progress = 0.0
            progress = max(0.0, min(1.0, progress))
            if message:
                self.callbacks.log(f"[{state}] {message}")
            self.callbacks.progress(0.62 + 0.20 * progress, message or state)
            return

    def _recover_connection(self) -> None:
        info = self.conn.reconnect()
        self.callbacks.log(
            f"已重连 {self.conn.config.address}"
            f"（server_id={info.get('server_id')}, hostname={info.get('hostname')}）"
        )

    def _recover_and_follow(self) -> Dict[str, Any]:
        """连接在执行期断开：重连 → 重新附着 → 用 job_status 跟踪到结束。"""
        self._recover_connection()
        self.callbacks.log("正在重新附着远端作业…")
        try:
            accepted = self._submit(resume_job_id=self.job_id)
            self.callbacks.log(f"已重新附着作业 {self.job_id}")
            self._upload_with_resume(accepted)
        except RemoteError as exc:
            if exc.code in (protocol.ERR_NOT_FOUND, protocol.ERR_CONFLICT):
                self.callbacks.log(
                    f"重新附着返回 {exc.code}，改用 job_status 继续跟踪"
                )
            else:
                raise
        status = self._query_status()
        state = str(status.get("state") or "")
        if state in ("queued", "uploading", "env_building"):
            try:
                self.conn.send_json({"kind": protocol.KIND_JOB_START, "job_id": self.job_id})
            except RemoteError as exc:
                if exc.code != protocol.ERR_CONFLICT:
                    raise
        while state not in _TERMINAL_STATES:
            self.callbacks.progress(0.65, f"作业状态: {state or 'unknown'}…")
            time.sleep(2.0)
            status = self._query_status()
            state = str(status.get("state") or "")
        return self._finished_from_status(status)

    def _finished_from_status(self, status: Dict[str, Any]) -> Dict[str, Any]:
        state = str(status.get("state") or "failed")
        duration = 0.0
        try:
            started = float(status.get("started_at") or 0.0)
            finished_at = float(status.get("finished_at") or 0.0)
            if started and finished_at:
                duration = max(0.0, finished_at - started)
        except (TypeError, ValueError):
            duration = 0.0
        return {
            "kind": protocol.KIND_JOB_FINISHED,
            "job_id": self.job_id,
            "state": state,
            "exit_code": 0 if state == "succeeded" else -1,
            "duration_s": duration,
            "outputs": status.get("outputs") or [],
            "log_tail": [],
        }

    def _query_status(self) -> Dict[str, Any]:
        self.conn.send_json({"kind": protocol.KIND_JOB_STATUS, "job_id": self.job_id})
        return self.conn.wait_for(
            [protocol.KIND_JOB_STATUS_OK], on_message=self._handle_passive
        )

    # -- 产物下载 ----------------------------------------------------------
    def _output_entries(self) -> List[OutputEntry]:
        self.conn.send_json({"kind": protocol.KIND_OUTPUT_LIST, "job_id": self.job_id})
        resp = self.conn.wait_for(
            [protocol.KIND_OUTPUT_LIST_OK], on_message=self._handle_passive
        )
        entries: List[OutputEntry] = []
        for item in resp.get("outputs") or []:
            if not isinstance(item, dict):
                continue
            try:
                entry = OutputEntry.from_dict(item)
            except (KeyError, TypeError, ValueError):
                continue
            if entry.kind != "file":
                continue
            entries.append(entry)
        return entries

    def _download_outputs(
        self, outputs: Sequence[Dict[str, Any]], download_dir: str
    ) -> List[Dict[str, Any]]:
        target_root = Path(download_dir).expanduser()
        if not target_root.is_absolute():
            target_root = Path.cwd() / target_root
        target_root = target_root.resolve()
        target_root.mkdir(parents=True, exist_ok=True)

        for attempt in range(MAX_DOWNLOAD_RECONNECTS + 1):
            try:
                entries = self._output_entries()
                break
            except ConnectionLost:
                if attempt >= MAX_DOWNLOAD_RECONNECTS:
                    raise
                self.callbacks.log("列产物时连接中断，正在重连…")
                self._recover_connection()
        else:  # pragma: no cover - 循环必然 break 或 raise
            entries = []

        if not entries:
            self.callbacks.log("任务没有产物需要下载")
            self.callbacks.progress(0.98, "无产物")
            return []

        total = sum(e.size for e in entries) or 1
        done = [0]
        downloaded: List[Dict[str, Any]] = []

        def on_bytes(count: int) -> None:
            done[0] += count
            ratio = min(1.0, done[0] / total)
            self.callbacks.progress(0.85 + 0.13 * ratio, f"下载产物 {ratio * 100:.0f}%")

        for entry in entries:
            for attempt in range(MAX_DOWNLOAD_RECONNECTS + 1):
                try:
                    record = self._download_one(entry, target_root, on_bytes)
                    break
                except ConnectionLost:
                    if attempt >= MAX_DOWNLOAD_RECONNECTS:
                        raise
                    self.callbacks.log(f"下载 {entry.path} 时连接中断，正在重连…")
                    self._recover_connection()
            downloaded.append(record)
            self.callbacks.log(
                f"  ↓ {entry.path} ({human_size(entry.size)}) sha256 校验通过"
            )
        self.callbacks.progress(0.99, "产物下载完成")
        return downloaded

    def _download_one(
        self,
        entry: OutputEntry,
        target_root: Path,
        on_bytes: Callable[[int], None],
    ) -> Dict[str, Any]:
        dest = _safe_output_path(target_root, entry.path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        restarted = False
        while True:
            offset = 0
            try:
                if tmp.is_file():
                    offset = int(tmp.stat().st_size)
            except OSError:
                offset = 0
            if offset > entry.size:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                offset = 0
            digest = hashlib.sha256()
            if offset:
                try:
                    with open(tmp, "rb") as prefix:
                        remaining = offset
                        while remaining > 0:
                            block = prefix.read(min(1024 * 1024, remaining))
                            if not block:
                                raise OSError("short .part prefix")
                            digest.update(block)
                            remaining -= len(block)
                except OSError:
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
                    offset = 0
                    digest = hashlib.sha256()
            try:
                with open(tmp, "ab" if offset else "wb") as fh:
                    while offset < entry.size:
                        header = {
                            "kind": protocol.KIND_OUTPUT_CHUNK,
                            "job_id": self.job_id,
                            "path": entry.path,
                            "offset": offset,
                        }
                        blob = b""
                        resp: Dict[str, Any] = {}
                        for attempt in range(MAX_DOWNLOAD_CHUNK_RETRIES + 1):
                            self.conn.send_json(header)
                            resp = self.conn.wait_for(
                                [protocol.KIND_OUTPUT_CHUNK], on_message=self._handle_passive
                            )
                            try:
                                length = int(resp.get("length", 0))
                            except (TypeError, ValueError):
                                length = 0
                            chunk_sha = str(resp.get("sha256") or "")
                            blob = self.conn.recv_blob()
                            if length == 0:
                                raise RemoteError(
                                    protocol.ERR_INTERNAL,
                                    f"产物分片长度为 0: {entry.path} offset={offset}",
                                )
                            if len(blob) != length:
                                raise RemoteError(
                                    protocol.ERR_INTERNAL,
                                    f"产物分片长度不符: 头 {length}，实际 {len(blob)}",
                                )
                            if chunk_sha and hashlib.sha256(blob).hexdigest() != chunk_sha:
                                if attempt < MAX_DOWNLOAD_CHUNK_RETRIES:
                                    self.callbacks.log(
                                        f"产物分片 sha256 不匹配，重试 offset={offset}"
                                    )
                                    continue
                                raise RemoteError(
                                    protocol.ERR_HASH_MISMATCH,
                                    f"产物分片 sha256 校验失败: {entry.path} offset={offset}",
                                )
                            break
                        fh.write(blob)
                        digest.update(blob)
                        offset += len(blob)
                        on_bytes(len(blob))
                        if resp.get("eof") and offset >= entry.size:
                            break
                if offset != entry.size:
                    raise RemoteError(
                        protocol.ERR_INTERNAL,
                        f"产物大小不符: {entry.path} 期望 {entry.size}，实际 {offset}",
                    )
                actual_sha = digest.hexdigest()
                if entry.sha256 and actual_sha != entry.sha256:
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
                    if not restarted:
                        restarted = True
                        self.callbacks.log(
                            f"产物 {entry.path} 断点续传后 sha256 不匹配，从 0 重下一次"
                        )
                        continue
                    raise RemoteError(
                        protocol.ERR_HASH_MISMATCH,
                        f"产物整文件 sha256 校验失败: {entry.path}",
                    )
                os.replace(tmp, dest)
            except ConnectionLost:
                # 保留 .part：重连后从当前大小继续。
                raise
            except BaseException:
                # 本地/协议错误：不保留可能损坏的半成品；断线专门走上面的分支。
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
                raise
            return {
                "path": entry.path,
                "local_path": str(dest),
                "size": entry.size,
                "sha256": entry.sha256 or actual_sha,
                "verified": True,
            }

    # -- 收尾 --------------------------------------------------------------
    def _result_from_finished(
        self,
        finished: Dict[str, Any],
        started_at: float,
        *,
        download_dir: str,
        do_download: bool,
    ) -> RunResult:
        state = str(finished.get("state") or "failed")
        try:
            job_exit = int(finished.get("exit_code", -1))
        except (TypeError, ValueError):
            job_exit = -1
        duration = finished.get("duration_s")
        finished_at = _now()
        try:
            duration_s = float(duration) if duration is not None else finished_at - started_at
        except (TypeError, ValueError):
            duration_s = finished_at - started_at
        log_tail = [str(x) for x in (finished.get("log_tail") or [])]
        error: Optional[Dict[str, str]] = None
        downloaded: List[Dict[str, Any]] = []

        if state == "succeeded":
            if do_download:
                try:
                    downloaded = self._download_outputs(
                        finished.get("outputs") or [], download_dir
                    )
                except ClientError as exc:
                    return _error_result(
                        exc.exit_code,
                        exc.code,
                        str(exc),
                        started_at,
                        job_id=self.job_id,
                        state="error",
                        spec_summary=self._spec_summary(),
                    )
            target = Path(download_dir).expanduser()
            target_str = str(target if target.is_absolute() else Path.cwd() / target)
            self.callbacks.log(
                f"任务成功（duration {duration_s:.1f}s），产物 {len(downloaded)} 个 → {target_str}"
            )
            self.callbacks.progress(1.0, "完成")
            return RunResult(
                ok=True,
                job_id=self.job_id,
                state=state,
                job_exit_code=0,
                client_exit=EXIT_OK,
                duration_s=duration_s,
                downloaded=downloaded,
                download_dir=target_str,
                env=dict(self.env),
                error=None,
                log_tail=log_tail,
                spec_summary=self._spec_summary(),
                started_at=started_at,
                finished_at=finished_at,
            )

        if state == "timeout":
            code, message = protocol.ERR_TIMEOUT, "远端任务超时"
            client_exit = EXIT_JOB_FAILED
        elif state == "cancelled":
            code, message = "CANCELLED", "远端任务已取消"
            client_exit = EXIT_JOB_FAILED
        elif state == "failed":
            code = "JOB_FAILED"
            message = f"远端任务失败（exit_code={job_exit}）"
            client_exit = EXIT_JOB_FAILED
        else:
            code = protocol.ERR_INTERNAL
            message = f"未知的作业状态: {state}"
            client_exit = EXIT_REJECTED
        error = {"code": code, "message": message}
        self.callbacks.log(f"任务结束 state={state} exit_code={job_exit}：{message}")
        self.callbacks.progress(1.0, state)
        return RunResult(
            ok=False,
            job_id=self.job_id,
            state=state,
            job_exit_code=job_exit,
            client_exit=client_exit,
            duration_s=duration_s,
            download_dir="",
            env=dict(self.env),
            error=error,
            log_tail=log_tail,
            spec_summary=self._spec_summary(),
            started_at=started_at,
            finished_at=finished_at,
        )

    def _spec_summary(self) -> Dict[str, Any]:
        return {
            "job_name": self.spec.job_name,
            "entry_argv": list(self.spec.entry_argv),
            "file_count": len(self.spec.inputs),
            "input_bytes": self.spec.total_input_bytes(),
            "env_mode": self.spec.env.mode,
            "env_fingerprint": self.spec.env.fingerprint,
            "gpu": self.spec.resources.gpu,
            "timeout_s": self.spec.resources.timeout_s,
            "meta": dict(self.spec.meta),
        }


def _safe_output_path(root: Path, relpath: str) -> Path:
    """把服务端产物路径安全地拼到本地下载目录下。"""
    rel = str(relpath).replace("\\", "/")
    parts = [part for part in rel.split("/") if part not in ("", ".")]
    if (
        not parts
        or rel.startswith("/")
        or any(part == ".." for part in parts)
        or (len(parts[0]) >= 2 and parts[0][1] == ":")
    ):
        raise RemoteError(protocol.ERR_BAD_REQUEST, f"产物路径不安全: {relpath!r}")
    return root.joinpath(*parts)


# ---------------------------------------------------------------------------
# 顶层便捷函数（CLI / GUI 共用）
# ---------------------------------------------------------------------------
def build_spec(options: SubmitOptions, callbacks: Optional[Callbacks] = None) -> JobSpec:
    """根据 ``SubmitOptions`` 打包并构建 JobSpec（不连接服务器）。"""
    cb = callbacks or Callbacks()
    entry_argv = packager.parse_entry_command(options.entry)

    if options.incremental is None:
        meta: Optional[Dict[str, object]] = None
    elif options.incremental:
        meta = packager.incremental_meta(options.project_dir)
    else:
        meta = {}

    def on_warning(message: str) -> None:
        cb.log(f"[打包] {message}")

    return packager.make_job_spec(
        options.project_dir,
        entry_argv,
        job_name=options.job_name,
        requirements_path=options.requirements,
        env_mode=options.env_mode,
        python_version=options.python_version,
        include=options.include,
        exclude=options.exclude,
        outputs=options.outputs,
        gpu=options.gpu,
        gpu_count=options.gpu_count,
        timeout_s=options.timeout_s,
        meta=meta,
        client_version=CLIENT_VERSION,
        on_warning=on_warning,
    )


def _log_server_info(
    callbacks: Callbacks, info: Dict[str, Any], options: SubmitOptions
) -> None:
    callbacks.log(
        f"已连接 {info.get('server_id', '?')}"
        f" hostname={info.get('hostname', '?')}"
        f" platform={info.get('platform', '?')}"
        f" python={info.get('python', '?')}"
    )
    gpus = [g for g in (info.get("gpus") or []) if isinstance(g, dict)]
    if gpus:
        desc = ", ".join(
            f"[{g.get('index')}] {g.get('name')}"
            f"(free {g.get('memory_free_mb', '?')}MB)"
            for g in gpus
        )
        callbacks.log(f"服务端 GPU: {desc}")
    else:
        callbacks.log("服务端 GPU: 无（将以 CPU 模式执行）")
    if options.gpu and not gpus:
        callbacks.log("警告: 本次请求 GPU，但服务端未探测到可用 GPU，任务可能失败")


def submit_project(
    config: ClientConfig,
    options: SubmitOptions,
    callbacks: Optional[Callbacks] = None,
) -> RunResult:
    """完整提交链路；所有可预期失败都转成 ``RunResult``（不抛 ClientError）。"""
    cb = callbacks or Callbacks()
    started_at = _now()
    try:
        cb.progress(0.02, "打包项目文件…")
        spec = build_spec(options, cb)
    except packager.PackagerError as exc:
        cb.log(f"打包失败: {exc}")
        return _error_result(EXIT_USAGE, protocol.ERR_BAD_REQUEST, str(exc), started_at)
    except ValueError as exc:
        cb.log(f"参数错误: {exc}")
        return _error_result(EXIT_USAGE, protocol.ERR_BAD_REQUEST, str(exc), started_at)

    cb.log(
        f"打包完成：{len(spec.inputs)} 个文件，共 "
        f"{human_size(spec.total_input_bytes())}；"
        f"忽略规则已应用（.git/.venv/__pycache__/node_modules/outputs/*.log 等）"
    )
    if spec.meta.get("mode") == "incremental_training":
        cb.log(
            f"增量训练模式：base_model={spec.meta.get('base_model')} "
            f"dataset={spec.meta.get('dataset')}"
        )
    cb.progress(0.05, "连接服务器…")

    conn = Connection(config, client_version=CLIENT_VERSION)
    try:
        try:
            info = conn.connect()
        except ClientError as exc:
            cb.log(f"连接失败: {exc}")
            return _error_result(
                exc.exit_code,
                exc.code,
                str(exc),
                started_at,
                spec_summary={
                    "job_name": spec.job_name,
                    "file_count": len(spec.inputs),
                    "input_bytes": spec.total_input_bytes(),
                },
            )
        _log_server_info(cb, info, options)
        cb.progress(0.08, "已连接")
        runner = JobRunner(
            conn,
            spec,
            options.project_dir,
            chunk_size=options.chunk_size,
            callbacks=cb,
            receipt_dir=options.receipt_dir,
            receipt_request_id=options.receipt_request_id,
        )
        return runner.run(download_dir=options.download_dir, watch=options.watch)
    finally:
        conn.close()


def query_job_status(config: ClientConfig, job_id: str) -> Dict[str, Any]:
    """查询作业状态（``client.main status``）。异常由调用方映射退出码。"""
    conn = Connection(config, client_version=CLIENT_VERSION)
    try:
        conn.connect()
        conn.send_json({"kind": protocol.KIND_JOB_STATUS, "job_id": job_id})
        return conn.wait_for([protocol.KIND_JOB_STATUS_OK])
    finally:
        conn.close()
