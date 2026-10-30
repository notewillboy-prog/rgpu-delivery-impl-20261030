"""TCP 服务端：监听、线程式会话、握手鉴权、协议消息分发。

线程模型：

- 主线程 ``accept()``，每个连接一个会话线程（``Session``）；
- ``Scheduler`` 有 ``max_parallel_jobs`` 个工作线程跑作业；
- 作业执行线程通过 ``JobRuntime`` 把 ``job_started/job_progress/job_event/job_finished``
  广播给订阅该作业的所有会话；
- 每个会话的 socket 写入用 ``Session._send_lock`` 串行化：``output_chunk`` 的
  「JSON 头 + BLOB」必须原子发出，不能被其它线程的 ``job_event`` 插进两帧之间。

协议实现严格按 ``docs/PROTOCOL.md`` v1（已冻结）：所有错误统一成
``{"kind":"error","code":...,"message":...}``，除 ``BAD_PROTOCOL``/``BAD_AUTH`` 外
不主动断连接。
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import logging
import os
import secrets
import socket
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

from common.config import ServerConfig
from common.hashing import sha256_file
from common.models import JobSpec, JobState, validate_job_spec
from common.protocol import (
    DEFAULT_CHUNK_SIZE,
    ERR_BAD_AUTH,
    ERR_BAD_PROTOCOL,
    ERR_BAD_REQUEST,
    ERR_BUSY,
    ERR_CONFLICT,
    ERR_ENV_BUILD,
    ERR_HASH_MISMATCH,
    ERR_INTERNAL,
    ERR_NOT_FOUND,
    ERR_TIMEOUT,
    KIND_BYE,
    KIND_FILE_CHUNK,
    KIND_FILE_DONE,
    KIND_HELLO,
    KIND_HELLO_OK,
    KIND_JOB_ACCEPTED,
    KIND_JOB_START,
    KIND_JOB_STATUS,
    KIND_JOB_SUBMIT,
    KIND_OUTPUT_ACK,
    KIND_OUTPUT_CHUNK,
    KIND_OUTPUT_END,
    KIND_OUTPUT_LIST,
    KIND_OUTPUT_LIST_OK,
    KIND_PING,
    KIND_PONG,
    MAX_FRAME_PAYLOAD,
    PROTOCOL_VERSION,
    FrameReader,
    FrameWriter,
    ProtocolError,
    check_version,
    make_error,
    require_fields,
)

from .envbuilder import EnvBuilder
from .executor import Executor
from .idempotency import IdempotencyConflict, IdempotencyError, IdempotencyIndex
from .jobstore import (
    JobConflictError,
    JobNotFoundError,
    JobStore,
    JobStoreError,
    canonical_spec_fingerprint,
)
from .runtime import JobRuntime, status_payload_from_state
from .scheduler import Scheduler, detect_gpus
from .weight_cache_prototype import WeightCache

LOG = logging.getLogger("rgpu.server")

HANDSHAKE_TIMEOUT_S = 60.0
_TERMINAL_STATES = {
    JobState.SUCCEEDED.value,
    JobState.FAILED.value,
    JobState.TIMEOUT.value,
    JobState.CANCELLED.value,
}
# 已知但当前 demo 不实现的「可选」消息：静默忽略，不算协议错误
_IGNORED_KINDS = {KIND_OUTPUT_ACK, KIND_OUTPUT_END, "upload_state"}


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{field} 必须是整数: {value!r}", ERR_BAD_REQUEST) from exc


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# 会话
# ---------------------------------------------------------------------------
class Session:
    """一个 TCP 连接上的会话：一次握手 + 若干请求/通知。"""

    _ids = itertools.count(1)

    def __init__(self, app: "ServerApp", sock: socket.socket, addr: Any) -> None:
        self.app = app
        self.sock = sock
        self.addr = addr
        self.session_id = f"conn-{next(Session._ids)}"
        self.reader = FrameReader(sock)
        self.writer = FrameWriter(sock)
        self._send_lock = threading.Lock()
        self._closed = threading.Event()
        self.authenticated = False
        self.job_id: Optional[str] = None
        self._subscribed: Set[str] = set()

    # -- 发送 --------------------------------------------------------------
    def send_json(self, message: Dict[str, Any]) -> None:
        if self._closed.is_set():
            raise ConnectionError("session closed")
        with self._send_lock:
            self.writer.send_json(message)

    def send_json_blob(self, header: Dict[str, Any], blob: bytes) -> None:
        """原子发送「JSON 头 + 紧跟的 BLOB 帧」。"""
        if self._closed.is_set():
            raise ConnectionError("session closed")
        with self._send_lock:
            self.writer.send_json(header)
            self.writer.send_blob(blob)

    # 广播用的统一入口
    def deliver(self, message: Dict[str, Any]) -> None:
        self.send_json(message)

    # -- 生命周期 ----------------------------------------------------------
    def run(self) -> None:
        try:
            self.sock.settimeout(HANDSHAKE_TIMEOUT_S)
            if not self._handshake():
                return
            self.sock.settimeout(float(self.app.config.idle_timeout_s))
            self._message_loop()
        except ConnectionError:
            LOG.debug("[%s] 连接已断开", self.session_id)
        except ProtocolError as exc:
            self._safe_error(exc.code, exc.message)
        except OSError as exc:
            LOG.debug("[%s] socket 错误: %s", self.session_id, exc)
        except Exception as exc:  # pragma: no cover - 兜底
            LOG.error("[%s] 会话异常: %r\n%s", self.session_id, exc, traceback.format_exc())
            self._safe_error(ERR_INTERNAL, f"服务端内部错误: {exc}")
        finally:
            self.close()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        for job_id in list(self._subscribed):
            runtime = self.app.find_runtime(job_id)
            if runtime is not None:
                runtime.unsubscribe(self)
        self._subscribed.clear()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
        self.app.unregister_session(self)
        LOG.debug("[%s] 会话关闭", self.session_id)

    # -- 握手 --------------------------------------------------------------
    def _handshake(self) -> bool:
        try:
            message = self.reader.read_json()
        except ProtocolError as exc:
            self._safe_error(exc.code, f"握手失败: {exc.message}")
            return False
        if message.get("kind") != KIND_HELLO:
            self._safe_error(ERR_BAD_PROTOCOL, "首条消息必须是 hello")
            return False
        try:
            check_version(message)
        except ProtocolError as exc:
            self._safe_error(exc.code, exc.message)
            return False
        role = str(message.get("role") or "")
        if role != "client":
            self._safe_error(ERR_BAD_REQUEST, f"role 必须为 'client'，收到 {role!r}")
            return False
        token = str(message.get("token") or "")
        expected = str(self.app.config.token or "")
        if not token or not expected or not hmac.compare_digest(token, expected):
            self._safe_error(ERR_BAD_AUTH, "token 无效")
            return False
        self.authenticated = True
        hello_ok = {
            "kind": KIND_HELLO_OK,
            "protocol": PROTOCOL_VERSION,
            "server_id": self.app.server_id,
            "hostname": self.app.hostname,
            "platform": sys.platform,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "max_frame": MAX_FRAME_PAYLOAD,
            "gpus": [gpu.to_dict() for gpu in self.app.gpus],
            "max_parallel_jobs": self.app.config.max_parallel_jobs,
            "env_cache_dir": str(self.app.store.envs_dir),
            "chunk_size": DEFAULT_CHUNK_SIZE,
        }
        self.send_json(hello_ok)
        LOG.info(
            "[%s] 客户端已连接（%s），%d 张 GPU",
            self.session_id, self.addr, len(self.app.gpus),
        )
        return True

    # -- 消息循环 ----------------------------------------------------------
    def _message_loop(self) -> None:
        handlers = {
            KIND_PING: self._on_ping,
            KIND_BYE: self._on_bye,
            KIND_HELLO: self._on_repeated_hello,
            KIND_JOB_SUBMIT: self._on_job_submit,
            KIND_FILE_CHUNK: self._on_file_chunk,
            KIND_FILE_DONE: self._on_file_done,
            KIND_JOB_START: self._on_job_start,
            KIND_JOB_STATUS: self._on_job_status,
            KIND_OUTPUT_LIST: self._on_output_list,
            KIND_OUTPUT_CHUNK: self._on_output_chunk,
        }
        while not self._closed.is_set():
            try:
                message = self.reader.read_json()
            except ProtocolError as exc:
                if exc.code == ERR_TIMEOUT:
                    LOG.info("[%s] 空闲超时（%ss），断开", self.session_id,
                             self.app.config.idle_timeout_s)
                    return
                self._safe_error(exc.code, exc.message)
                return
            kind = message.get("kind")
            if not isinstance(kind, str):
                self._safe_error(ERR_BAD_REQUEST, "消息缺少 kind")
                continue
            if kind in _IGNORED_KINDS:
                continue
            handler = handlers.get(kind)
            if handler is None:
                self._safe_error(ERR_BAD_REQUEST, f"未知消息类型: {kind}")
                continue
            try:
                handler(message)
            except ProtocolError as exc:
                self._safe_error(exc.code, exc.message)
                if exc.code in (ERR_BAD_PROTOCOL, ERR_BAD_AUTH):
                    return
            except JobNotFoundError as exc:
                self._safe_error(ERR_NOT_FOUND, str(exc))
            except JobConflictError as exc:
                self._safe_error(ERR_CONFLICT, str(exc))
            except JobStoreError as exc:
                self._safe_error(ERR_INTERNAL, str(exc))
            except (ConnectionError, OSError):
                return
            except Exception as exc:  # pragma: no cover - 兜底
                LOG.error("[%s] 处理 %s 异常: %r\n%s", self.session_id, kind, exc,
                          traceback.format_exc())
                self._safe_error(ERR_INTERNAL, f"服务端内部错误: {exc}")

    # -- 简单消息 ----------------------------------------------------------
    def _on_ping(self, _message: Dict[str, Any]) -> None:
        self.send_json({"kind": KIND_PONG, "ts": time.time()})

    def _on_bye(self, _message: Dict[str, Any]) -> None:
        self._closed.set()

    def _on_repeated_hello(self, _message: Dict[str, Any]) -> None:
        # 握手只允许一次；重发 hello 视为协议错误（客户端应重连而不是重放）。
        raise ProtocolError("握手已完成，重复 hello 非法", ERR_BAD_PROTOCOL)

    # -- job_submit --------------------------------------------------------
    def _on_job_submit(self, message: Dict[str, Any]) -> None:
        require_fields(message, "spec")
        spec_data = message.get("spec")
        if not isinstance(spec_data, dict):
            raise ProtocolError("spec 必须是 JSON 对象", ERR_BAD_REQUEST)
        try:
            spec = JobSpec.from_dict(spec_data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"spec 非法: {exc}", ERR_BAD_REQUEST) from exc
        try:
            validate_job_spec(spec)
        except ValueError as exc:
            raise ProtocolError(f"spec 校验失败: {exc}", ERR_BAD_REQUEST) from exc
        if spec.env.mode == "docker" and not self.app.config.enable_docker:
            raise ProtocolError(
                "服务端未启用 docker 环境模式（启动参数加 --enable-docker，"
                "或客户端改用 --env-mode venv）",
                ERR_ENV_BUILD,
            )
        resume_raw = message.get("resume_job_id")
        resume_job_id = str(resume_raw).strip() if resume_raw else ""
        reused = False
        idem_key = ""
        if not resume_job_id:
            meta = spec.meta if isinstance(spec.meta, dict) else {}
            idem_key = str(meta.get("idempotency_key") or "").strip()
        if resume_job_id:
            runtime = self.app.runtime_for_resume(
                resume_job_id, canonical_spec_fingerprint(spec)
            )
            LOG.info("[%s] 续传作业 %s", self.session_id, runtime.job_id)
        elif idem_key:
            fingerprint = canonical_spec_fingerprint(spec)
            try:
                job_id, reused = self.app.idempotency.get_or_create(
                    idem_key,
                    fingerprint,
                    lambda: self.app.store.create_job(spec)[0],
                )
                runtime = self.app.attach_runtime(job_id, fingerprint)
            except IdempotencyConflict as exc:
                raise ProtocolError(str(exc), ERR_CONFLICT) from exc
            except ValueError as exc:
                raise ProtocolError(f"idempotency key 非法: {exc}", ERR_BAD_REQUEST) from exc
            except IdempotencyError as exc:
                raise ProtocolError(f"idempotency index 错误: {exc}", ERR_INTERNAL) from exc
            LOG.info(
                "[%s] %s作业 %s（idempotency_key=%s）",
                self.session_id, "复用" if reused else "新", runtime.job_id, idem_key,
            )
        else:
            runtime = self.app.create_job(spec)
            LOG.info("[%s] 新作业 %s（%s）", self.session_id, runtime.job_id, spec.job_name)
        store = self.app.store
        present, missing = store.present_missing(runtime.job_id, spec)
        cache_hits: List[str] = []
        present, missing, cache_hits = self.app.cache_missing_inputs(
            runtime.job_id, spec, present, missing
        )
        received: Dict[str, int] = {}
        for entry in missing:
            try:
                target = store.input_path(runtime.job_id, entry.path)
                existing = target.stat().st_size
            except OSError:
                continue
            if 0 < existing < entry.size:
                received[entry.path] = int(existing)
        self.job_id = runtime.job_id
        self.send_json(
            {
                "kind": KIND_JOB_ACCEPTED,
                "job_id": runtime.job_id,
                "missing": [entry.to_dict() for entry in missing],
                "present": present,
                "received": received,
                "cache_hits": cache_hits,
                "reused": reused,
                "state": runtime.state.value,
                "env": {
                    "fingerprint": spec.env.fingerprint,
                    "reuse": bool(spec.env.reuse),
                },
                "chunk_size": DEFAULT_CHUNK_SIZE,
            }
        )

    # -- 上传 --------------------------------------------------------------
    def _require_job(self) -> Tuple[JobRuntime, JobSpec]:
        if not self.job_id:
            raise ProtocolError("尚未 job_submit，无法上传文件", ERR_BAD_REQUEST)
        runtime = self.app.require_runtime(self.job_id)
        return runtime, runtime.spec

    def _on_file_chunk(self, message: Dict[str, Any]) -> None:
        # 协议：JSON 头之后紧跟一个 BLOB 帧。**无论头是否合法都先把 BLOB 读掉**，
        # 否则帧流会错位（下一条 read_json 会把 blob 当 JSON）。
        path = str(message.get("path") or "")
        offset_raw = message.get("offset")
        length_raw = message.get("length")
        sha_expected = str(message.get("sha256") or "")
        blob = self.reader.read_blob()
        runtime, spec = self._require_job()
        entry = spec.input_index().get(path)
        if entry is None:
            raise ProtocolError(f"文件未在 spec.inputs 中声明: {path!r}", ERR_BAD_REQUEST)
        offset = _as_int(offset_raw, "offset")
        length = _as_int(length_raw, "length")
        if length <= 0 or length != len(blob):
            raise ProtocolError(
                f"分片长度不匹配: length={length} blob={len(blob)}", ERR_BAD_REQUEST
            )
        if offset < 0 or offset + length > entry.size:
            raise ProtocolError(
                f"分片越界: offset={offset} length={length} size={entry.size}",
                ERR_BAD_REQUEST,
            )
        actual = _sha256_hex(blob)
        if sha_expected and not hmac.compare_digest(actual, sha_expected):
            raise ProtocolError(
                f"分片 sha256 不匹配: {path} offset={offset}", ERR_HASH_MISMATCH
            )
        try:
            target = self.app.store.input_path(runtime.job_id, path)
        except ValueError as exc:
            raise ProtocolError(str(exc), ERR_BAD_REQUEST) from exc
        try:
            if target.exists():
                with open(target, "r+b") as fh:
                    fh.seek(0, os.SEEK_END)
                    current = fh.tell()
                    if offset > current:
                        raise ProtocolError(
                            f"offset 不连续: 期望 {current}，收到 {offset}", ERR_BAD_REQUEST
                        )
                    fh.seek(offset)
                    fh.truncate(offset)
                    fh.write(blob)
                    fh.flush()
            else:
                if offset != 0:
                    raise ProtocolError(
                        f"offset 不连续: 新文件必须从 0 开始，收到 {offset}", ERR_BAD_REQUEST
                    )
                with open(target, "wb") as fh:
                    fh.write(blob)
                    fh.flush()
        except OSError as exc:
            raise ProtocolError(f"写入分片失败: {exc}", ERR_INTERNAL) from exc
        self.send_json(
            {"kind": "file_ok", "path": path, "offset": offset + length}
        )

    def _on_file_done(self, message: Dict[str, Any]) -> None:
        runtime, spec = self._require_job()
        path = str(message.get("path") or "")
        entry = spec.input_index().get(path)
        if entry is None:
            raise ProtocolError(f"文件未在 spec.inputs 中声明: {path!r}", ERR_BAD_REQUEST)
        try:
            target = self.app.store.input_path(runtime.job_id, path)
        except ValueError as exc:
            raise ProtocolError(str(exc), ERR_BAD_REQUEST) from exc
        if not target.is_file():
            raise ProtocolError(f"文件尚未上传: {path}", ERR_BAD_REQUEST)
        size = target.stat().st_size
        if size != entry.size:
            # 大小不对：清掉让人从头重传（协议 3.3：整文件不匹配 → 从 0 重传）
            try:
                target.write_bytes(b"")
            except OSError:
                pass
            raise ProtocolError(
                f"文件大小不匹配: {path} 期望 {entry.size}，实际 {size}",
                ERR_HASH_MISMATCH,
            )
        actual = sha256_file(target)
        if not hmac.compare_digest(actual, entry.sha256):
            try:
                target.write_bytes(b"")
            except OSError:
                pass
            raise ProtocolError(
                f"整文件 sha256 不匹配: {path}（已清空，请从 0 重传）",
                ERR_HASH_MISMATCH,
            )
        self.app.store.record_upload(runtime.job_id, entry)
        if self.app.weight_cache is not None:
            try:
                self.app.weight_cache.put(target, entry.sha256, entry.size)
            except (OSError, ValueError) as exc:
                LOG.warning(
                    "[%s] 写入权重缓存失败（不影响作业）: %s: %s",
                    self.session_id, path, exc,
                )
        self.send_json({"kind": "file_ok", "path": path, "done": True})
        LOG.info("[%s] 文件完成: %s（%d B）", self.session_id, path, size)

    # -- 启动 --------------------------------------------------------------
    def _on_job_start(self, message: Dict[str, Any]) -> None:
        require_fields(message, "job_id")
        job_id = str(message.get("job_id") or "")
        runtime = self.app.require_runtime(job_id)
        _present, missing = self.app.store.present_missing(job_id, runtime.spec)
        if missing:
            names = ", ".join(entry.path for entry in missing[:5])
            raise ProtocolError(f"还有 {len(missing)} 个文件未上传完: {names}", ERR_BAD_REQUEST)
        self._subscribe(runtime)
        if runtime.state.value in _TERMINAL_STATES:
            # 断线重连时作业已经结束：补发终态，客户端不用再等
            self.send_json(runtime.job_started_message(attached=True))
            self.send_json(runtime.finished_message())
            return
        if runtime.enqueued or runtime.state.value not in (
            JobState.UPLOADING.value,
            JobState.QUEUED.value,
        ):
            # 已经排队/正在跑：重连的会话只补一条快照，不重复入队
            self.send_json(runtime.job_started_message(attached=True))
            return
        runtime.set_state(JobState.QUEUED)
        if not self.app.scheduler.submit(runtime):
            raise ProtocolError("服务端队列已满（BUSY），请稍后重试 job_start", ERR_BUSY)
        LOG.info("[%s] 作业 %s 已入队", self.session_id, job_id)

    def _subscribe(self, runtime: JobRuntime) -> None:
        if runtime.job_id not in self._subscribed:
            runtime.subscribe(self)
            self._subscribed.add(runtime.job_id)
        self.job_id = runtime.job_id

    # -- 状态 / 产物 -------------------------------------------------------
    def _on_job_status(self, message: Dict[str, Any]) -> None:
        require_fields(message, "job_id")
        job_id = str(message.get("job_id") or "")
        runtime = self.app.find_runtime(job_id)
        if runtime is not None:
            self.send_json(runtime.status_payload())
            return
        try:
            state = self.app.store.load_state(job_id)
        except JobStoreError as exc:
            raise ProtocolError(str(exc), ERR_NOT_FOUND) from exc
        if self.app.config.strict_recovery and str(state.get("state") or "") not in _TERMINAL_STATES:
            raise ProtocolError(
                "服务进程内没有此作业的运行记录；原进程及执行状态未知，请核查原任务，禁止自动重投或重启", ERR_CONFLICT
            )
        payload = status_payload_from_state(state)
        payload["job_id"] = job_id
        self.send_json(payload)

    def _outputs_of(self, job_id: str) -> List[Dict[str, Any]]:
        runtime = self.app.find_runtime(job_id)
        if runtime is not None:
            return [dict(item) for item in runtime.outputs]
        return self.app.store.outputs_from_state(job_id)

    def _on_output_list(self, message: Dict[str, Any]) -> None:
        require_fields(message, "job_id")
        job_id = str(message.get("job_id") or "")
        self.app.require_job_exists(job_id)
        self.send_json(
            {
                "kind": KIND_OUTPUT_LIST_OK,
                "job_id": job_id,
                "outputs": self._outputs_of(job_id),
            }
        )

    def _on_output_chunk(self, message: Dict[str, Any]) -> None:
        require_fields(message, "job_id", "path")
        job_id = str(message.get("job_id") or "")
        path = str(message.get("path") or "")
        offset = _as_int(message.get("offset"), "offset")
        self.app.require_job_exists(job_id)
        outputs = {str(item.get("path")): item for item in self._outputs_of(job_id)}
        entry = outputs.get(path)
        if entry is None:
            raise ProtocolError(f"产物不存在: {path!r}", ERR_NOT_FOUND)
        try:
            source = self.app.store.output_path(job_id, path)
        except ValueError as exc:
            raise ProtocolError(str(exc), ERR_BAD_REQUEST) from exc
        if not source.is_file():
            raise ProtocolError(f"产物文件缺失: {path}", ERR_NOT_FOUND)
        size = int(entry.get("size") or source.stat().st_size)
        if offset < 0 or offset > size:
            raise ProtocolError(
                f"产物 offset 越界: offset={offset} size={size}", ERR_BAD_REQUEST
            )
        if size == 0:
            header = {
                "kind": KIND_OUTPUT_CHUNK,
                "job_id": job_id,
                "path": path,
                "offset": 0,
                "length": 0,
                "sha256": _sha256_hex(b""),
                "eof": True,
            }
            self.send_json_blob(header, b"")
            return
        length = min(DEFAULT_CHUNK_SIZE, size - offset)
        try:
            with open(source, "rb") as fh:
                fh.seek(offset)
                blob = fh.read(length)
        except OSError as exc:
            raise ProtocolError(f"读取产物失败: {path}: {exc}", ERR_INTERNAL) from exc
        header = {
            "kind": KIND_OUTPUT_CHUNK,
            "job_id": job_id,
            "path": path,
            "offset": offset,
            "length": len(blob),
            "sha256": _sha256_hex(blob),
            "eof": offset + len(blob) >= size,
        }
        self.send_json_blob(header, blob)

    # -- 错误 --------------------------------------------------------------
    def _safe_error(self, code: str, message: str) -> None:
        try:
            self.send_json(make_error(code, message))
        except (ConnectionError, OSError):
            pass


# ---------------------------------------------------------------------------
# 服务端应用
# ---------------------------------------------------------------------------
class ServerApp:
    """持有 socket、作业仓库、环境构建器、调度器与执行器。"""

    def __init__(
        self,
        config: ServerConfig,
        gpus: Optional[List[Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not str(config.data_dir or "").strip():
            raise ValueError("ServerConfig.data_dir 不能为空")
        self.config = config
        self.logger = logger or LOG
        self.store = JobStore(config.data_dir)
        self.idempotency = IdempotencyIndex(self.store.data_dir / "idempotency")
        self.weight_cache = None
        if int(getattr(config, "weight_cache_bytes", 0) or 0) > 0:
            self.weight_cache = WeightCache(
                self.store.data_dir / "weights-cache",
                quota_bytes=int(config.weight_cache_bytes),
                reserve_bytes=1024 ** 3,
            )
        self.gpus = detect_gpus(logger=self.log) if gpus is None else list(gpus)
        self.envbuilder = EnvBuilder(
            self.store.envs_dir,
            python_bin=config.python_bin,
            enable_docker=config.enable_docker,
            logger=self.log,
        )
        self.executor = Executor(
            self.store,
            self.envbuilder,
            python_bin=config.python_bin,
            enable_docker=config.enable_docker,
            logger=self.log,
        )
        self.scheduler = Scheduler(
            max_parallel=config.max_parallel_jobs,
            gpus=self.gpus,
            runner=self.executor.run,
            logger=self.log,
            strict_gpu=config.strict_gpu,
            min_gpu_free_mb=config.min_gpu_free_mb,
        )
        if config.strict_gpu:
            self.executor.gpu_prelaunch_check = self.scheduler.validate_prelaunch
        self.server_id = "srv-" + secrets.token_hex(4)
        self.hostname = socket.gethostname()
        self._jobs: Dict[str, JobRuntime] = {}
        self._jobs_lock = threading.Lock()
        self._sessions: Set[Session] = set()
        self._sessions_lock = threading.Lock()
        self._listener: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._shutdown_done = False
        self._ready_host = str(config.host)
        self._ready_port = int(config.port)

    # -- 日志 --------------------------------------------------------------
    def log(self, message: str) -> None:
        self.logger.info("%s", message)

    # -- 生命周期 ----------------------------------------------------------
    def bind(self) -> Tuple[str, int]:
        host = str(self.config.host or "0.0.0.0")
        port = int(self.config.port or 0)
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(128)
        self._listener = sock
        sockname = sock.getsockname()
        self._ready_host = str(sockname[0])
        self._ready_port = int(sockname[1])
        self.log(
            f"[server] 监听 {self._ready_host}:{self._ready_port}，"
            f"数据目录 {self.store.data_dir}，最大并发 {self.config.max_parallel_jobs}"
        )
        return self._ready_host, self._ready_port

    def ready_line(self) -> str:
        return (
            f"RGPU_SERVER_READY host={self._ready_host} "
            f"port={self._ready_port} data_dir={self.store.data_dir}"
        )

    def print_ready(self) -> None:
        """测试靠这一行判断服务端就绪：必须打到 stdout 并立即 flush。"""
        print(self.ready_line(), flush=True)

    def serve_forever(self) -> None:
        if self._listener is None:
            self.bind()
        self.scheduler.start()
        self.print_ready()
        try:
            while not self._stop.is_set():
                assert self._listener is not None
                try:
                    conn, addr = self._listener.accept()
                except OSError:
                    break
                session = Session(self, conn, addr)
                with self._sessions_lock:
                    self._sessions.add(session)
                thread = threading.Thread(
                    target=session.run,
                    name=f"rgpu-session-{session.session_id}",
                    daemon=True,
                )
                thread.start()
        except KeyboardInterrupt:  # pragma: no cover - 交互式停止
            self.log("[server] 收到 Ctrl+C，准备退出")
        finally:
            self.shutdown()

    def serve_in_background(self) -> threading.Thread:
        """测试/嵌入用：后台线程起服务，返回线程对象。"""
        thread = threading.Thread(target=self.serve_forever, name="rgpu-server", daemon=True)
        thread.start()
        return thread

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._stop.set()
        self.log("[server] 正在关闭…")
        try:
            self.scheduler.stop()
        except Exception:  # pragma: no cover
            pass
        try:
            self.executor.terminate_all()
        except Exception:  # pragma: no cover
            pass
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        with self._sessions_lock:
            sessions = list(self._sessions)
        for session in sessions:
            try:
                session.close()
            except Exception:  # pragma: no cover
                pass

    # -- 会话注册 ----------------------------------------------------------
    def register_session(self, session: Session) -> None:
        with self._sessions_lock:
            self._sessions.add(session)

    def unregister_session(self, session: Session) -> None:
        with self._sessions_lock:
            self._sessions.discard(session)

    # -- 作业仓库 ----------------------------------------------------------
    def find_runtime(self, job_id: str) -> Optional[JobRuntime]:
        with self._jobs_lock:
            return self._jobs.get(str(job_id or ""))

    def _register_runtime(self, runtime: JobRuntime) -> JobRuntime:
        with self._jobs_lock:
            self._jobs[runtime.job_id] = runtime
        return runtime

    def create_job(self, spec: JobSpec) -> JobRuntime:
        job_id, _fingerprint = self.store.create_job(spec)
        state = self.store.load_state(job_id)
        runtime = JobRuntime(self.store, job_id, spec, state)
        return self._register_runtime(runtime)

    def attach_runtime(self, job_id: str, spec_fingerprint: str) -> JobRuntime:
        """Attach to an existing job by idempotency key, including terminal jobs."""
        self.require_job_exists(job_id)
        state = self.store.load_state(job_id)
        stored = str(state.get("spec_fingerprint") or "")
        if stored and stored != spec_fingerprint:
            raise ProtocolError(
                "idempotency key 已绑定到不同 spec 的作业", ERR_CONFLICT
            )
        runtime = self.find_runtime(job_id)
        if runtime is not None:
            return runtime
        try:
            spec = self.store.load_spec(job_id)
        except JobStoreError as exc:
            raise ProtocolError(str(exc), ERR_NOT_FOUND) from exc
        runtime = JobRuntime(self.store, job_id, spec, state)
        return self._register_runtime(runtime)

    def cache_missing_inputs(
        self,
        job_id: str,
        spec: JobSpec,
        present: List[str],
        missing: List[Any],
    ) -> Tuple[List[str], List[Any], List[str]]:
        """Materialize cacheable missing inputs; returns updated lists + hit paths."""
        if self.weight_cache is None:
            return present, missing, []
        present = list(present)
        missing = list(missing)
        hits: List[str] = []
        for entry in list(missing):
            try:
                target = self.store.input_path(job_id, entry.path)
                if target.exists():
                    if target.stat().st_size == 0:
                        target.unlink()
                    else:
                        continue
                if not self.weight_cache.contains(entry.sha256, entry.size):
                    continue
                self.weight_cache.materialize(entry.sha256, entry.size, target)
                self.store.record_upload(job_id, entry)
                present.append(entry.path)
                missing.remove(entry)
                hits.append(entry.path)
            except (OSError, ValueError) as exc:
                self.log(f"缓存命中失败，回退上传: {entry.path}: {exc}")
        return present, missing, hits

    def require_job_exists(self, job_id: str) -> None:
        try:
            self.store.validate_job_id(job_id)
        except JobStoreError as exc:
            raise ProtocolError(str(exc), ERR_NOT_FOUND) from exc
        if not self.store.job_exists(job_id):
            raise ProtocolError(f"作业不存在: {job_id}", ERR_NOT_FOUND)

    def require_runtime(self, job_id: str) -> JobRuntime:
        job_id = str(job_id or "")
        self.require_job_exists(job_id)
        runtime = self.find_runtime(job_id)
        if runtime is not None:
            return runtime
        # 服务端重启后内存态丢失：终态作业可以按磁盘 state 补发终态，
        # 运行中的作业无法恢复执行（demo 明确不做断点执行恢复）。
        state = self.store.load_state(job_id)
        if str(state.get("state") or "") in _TERMINAL_STATES:
            try:
                spec = self.store.load_spec(job_id)
            except JobStoreError as exc:
                raise ProtocolError(str(exc), ERR_NOT_FOUND) from exc
            runtime = JobRuntime(self.store, job_id, spec, state)
            return self._register_runtime(runtime)
        raise ProtocolError(
            "作业运行态丢失（服务端可能重启过），请重新提交该任务", ERR_CONFLICT
        )

    def runtime_for_resume(self, job_id: str, spec_fingerprint: str) -> JobRuntime:
        """断线重连：校验 spec 指纹后复用同一 job。"""
        self.require_job_exists(job_id)
        state = self.store.load_state(job_id)
        stored = str(state.get("spec_fingerprint") or "")
        if stored and stored != spec_fingerprint:
            raise ProtocolError(
                "resume_job_id 对应的任务描述与本次提交不一致（spec 指纹不同）",
                ERR_CONFLICT,
            )
        if str(state.get("state") or "") in _TERMINAL_STATES:
            raise ProtocolError(
                f"作业 {job_id} 已结束（{state.get('state')}），无法续传，请重新提交",
                ERR_CONFLICT,
            )
        runtime = self.find_runtime(job_id)
        if runtime is not None:
            return runtime
        if self.config.strict_recovery:
            raise ProtocolError(
                "服务进程内没有此作业的运行记录；原进程及执行状态未知，请核查原任务，禁止自动重投或重启", ERR_CONFLICT
            )
        try:
            spec = self.store.load_spec(job_id)
        except JobStoreError as exc:
            raise ProtocolError(str(exc), ERR_NOT_FOUND) from exc
        runtime = JobRuntime(self.store, job_id, spec, state)
        return self._register_runtime(runtime)


__all__ = ["ServerApp", "Session"]
