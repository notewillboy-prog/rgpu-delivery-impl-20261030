"""作业运行态：状态机、会话订阅广播、``state.json`` 落盘。

一个作业从 ``job_submit`` 起就有一个 ``JobRuntime``：

- 记录 ``JobState`` / 时间戳 / 退出码 / 产物清单 / 环境状态；
- 维护「正在关注这个作业的会话」集合，执行期把 ``job_started`` /
  ``job_progress`` / ``job_event`` / ``job_finished`` 推给所有订阅者
  （断线重连后新会话重新订阅即可继续看日志）；
- 变化写入 ``state.json``（保留 ``jobstore`` 记录的 ``uploads`` 等字段）。

线程模型：执行器线程、会话线程、调度线程都会触碰本对象，所有可变字段用
``RLock`` 保护；向会话写 socket 的操作在锁外进行，避免慢客户端阻塞状态更新。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from common.models import JobSpec, JobState

from .jobstore import JobStore

LOG_TAIL_LINES = 20
_TERMINAL = {
    JobState.SUCCEEDED,
    JobState.FAILED,
    JobState.TIMEOUT,
    JobState.CANCELLED,
}


class JobRuntime:
    """单个作业的内存运行态 + 广播。"""

    def __init__(
        self,
        store: JobStore,
        job_id: str,
        spec: JobSpec,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.store = store
        self.job_id = job_id
        self.spec = spec
        self._lock = threading.RLock()
        self._subs: Dict[str, Any] = {}
        self._seq = 0
        self._log_tail: Deque[str] = deque(maxlen=LOG_TAIL_LINES)
        self.state = JobState.UPLOADING
        self.progress = 0.0
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.exit_code: Optional[int] = None
        self.duration_s: Optional[float] = None
        self.outputs: List[Dict[str, Any]] = []
        self.env_state: Optional[str] = None
        self.env_fingerprint = spec.env.fingerprint or ""
        self.gpus: List[int] = []
        self.note = ""
        self.enqueued = False
        if state:
            self.restore(state)

    # -- 状态恢复 ----------------------------------------------------------
    def restore(self, state: Dict[str, Any]) -> None:
        with self._lock:
            raw = str(state.get("state") or JobState.UPLOADING.value)
            try:
                self.state = JobState(raw)
            except ValueError:
                self.state = JobState.FAILED
            self.progress = float(state.get("progress") or 0.0)
            self.created_at = float(state.get("created_at") or time.time())
            self.updated_at = float(state.get("updated_at") or self.created_at)
            self.started_at = state.get("started_at")
            self.finished_at = state.get("finished_at")
            self.exit_code = state.get("exit_code")
            self.duration_s = state.get("duration_s")
            self.outputs = [dict(item) for item in state.get("outputs") or [] if isinstance(item, dict)]
            self.env_state = state.get("env_state")
            self.gpus = [int(x) for x in state.get("gpus") or []]
            self.note = str(state.get("note") or "")
            self._log_tail = deque(
                [str(line) for line in state.get("log_tail") or []][-LOG_TAIL_LINES:],
                maxlen=LOG_TAIL_LINES,
            )

    # -- 订阅 --------------------------------------------------------------
    def subscribe(self, session: Any) -> None:
        with self._lock:
            self._subs[str(session.session_id)] = session

    def unsubscribe(self, session: Any) -> None:
        with self._lock:
            self._subs.pop(str(session.session_id), None)

    def subscribers(self) -> List[Any]:
        with self._lock:
            return list(self._subs.values())

    def emit(self, message: Dict[str, Any]) -> None:
        """广播一条消息；写失败的会话自动退订（连接已断）。"""
        dead: List[Any] = []
        for session in self.subscribers():
            try:
                session.deliver(message)
            except Exception:  # 连接断了/写失败：退订即可
                dead.append(session)
        for session in dead:
            self.unsubscribe(session)

    # -- 状态更新 ----------------------------------------------------------
    @property
    def state_value(self) -> str:
        return self.state.value

    def persist(self) -> None:
        """把运行态写回 state.json（保留 uploads 等其它字段）。"""
        try:
            self.updated_at = time.time()
            self.store.update_state(
                self.job_id,
                state=self.state.value,
                progress=self.progress,
                created_at=self.created_at,
                updated_at=self.updated_at,
                started_at=self.started_at,
                finished_at=self.finished_at,
                exit_code=self.exit_code,
                duration_s=self.duration_s,
                outputs=self.outputs,
                log_tail=list(self._log_tail),
                env_state=self.env_state,
                env_fingerprint=self.env_fingerprint,
                gpus=self.gpus,
                note=self.note,
            )
        except Exception:  # 落盘失败不阻断执行（内存态仍然正确）
            pass

    def set_state(self, state: JobState, persist: bool = True) -> None:
        with self._lock:
            self.state = state
            if state is JobState.RUNNING and self.started_at is None:
                self.started_at = time.time()
            if state in _TERMINAL:
                self.finished_at = self.finished_at or time.time()
                if self.started_at:
                    self.duration_s = round(self.finished_at - self.started_at, 3)
        if persist:
            self.persist()

    def set_env(self, fingerprint: str, state: str) -> None:
        with self._lock:
            self.env_fingerprint = fingerprint
            self.env_state = state
        self.persist()

    def set_gpus(self, gpus: List[int]) -> None:
        with self._lock:
            self.gpus = list(gpus)

    def set_progress(self, value: float, note: str = "", persist: bool = True) -> None:
        with self._lock:
            self.progress = max(0.0, min(1.0, float(value)))
            if note:
                self.note = note
        if persist:
            self.persist()

    # -- 日志 / 事件 -------------------------------------------------------
    def append_log_line(self, text: str, stream: str = "stdout") -> None:
        """追加一行作业日志：写 ``logs/job.log`` 并广播 ``job_event``。"""
        line = str(text)
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._log_tail.append(line.rstrip("\r\n"))
        self.store.append_log(self.job_id, line if line.endswith("\n") else line + "\n")
        self.emit(
            {
                "kind": "job_event",
                "job_id": self.job_id,
                "seq": seq,
                "stream": stream,
                "data": line if line.endswith("\n") else line + "\n",
                "ts": time.time(),
            }
        )

    def append_raw_log(self, text: str) -> None:
        """只写日志文件，不广播（用于环境构建过程的原始输出）。"""
        self.store.append_log(self.job_id, text)

    def emit_progress(self, state: str, progress: float, message: str = "") -> None:
        with self._lock:
            self.progress = max(0.0, min(1.0, float(progress)))
            if message:
                self.note = message
        self.persist()
        self.emit(
            {
                "kind": "job_progress",
                "job_id": self.job_id,
                "state": state,
                "progress": round(self.progress, 3),
                "message": message,
                "ts": time.time(),
            }
        )

    # -- 结果 --------------------------------------------------------------
    def job_started_message(self, attached: bool = False) -> Dict[str, Any]:
        with self._lock:
            env_state = self.env_state or "built"
            gpus = list(self.gpus)
            note = self.note
        message: Dict[str, Any] = {
            "kind": "job_started",
            "job_id": self.job_id,
            "env": {"fingerprint": self.env_fingerprint, "state": env_state},
            "gpus": gpus,
        }
        if note:
            message["note"] = note
        if attached:
            message["attached"] = True
        return message

    def finished_message(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "kind": "job_finished",
                "job_id": self.job_id,
                "state": self.state.value,
                "exit_code": self.exit_code,
                "duration_s": self.duration_s,
                "outputs": [dict(item) for item in self.outputs],
                "log_tail": list(self._log_tail),
                "env": {"fingerprint": self.env_fingerprint, "state": self.env_state or "built"},
            }

    def mark_finished(
        self,
        state: JobState,
        exit_code: Optional[int],
        outputs: Optional[List[Dict[str, Any]]] = None,
        log_tail: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            self.state = state
            self.exit_code = -1 if exit_code is None else int(exit_code)
            if outputs is not None:
                self.outputs = [dict(item) for item in outputs]
            if log_tail is not None:
                self._log_tail = deque(
                    [str(line) for line in log_tail][-LOG_TAIL_LINES:], maxlen=LOG_TAIL_LINES
                )
            self.finished_at = time.time()
            if self.started_at:
                self.duration_s = round(self.finished_at - self.started_at, 3)
            self.progress = 1.0 if state is JobState.SUCCEEDED else self.progress
        self.persist()
        finished = self.finished_message()
        self.emit(finished)
        return finished

    # -- 查询 --------------------------------------------------------------
    def status_payload(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "kind": "job_status_ok",
                "job_id": self.job_id,
                "state": self.state.value,
                "progress": round(self.progress, 3),
                "outputs": [dict(item) for item in self.outputs],
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "exit_code": self.exit_code,
                "error": self.note or None,
                "log_tail": list(self._log_tail),
                "gpus": list(self.gpus),
                "env": {"fingerprint": self.env_fingerprint, "state": self.env_state},
            }

    @property
    def log_tail(self) -> List[str]:
        with self._lock:
            return list(self._log_tail)


def status_payload_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """从磁盘 state.json 组装 ``job_status_ok``（服务端重启后的查询兜底）。"""
    return {
        "kind": "job_status_ok",
        "job_id": str(state.get("job_id") or ""),
        "state": str(state.get("state") or ""),
        "progress": float(state.get("progress") or 0.0),
        "outputs": [dict(item) for item in state.get("outputs") or [] if isinstance(item, dict)],
        "created_at": state.get("created_at"),
        "updated_at": state.get("updated_at") or state.get("created_at"),
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "exit_code": state.get("exit_code"),
        "error": state.get("error") or state.get("note"),
        "log_tail": [str(line) for line in state.get("log_tail") or []],
        "gpus": [int(x) for x in state.get("gpus") or []],
        "env": {
            "fingerprint": state.get("env_fingerprint"),
            "state": state.get("env_state"),
        },
    }


__all__ = ["JobRuntime", "status_payload_from_state", "LOG_TAIL_LINES"]
