"""作业目录与 ``state.json`` 原子读写。

目录布局（``docs/PROTOCOL.md`` 第 5 节）::

    <data_dir>/
      envs/<fingerprint>/                 # 环境缓存（envbuilder 管理）
      jobs/<job_id>/
        spec.json                         # 原始 JobSpec
        inputs/...                        # 客户端上传的文件（保持相对路径）
        outputs/...                       # 执行结束后收集的产物
        logs/job.log                      # stdout+stderr 合并日志
        state.json                        # 运行态、时间戳、退出码、产物清单、上传记录

约定：

- 所有相对路径一律 POSIX 风格（``a/b/c``），拒绝绝对路径、盘符与 ``..``。
- ``state.json`` 写入用「同目录临时文件 + ``os.replace``」，避免进程被杀时留下半截 JSON。
- ``state.json`` 里额外记录 ``uploads``（每个输入文件的完成状态），用于断线续传时
  判断 ``present`` / ``missing``；该字段不参与协议。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.hashing import sha256_file
from common.models import FileEntry, JobSpec, JobState, OutputEntry

# job-20261022-173000-ab12cd
_JOB_ID_RE = re.compile(r"^job-\d{8}-\d{6}-[0-9a-f]{6}$")

# 续传时校验 present 文件完整性的体积阈值：小文件每次重算 sha256，大文件只信
# ``uploads`` 记录 + 文件大小（避免几百 GiB 时每次提交都全量读盘）。
_VERIFY_SHA_MAX_BYTES = 16 * 1024 * 1024


class JobStoreError(Exception):
    """作业目录层错误基类。"""


class JobNotFoundError(JobStoreError):
    """作业不存在（协议错误码 ``NOT_FOUND``）。"""


class JobConflictError(JobStoreError):
    """作业状态/指纹冲突（协议错误码 ``CONFLICT``）。"""


def canonical_spec_fingerprint(spec: JobSpec) -> str:
    """任务指纹：``resume_job_id`` 复用时判断「是否同一个 spec」。

    ``inputs`` 先按 path 排序，其余字段走 ``sort_keys``，保证同一份任务描述
    无论客户端 JSON 键顺序如何，都得到同一指纹。
    """
    data = spec.to_dict()
    data["inputs"] = sorted(data.get("inputs") or [], key=lambda item: str(item.get("path")))
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """原子写 JSON：临时文件 + flush/fsync + ``os.replace``。"""
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:  # pragma: no cover - 某些文件系统不支持 fsync
            pass
    os.replace(tmp, path)


def safe_join(root: Path, relpath: str) -> Path:
    """把协议里的 POSIX 相对路径安全地拼到 ``root`` 下。

    拒绝：空路径、绝对路径、盘符、``..`` 段、解析后逃出 ``root`` 的路径。
    """
    rel = str(relpath or "").replace("\\", "/").strip()
    if not rel:
        raise ValueError("路径为空")
    if rel.startswith("/"):
        raise ValueError(f"不允许绝对路径: {relpath!r}")
    parts = [part for part in rel.split("/") if part not in ("", ".")]
    if not parts:
        raise ValueError(f"非法路径: {relpath!r}")
    if any(part == ".." for part in parts):
        raise ValueError(f"不允许 '..' 路径段: {relpath!r}")
    if ":" in parts[0]:
        raise ValueError(f"不允许盘符: {relpath!r}")
    candidate = root.joinpath(*parts)
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"路径逃出作业目录: {relpath!r}") from exc
    return candidate


class JobStore:
    """``<data_dir>/jobs`` 与 ``<data_dir>/envs`` 的薄封装（线程安全）。"""

    def __init__(self, data_dir: str | os.PathLike) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.jobs_dir = self.data_dir / "jobs"
        self.envs_dir = self.data_dir / "envs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.envs_dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, Any] = {}
        self._locks_guard = threading.Lock()

    # -- 路径 --------------------------------------------------------------
    def validate_job_id(self, job_id: str) -> str:
        text = str(job_id or "").strip()
        if not _JOB_ID_RE.match(text):
            raise JobNotFoundError(f"非法 job_id: {job_id!r}")
        return text

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / self.validate_job_id(job_id)

    def job_exists(self, job_id: str) -> bool:
        try:
            return self.job_dir(job_id).is_dir()
        except JobStoreError:
            return False

    def inputs_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "inputs"

    def outputs_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "outputs"

    def logs_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "logs"

    def log_path(self, job_id: str) -> Path:
        return self.logs_dir(job_id) / "job.log"

    def state_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "state.json"

    def spec_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "spec.json"

    def env_dir(self, fingerprint: str) -> Path:
        return self.envs_dir / str(fingerprint)

    def job_lock(self, job_id: str):
        """同一作业的 state.json 读改写用同一把可重入锁串行化。"""
        with self._locks_guard:
            lock = self._locks.get(job_id)
            if lock is None:
                lock = threading.RLock()
                self._locks[job_id] = lock
            return lock

    # -- 创建 / 读取 -------------------------------------------------------
    def new_job_id(self) -> str:
        while True:
            stamp = time.strftime("job-%Y%m%d-%H%M%S")
            job_id = f"{stamp}-{secrets.token_hex(3)}"
            if not self.job_exists(job_id):
                return job_id

    def create_job(self, spec: JobSpec, job_id: Optional[str] = None) -> Tuple[str, str]:
        """Create a job directory and persist spec/state.

        ``job_id`` is normally generated here.  Idempotency reservations pass a
        previously persisted id so a crash between "reserve id" and "create
        directory" can be retried without creating a second job.  With an
        explicit id this operation is idempotent **only** when the on-disk spec
        fingerprint matches; a different spec is rejected instead of silently
        reusing another caller's job.
        """
        if job_id is None:
            job_id = self.new_job_id()
        else:
            job_id = self.validate_job_id(str(job_id))
        fingerprint = canonical_spec_fingerprint(spec)
        directory = self.job_dir(job_id)
        with self.job_lock(job_id):
            state_path = self.state_path(job_id)
            if state_path.is_file():
                existing = self.load_state(job_id)
                stored = str(existing.get("spec_fingerprint") or "")
                if stored and stored != fingerprint:
                    raise JobConflictError(
                        f"作业 {job_id} 已绑定到不同 spec，拒绝幂等重建"
                    )
                return job_id, fingerprint
            # A crash can leave an empty/partial directory after mkdir.  Fill
            # in the missing pieces; never delete a directory here.
            (directory / "inputs").mkdir(parents=True, exist_ok=True)
            (directory / "outputs").mkdir(parents=True, exist_ok=True)
            (directory / "logs").mkdir(parents=True, exist_ok=True)
            _atomic_write_json(self.spec_path(job_id), spec.to_dict())
            now = time.time()
            state: Dict[str, Any] = {
                "job_id": job_id,
                "spec_fingerprint": fingerprint,
                "state": JobState.UPLOADING.value,
                "progress": 0.0,
                "created_at": now,
                "updated_at": now,
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "duration_s": None,
                "outputs": [],
                "log_tail": [],
                "env_state": None,
                "gpus": [],
                "uploads": {},
                "error": None,
            }
            _atomic_write_json(self.state_path(job_id), state)
            return job_id, fingerprint

    def load_spec(self, job_id: str) -> JobSpec:
        path = self.spec_path(job_id)
        if not path.is_file():
            raise JobNotFoundError(f"作业不存在: {job_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise JobStoreError(f"读取 spec.json 失败: {job_id}: {exc}") from exc
        return JobSpec.from_dict(data)

    def load_state(self, job_id: str) -> Dict[str, Any]:
        path = self.state_path(job_id)
        if not path.is_file():
            raise JobNotFoundError(f"作业不存在: {job_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise JobStoreError(f"读取 state.json 失败: {job_id}: {exc}") from exc
        if not isinstance(data, dict):
            raise JobStoreError(f"state.json 格式错误: {job_id}")
        return data

    def write_state(self, job_id: str, state: Dict[str, Any]) -> None:
        with self.job_lock(job_id):
            state["updated_at"] = time.time()
            _atomic_write_json(self.state_path(job_id), state)

    def update_state(self, job_id: str, **fields: Any) -> Dict[str, Any]:
        with self.job_lock(job_id):
            state = self.load_state(job_id)
            state.update(fields)
            state["updated_at"] = float(fields.get("updated_at") or time.time())
            _atomic_write_json(self.state_path(job_id), state)
            return state

    # -- 上传 --------------------------------------------------------------
    def input_path(self, job_id: str, relpath: str) -> Path:
        """校验并把协议路径映射到 ``inputs/`` 下的真实路径。"""
        root = self.inputs_dir(job_id)
        path = safe_join(root, relpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def output_path(self, job_id: str, relpath: str) -> Path:
        root = self.outputs_dir(job_id)
        return safe_join(root, relpath)

    def record_upload(self, job_id: str, entry: FileEntry) -> None:
        with self.job_lock(job_id):
            state = self.load_state(job_id)
            uploads = state.setdefault("uploads", {})
            uploads[entry.path] = {
                "size": int(entry.size),
                "sha256": entry.sha256,
                "done": True,
                "ts": time.time(),
            }
            _atomic_write_json(self.state_path(job_id), state)

    def present_missing(
        self,
        job_id: str,
        spec: JobSpec,
        verify: bool = True,
    ) -> Tuple[List[str], List[FileEntry]]:
        """返回 ``(present_paths, missing_entries)``。

        ``present`` 的判定：``uploads`` 标记 done + 文件存在 + 大小一致
        （小文件额外重算 sha256，防止磁盘被意外改写）。
        """
        state = self.load_state(job_id)
        uploads = state.get("uploads") or {}
        inputs_dir = self.inputs_dir(job_id)
        present: List[str] = []
        missing: List[FileEntry] = []
        for entry in spec.inputs:
            record = uploads.get(entry.path) or {}
            path = inputs_dir.joinpath(*entry.path.split("/"))
            ok = bool(record.get("done")) and path.is_file()
            if ok:
                try:
                    ok = path.stat().st_size == entry.size
                except OSError:
                    ok = False
            if ok and verify and entry.size <= _VERIFY_SHA_MAX_BYTES:
                try:
                    ok = sha256_file(path) == entry.sha256
                except OSError:
                    ok = False
            if ok:
                present.append(entry.path)
            else:
                missing.append(entry)
        return present, missing

    def all_inputs_present(self, job_id: str, spec: JobSpec) -> bool:
        _present, missing = self.present_missing(job_id, spec)
        return not missing

    # -- 日志 --------------------------------------------------------------
    def append_log(self, job_id: str, text: str) -> None:
        path = self.log_path(job_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8", errors="replace") as fh:
                fh.write(text)
        except OSError:  # 日志写失败不应影响作业
            pass

    def read_log_tail(self, job_id: str, lines: int = 20) -> List[str]:
        path = self.log_path(job_id)
        if not path.is_file():
            return []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return [line.rstrip("\r\n") for line in fh.readlines()[-lines:]]
        except OSError:
            return []

    # -- 产物 --------------------------------------------------------------
    def outputs_from_state(self, job_id: str) -> List[Dict[str, Any]]:
        try:
            state = self.load_state(job_id)
        except JobStoreError:
            return []
        result: List[Dict[str, Any]] = []
        for item in state.get("outputs") or []:
            try:
                result.append(OutputEntry.from_dict(item).to_dict())
            except (KeyError, TypeError, ValueError):
                continue
        return result

    # -- 清理 --------------------------------------------------------------
    def list_job_ids(self) -> List[str]:
        ids: List[str] = []
        for entry in self.jobs_dir.iterdir():
            if entry.is_dir() and _JOB_ID_RE.match(entry.name):
                ids.append(entry.name)
        ids.sort()
        return ids

    def cleanup_finished(self, keep: int = 50) -> List[str]:
        """只保留最近 ``keep`` 个已结束作业目录（运行中的不动）。"""
        removed: List[str] = []
        if keep <= 0:
            return removed
        candidates: List[Tuple[float, str]] = []
        for job_id in self.list_job_ids():
            try:
                state = self.load_state(job_id)
            except JobStoreError:
                continue
            state_name = str(state.get("state") or "")
            if state_name in (
                JobState.SUCCEEDED.value,
                JobState.FAILED.value,
                JobState.TIMEOUT.value,
                JobState.CANCELLED.value,
            ):
                try:
                    mtime = self.job_dir(job_id).stat().st_mtime
                except OSError:
                    mtime = 0.0
                candidates.append((mtime, job_id))
        candidates.sort(reverse=True)
        for _mtime, job_id in candidates[keep:]:
            shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
            removed.append(job_id)
        return removed

    def recover_interrupted(
        self,
        reason: str = "服务进程重启，原执行状态未知；未自动重跑，请人工核查",
    ) -> List[str]:
        """Mark non-terminal jobs found on disk as FAILED; never auto-runs."""
        recovered: List[str] = []
        for job_id in self.list_job_ids():
            try:
                state = self.load_state(job_id)
            except JobStoreError:
                continue
            if str(state.get("state") or "") in (
                JobState.SUCCEEDED.value,
                JobState.FAILED.value,
                JobState.TIMEOUT.value,
                JobState.CANCELLED.value,
            ):
                continue
            now = time.time()
            tail = list(state.get("log_tail") or [])[-19:] + [f"[server] {reason}"]
            self.update_state(
                job_id,
                state=JobState.FAILED.value,
                error=reason,
                finished_at=now,
                log_tail=tail,
            )
            self.append_log(job_id, f"\n[server] {reason}\n")
            recovered.append(job_id)
        return recovered


__all__ = [
    "JobStore",
    "JobStoreError",
    "JobNotFoundError",
    "JobConflictError",
    "canonical_spec_fingerprint",
    "safe_join",
]
