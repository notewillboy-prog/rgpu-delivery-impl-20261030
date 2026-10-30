"""协议参考实现（测试专用「假服务端」）。

目的：在真 ``server/`` 落地前，用一份按 ``docs/PROTOCOL.md`` v1 写的最小实现，
把 ``tests/rawflow.py`` + ``tests/raw_scenarios.py`` 的断线续传 / 哈希篡改测试逻辑
先跑通，避免「测试代码自己写错却被当成服务端 bug」。

实现边界（够用即可，不是产品代码）：
- hello 鉴权（BAD_AUTH）、版本校验（BAD_PROTOCOL）、ping/pong；
- job_submit（含 resume_job_id：present 已完成的文件、missing 其余文件）；
- file_chunk/file_done：逐片 + 整文件 sha256 校验，失败返回 HASH_MISMATCH；
- job_start：job_started(env.state=built/cached) → 两条 epoch job_event →
  生成 outputs/model.json + metrics.json → job_finished(succeeded)；
- job_status / output_list / output_chunk（带 eof）/ bye。

不支持：真实执行、并发队列、docker、超时。仅绑定 127.0.0.1，随机端口。
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import protocol
from common.hashing import sha256_bytes, sha256_file
from common.models import JobSpec

_FAKE_CHUNK = 1024 * 1024


class FakeServer:
    def __init__(self, *, port: int = 0, token: str = "e2e-token",
                 data_dir: Optional[Path] = None, host: str = "127.0.0.1") -> None:
        self.host = host
        self.token = token
        self.data_dir = Path(data_dir) if data_dir is not None else Path(".")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, int(port)))
        self._sock.listen(8)
        self.port = int(self._sock.getsockname()[1])
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._clients: List[socket.socket] = []
        self._lock = threading.Lock()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.built_envs: set = set()
        self.errors: List[str] = []

    # -- 生命周期 ----------------------------------------------------------
    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    def start(self) -> "FakeServer":
        self._thread = threading.Thread(target=self._accept_loop,
                                        name="fake-server", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        with self._lock:
            clients = list(self._clients)
        for conn in clients:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __enter__(self) -> "FakeServer":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def _record(self, message: str) -> None:
        with self._lock:
            self.errors.append(message)

    # -- 连接处理 ----------------------------------------------------------
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except OSError:
                break
            with self._lock:
                self._clients.append(conn)
            threading.Thread(target=self._handle, args=(conn,),
                             name="fake-conn", daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(30.0)
        reader = protocol.FrameReader(conn)
        writer = protocol.FrameWriter(conn)
        authed = False
        job: Optional[Dict[str, Any]] = None
        try:
            while not self._stop.is_set():
                try:
                    msg = reader.read_json()
                except (ConnectionError, OSError, protocol.ProtocolError) as exc:
                    self._record(f"读取消息失败: {exc}")
                    return
                kind = msg.get("kind")
                if kind == protocol.KIND_HELLO:
                    if msg.get("protocol") != protocol.PROTOCOL_VERSION:
                        writer.send_json(protocol.make_error(
                            protocol.ERR_BAD_PROTOCOL, "协议版本不匹配"))
                        return
                    if msg.get("token") != self.token:
                        writer.send_json(protocol.make_error(
                            protocol.ERR_BAD_AUTH, "token 无效"))
                        return
                    authed = True
                    writer.send_json({
                        "kind": protocol.KIND_HELLO_OK,
                        "protocol": protocol.PROTOCOL_VERSION,
                        "server_id": "srv-fake-0001",
                        "hostname": "fake-local",
                        "platform": "windows",
                        "python": "3.12",
                        "max_frame": protocol.MAX_FRAME_PAYLOAD,
                        "gpus": [],
                        "max_parallel_jobs": 1,
                        "env_cache_dir": str(self.data_dir / "envs"),
                    })
                    continue
                if not authed:
                    writer.send_json(protocol.make_error(
                        protocol.ERR_BAD_AUTH, "未握手就发送消息"))
                    return
                if kind == protocol.KIND_PING:
                    writer.send_json({"kind": protocol.KIND_PONG, "ts": time.time()})
                elif kind == protocol.KIND_JOB_SUBMIT:
                    job = self._job_submit(writer, msg)
                elif kind == protocol.KIND_FILE_CHUNK:
                    self._file_chunk(reader, writer, msg, job)
                elif kind == protocol.KIND_FILE_DONE:
                    self._file_done(writer, msg, job)
                elif kind == protocol.KIND_JOB_START:
                    self._job_start(writer, msg, job)
                elif kind == protocol.KIND_JOB_STATUS:
                    self._job_status(writer, msg)
                elif kind == protocol.KIND_OUTPUT_LIST:
                    self._output_list(writer, msg)
                elif kind == protocol.KIND_OUTPUT_CHUNK:
                    self._output_chunk(writer, msg)
                elif kind == protocol.KIND_BYE:
                    return
                else:
                    writer.send_json(protocol.make_error(
                        protocol.ERR_BAD_REQUEST, f"未知消息 kind: {kind!r}"))
        finally:
            with self._lock:
                if conn in self._clients:
                    self._clients.remove(conn)
            try:
                conn.close()
            except OSError:
                pass

    # -- 作业辅助 ----------------------------------------------------------
    def _job_of(self, job_id: Any) -> Optional[Dict[str, Any]]:
        return self.jobs.get(str(job_id)) if job_id else None

    def _job_for(self, msg: Dict[str, Any],
                 job: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if job is not None:
            return job
        return self._job_of(msg.get("job_id"))

    @staticmethod
    def _safe_join(root: Path, relpath: str) -> Optional[Path]:
        rel = str(relpath).replace("\\", "/")
        parts = [p for p in rel.split("/") if p not in ("", ".")]
        if not parts or rel.startswith("/") or any(p == ".." for p in parts):
            return None
        return root.joinpath(*parts)

    # -- 消息处理 ----------------------------------------------------------
    def _job_submit(self, writer: protocol.FrameWriter, msg: Dict[str, Any]):
        try:
            spec = JobSpec.from_dict(msg["spec"])
        except (KeyError, TypeError, ValueError) as exc:
            writer.send_json(protocol.make_error(
                protocol.ERR_BAD_REQUEST, f"spec 非法: {exc}"))
            return None

        resume_id = msg.get("resume_job_id")
        job = self._job_of(resume_id)
        if job is None or job["state"] not in ("uploading", "created", "queued"):
            job_id = f"job-fake-{len(self.jobs) + 1:04d}"
            job_dir = self.data_dir / "jobs" / job_id
            inputs = job_dir / "inputs"
            inputs.mkdir(parents=True, exist_ok=True)
            job = {
                "job_id": job_id,
                "spec": spec,
                "dir": job_dir,
                "inputs": inputs,
                "completed": {},
                "state": "uploading",
                "outputs": [],
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
            }
            self.jobs[job_id] = job
        else:
            job["spec"] = spec
            job["state"] = "uploading"

        # 未完成文件按协议 3.3 从 0 重传：先清掉上一轮的半成品
        for entry in spec.inputs:
            if job["completed"].get(entry.path):
                continue
            target = self._safe_join(job["inputs"], entry.path)
            if target is not None and target.exists():
                try:
                    target.unlink()
                except OSError:
                    pass

        fingerprint = spec.env.fingerprint or spec.env.compute_fingerprint()
        if fingerprint in self.built_envs:
            env_state = "cached"
        else:
            env_state = "built"
            self.built_envs.add(fingerprint)
        job["env_state"] = env_state

        missing = [e.to_dict() for e in spec.inputs
                   if not job["completed"].get(e.path)]
        present = [e.path for e in spec.inputs if job["completed"].get(e.path)]
        writer.send_json({
            "kind": protocol.KIND_JOB_ACCEPTED,
            "job_id": job["job_id"],
            "missing": missing,
            "present": present,
            "env": {"fingerprint": fingerprint, "reuse": True, "state": env_state},
            "chunk_size": _FAKE_CHUNK,
        })
        return job

    def _file_chunk(self, reader: protocol.FrameReader,
                    writer: protocol.FrameWriter, msg: Dict[str, Any],
                    job: Optional[Dict[str, Any]]) -> None:
        blob = reader.read_blob()
        job = self._job_for(msg, job)
        if job is None:
            writer.send_json(protocol.make_error(protocol.ERR_NOT_FOUND, "作业不存在"))
            return
        path = str(msg.get("path") or "")
        target = self._safe_join(job["inputs"], path)
        if target is None:
            writer.send_json(protocol.make_error(
                protocol.ERR_BAD_REQUEST, f"非法文件路径: {path!r}"))
            return
        try:
            offset = int(msg["offset"])
            length = int(msg["length"])
        except (KeyError, TypeError, ValueError):
            writer.send_json(protocol.make_error(
                protocol.ERR_BAD_REQUEST, "file_chunk 缺少 offset/length"))
            return
        if len(blob) != length:
            writer.send_json(protocol.make_error(
                protocol.ERR_BAD_REQUEST,
                f"分片长度不符: 头 {length}，实际 {len(blob)}"))
            return
        if hashlib.sha256(blob).hexdigest() != str(msg.get("sha256")):
            writer.send_json(protocol.make_error(
                protocol.ERR_HASH_MISMATCH, f"分片 sha256 不符: {path} offset={offset}"))
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+b" if target.exists() else "wb"
        with open(target, mode) as fh:
            fh.seek(offset)
            fh.write(blob)
            fh.flush()
        writer.send_json({"kind": protocol.KIND_FILE_OK, "path": path,
                          "offset": offset + length})

    def _file_done(self, writer: protocol.FrameWriter, msg: Dict[str, Any],
                   job: Optional[Dict[str, Any]]) -> None:
        job = self._job_for(msg, job)
        if job is None:
            writer.send_json(protocol.make_error(protocol.ERR_NOT_FOUND, "作业不存在"))
            return
        spec: JobSpec = job["spec"]
        path = str(msg.get("path") or "")
        entry = spec.input_index().get(path)
        target = self._safe_join(job["inputs"], path)
        if entry is None or target is None:
            writer.send_json(protocol.make_error(
                protocol.ERR_NOT_FOUND, f"文件不在任务清单: {path!r}"))
            return
        if not target.is_file():
            writer.send_json(protocol.make_error(
                protocol.ERR_BAD_REQUEST, f"文件尚未上传: {path}"))
            return
        actual = sha256_file(target)
        if actual != entry.sha256:
            try:
                target.unlink()
            except OSError:
                pass
            writer.send_json(protocol.make_error(
                protocol.ERR_HASH_MISMATCH,
                f"整文件 sha256 不符: {path}（期望 {entry.sha256[:12]}…，"
                f"实际 {actual[:12]}…）"))
            return
        job["completed"][path] = True
        writer.send_json({"kind": protocol.KIND_FILE_OK, "path": path, "done": True})

    def _job_start(self, writer: protocol.FrameWriter, msg: Dict[str, Any],
                   job: Optional[Dict[str, Any]]) -> None:
        job = self._job_for(msg, job)
        if job is None:
            writer.send_json(protocol.make_error(protocol.ERR_NOT_FOUND, "作业不存在"))
            return
        job_id = job["job_id"]
        if job["state"] == "succeeded":
            self._send_finished(writer, job)
            return
        job["state"] = "running"
        job["started_at"] = time.time()
        writer.send_json({
            "kind": protocol.KIND_JOB_STARTED,
            "job_id": job_id,
            "env": {"fingerprint": job["spec"].env.fingerprint,
                    "state": job.get("env_state", "built")},
        })
        for index in (1, 2):
            writer.send_json({
                "kind": protocol.KIND_JOB_EVENT, "job_id": job_id,
                "seq": index, "stream": "stdout",
                "data": f"[train] epoch {index}/2 loss=0.100000\n",
                "ts": time.time(),
            })
            time.sleep(0.005)

        out_dir = job["inputs"] / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        payloads = {
            "outputs/model.json": b'{"runs": 2, "epochs_total": 4, "w": 1.0, "b": 0.5}\n',
            "outputs/metrics.json": b'{"mode": "incremental_training", "runs": 2}\n',
        }
        manifest = []
        for rel, data in payloads.items():
            target = self._safe_join(job["inputs"], rel)
            assert target is not None
            target.write_bytes(data)
            manifest.append({"path": rel, "size": len(data),
                             "sha256": sha256_bytes(data), "kind": "file"})
        job["outputs"] = manifest
        job["state"] = "succeeded"
        job["finished_at"] = time.time()
        self._send_finished(writer, job)

    @staticmethod
    def _send_finished(writer: protocol.FrameWriter,
                       job: Dict[str, Any]) -> None:
        started = job.get("started_at") or 0.0
        finished = job.get("finished_at") or time.time()
        writer.send_json({
            "kind": protocol.KIND_JOB_FINISHED,
            "job_id": job["job_id"],
            "state": job["state"],
            "exit_code": 0 if job["state"] == "succeeded" else -1,
            "duration_s": max(0.0, finished - started),
            "outputs": job["outputs"],
            "log_tail": ["[train] 完成"],
        })

    def _job_status(self, writer: protocol.FrameWriter, msg: Dict[str, Any]) -> None:
        job = self._job_of(msg.get("job_id"))
        if job is None:
            writer.send_json(protocol.make_error(protocol.ERR_NOT_FOUND, "作业不存在"))
            return
        writer.send_json({
            "kind": protocol.KIND_JOB_STATUS_OK,
            "job_id": job["job_id"],
            "state": job["state"],
            "progress": 1.0 if job["state"] == "succeeded" else 0.5,
            "outputs": job["outputs"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
        })

    def _output_list(self, writer: protocol.FrameWriter, msg: Dict[str, Any]) -> None:
        job = self._job_of(msg.get("job_id"))
        if job is None:
            writer.send_json(protocol.make_error(protocol.ERR_NOT_FOUND, "作业不存在"))
            return
        writer.send_json({"kind": protocol.KIND_OUTPUT_LIST_OK,
                          "job_id": job["job_id"], "outputs": job["outputs"]})

    def _output_chunk(self, writer: protocol.FrameWriter, msg: Dict[str, Any]) -> None:
        job = self._job_of(msg.get("job_id"))
        if job is None:
            writer.send_json(protocol.make_error(protocol.ERR_NOT_FOUND, "作业不存在"))
            return
        path = str(msg.get("path") or "")
        target = self._safe_join(job["inputs"], path)
        if target is None or not target.is_file():
            writer.send_json(protocol.make_error(
                protocol.ERR_NOT_FOUND, f"产物不存在: {path!r}"))
            return
        size = target.stat().st_size
        try:
            offset = int(msg.get("offset") or 0)
        except (TypeError, ValueError):
            offset = 0
        if offset < 0 or offset >= size:
            writer.send_json(protocol.make_error(
                protocol.ERR_BAD_REQUEST, f"offset 越界: {offset} (size={size})"))
            return
        length = min(_FAKE_CHUNK, size - offset)
        with open(target, "rb") as fh:
            fh.seek(offset)
            blob = fh.read(length)
        writer.send_json({
            "kind": protocol.KIND_OUTPUT_CHUNK,
            "job_id": job["job_id"],
            "path": path,
            "offset": offset,
            "length": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "eof": offset + len(blob) >= size,
        })
        writer.send_blob(blob)
