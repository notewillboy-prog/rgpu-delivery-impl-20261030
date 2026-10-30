"""帧协议：TCP 上传输 JSON 控制消息与二进制数据分片。

帧格式（大端）::

    +----------------+------------+------------------+
    | payload_len u32| msg_type u8| payload (bytes)  |
    +----------------+------------+------------------+

``payload_len`` 只统计 payload 本身，不含 5 字节帧头。

- ``MSG_JSON``：payload 是 UTF-8 编码的 JSON 对象（必须是 dict），字段 ``kind`` 区分消息类型。
- ``MSG_BLOB``：payload 是原始二进制分片，语义由紧挨着的前一条 ``file_chunk`` /
  ``output_chunk`` JSON 消息确定（``offset`` / ``length`` / ``sha256``）。

单帧上限 ``MAX_FRAME_PAYLOAD``；文件分片建议 ``DEFAULT_CHUNK_SIZE``（1 MiB），
上限 ``MAX_CHUNK_SIZE``。这样任何大小的文件都能传，且断点续传只需比对 offset。
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, Optional

PROTOCOL_VERSION = 1

MSG_JSON = 1
MSG_BLOB = 2

MAX_FRAME_PAYLOAD = 16 * 1024 * 1024  # 16 MiB
DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1 MiB
MAX_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB

_FRAME_HEADER = struct.Struct("!IB")  # payload_len, msg_type
_FRAME_HEADER_SIZE = _FRAME_HEADER.size  # 5

# ---- 消息 kind（客户端 → 服务端 / 服务端 → 客户端） --------------------------
# 握手
KIND_HELLO = "hello"
KIND_HELLO_OK = "hello_ok"
KIND_PING = "ping"
KIND_PONG = "pong"
KIND_ERROR = "error"
KIND_BYE = "bye"

# 任务提交与上传
KIND_JOB_SUBMIT = "job_submit"
KIND_JOB_ACCEPTED = "job_accepted"      # 服务端返回 job_id + 还需要上传哪些文件
KIND_FILE_CHUNK = "file_chunk"          # JSON 头，后面紧跟一个 blob 帧
KIND_FILE_DONE = "file_done"            # 单文件上传完成
KIND_FILE_OK = "file_ok"
KIND_UPLOAD_STATE = "upload_state"      # 续传：服务端告诉客户端已有分片
KIND_JOB_START = "job_start"
KIND_JOB_STARTED = "job_started"

# 执行期
KIND_JOB_EVENT = "job_event"            # {seq, stream, data, ts}
KIND_JOB_PROGRESS = "job_progress"      # {state, progress, message}
KIND_JOB_FINISHED = "job_finished"      # {exit_code, outputs, duration_s, state}
KIND_JOB_STATUS = "job_status"          # 查询
KIND_JOB_STATUS_OK = "job_status_ok"

# 产物下载
KIND_OUTPUT_LIST = "output_list"
KIND_OUTPUT_LIST_OK = "output_list_ok"
KIND_OUTPUT_CHUNK = "output_chunk"      # JSON 头，后面紧跟一个 blob 帧
KIND_OUTPUT_END = "output_end"
KIND_OUTPUT_ACK = "output_ack"

# ---- 错误码 ---------------------------------------------------------------
ERR_BAD_PROTOCOL = "BAD_PROTOCOL"
ERR_BAD_REQUEST = "BAD_REQUEST"
ERR_BAD_AUTH = "BAD_AUTH"
ERR_NOT_FOUND = "NOT_FOUND"
ERR_CONFLICT = "CONFLICT"
ERR_HASH_MISMATCH = "HASH_MISMATCH"
ERR_ENV_BUILD = "ENV_BUILD_FAILED"
ERR_JOB_FAILED = "JOB_FAILED"
ERR_TIMEOUT = "TIMEOUT"
ERR_BUSY = "BUSY"
ERR_INTERNAL = "INTERNAL"

ERROR_CODES = {
    ERR_BAD_PROTOCOL: "协议版本或帧格式错误",
    ERR_BAD_REQUEST: "请求字段缺失或非法",
    ERR_BAD_AUTH: "鉴权失败",
    ERR_NOT_FOUND: "任务/文件/产物不存在",
    ERR_CONFLICT: "资源冲突（重复提交、状态不允许等）",
    ERR_HASH_MISMATCH: "文件哈希不匹配",
    ERR_ENV_BUILD: "远端环境构建失败",
    ERR_JOB_FAILED: "作业执行失败",
    ERR_TIMEOUT: "操作超时",
    ERR_BUSY: "服务端繁忙",
    ERR_INTERNAL: "服务端内部错误",
}


class ProtocolError(Exception):
    """帧/消息层错误；``code`` 使用上方错误码。"""

    def __init__(self, message: str, code: str = ERR_BAD_PROTOCOL):
        super().__init__(message)
        self.code = code
        self.message = message


def encode_frame(msg_type: int, payload: bytes = b"") -> bytes:
    """把 payload 打成完整帧（帧头 + 数据）。"""
    if msg_type not in (MSG_JSON, MSG_BLOB):
        raise ValueError(f"unknown msg_type: {msg_type}")
    if len(payload) > MAX_FRAME_PAYLOAD:
        raise ValueError(
            f"payload too large: {len(payload)} > {MAX_FRAME_PAYLOAD}"
        )
    return _FRAME_HEADER.pack(len(payload), msg_type) + payload


def decode_json_payload(payload: bytes) -> dict:
    """校验并解析 JSON 帧 payload。"""
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON payload: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("JSON payload must be an object")
    return obj


class FrameReader:
    """从 socket 读取帧，内部带缓冲，兼容大文件分片的连续读取。"""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._buf = bytearray()

    def _read_exact(self, need: int) -> bytes:
        while len(self._buf) < need:
            try:
                chunk = self._sock.recv(max(65536, need - len(self._buf)))
            except socket.timeout as exc:
                raise ProtocolError("socket timeout while reading frame", ERR_TIMEOUT) from exc
            except OSError as exc:
                raise ProtocolError(f"socket read failed: {exc}") from exc
            if not chunk:
                raise ConnectionError("connection closed by peer")
            self._buf.extend(chunk)
        out = bytes(self._buf[:need])
        del self._buf[:need]
        return out

    def read_frame(self) -> tuple[int, bytes]:
        """返回 ``(msg_type, payload)``；连接断开抛 ``ConnectionError``。"""
        header = self._read_exact(_FRAME_HEADER_SIZE)
        payload_len, msg_type = _FRAME_HEADER.unpack(header)
        if msg_type not in (MSG_JSON, MSG_BLOB):
            raise ProtocolError(f"unknown frame type: {msg_type}")
        if payload_len > MAX_FRAME_PAYLOAD:
            raise ProtocolError(
                f"frame too large: {payload_len} > {MAX_FRAME_PAYLOAD}", ERR_BAD_PROTOCOL
            )
        payload = self._read_exact(payload_len) if payload_len else b""
        return msg_type, payload

    def read_json(self) -> dict:
        msg_type, payload = self.read_frame()
        if msg_type != MSG_JSON:
            raise ProtocolError("expected JSON frame")
        return decode_json_payload(payload)

    def read_blob(self) -> bytes:
        msg_type, payload = self.read_frame()
        if msg_type != MSG_BLOB:
            raise ProtocolError("expected blob frame")
        return payload


class FrameWriter:
    """封装发送端，保证同一 socket 上的帧写入是原子的。"""

    def __init__(self, sock: socket.socket):
        self._sock = sock

    def send_json(self, obj: dict) -> None:
        payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send(MSG_JSON, payload)

    def send_blob(self, data: bytes) -> None:
        self._send(MSG_BLOB, bytes(data))

    def _send(self, msg_type: int, payload: bytes) -> None:
        if len(payload) > MAX_FRAME_PAYLOAD:
            raise ProtocolError(f"payload too large: {len(payload)}", ERR_BAD_REQUEST)
        try:
            self._sock.sendall(encode_frame(msg_type, payload))
        except OSError as exc:
            raise ConnectionError(f"socket write failed: {exc}") from exc


def send_json(sock: socket.socket, obj: dict) -> None:
    """无状态便捷函数（单线程发送场景）；并发场景用 ``FrameWriter`` 并加锁。"""
    FrameWriter(sock).send_json(obj)


def recv_json(reader: FrameReader) -> dict:
    return reader.read_json()


def send_blob(sock: socket.socket, data: bytes) -> None:
    FrameWriter(sock).send_blob(data)


def make_error(code: str, message: str, **detail: Any) -> dict:
    """构造 ``error`` 消息。"""
    msg = {"kind": KIND_ERROR, "code": code, "message": message}
    if detail:
        msg["detail"] = detail
    return msg


def is_error(msg: dict) -> bool:
    return msg.get("kind") == KIND_ERROR


def require_fields(msg: dict, *fields: str) -> None:
    """字段校验，缺字段抛 ``ProtocolError(ERR_BAD_REQUEST)``。"""
    missing = [f for f in fields if f not in msg]
    if missing:
        raise ProtocolError(f"missing fields: {', '.join(missing)}", ERR_BAD_REQUEST)


def check_version(msg: dict) -> None:
    version = msg.get("protocol")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol version mismatch: client={version} server={PROTOCOL_VERSION}",
            ERR_BAD_PROTOCOL,
        )


def iter_chunks(total_size: int, chunk_size: int = DEFAULT_CHUNK_SIZE):
    """产出 ``(offset, length)``，用于上传/下载同一套分片循环。"""
    if chunk_size <= 0 or chunk_size > MAX_CHUNK_SIZE:
        raise ValueError(f"chunk_size out of range: {chunk_size}")
    offset = 0
    while offset < total_size:
        length = min(chunk_size, total_size - offset)
        yield offset, length
        offset += length


def choose_chunk_size(size: int, preferred: int = DEFAULT_CHUNK_SIZE) -> int:
    """小文件不切分，避免无谓帧开销。"""
    if size <= preferred:
        return max(1, size)
    return preferred


def recv_json_or_none(reader: FrameReader, timeout: Optional[float] = None):
    """读取 JSON；``timeout`` 为 None 时用 socket 自身超时。"""
    _ = timeout
    return reader.read_json()
