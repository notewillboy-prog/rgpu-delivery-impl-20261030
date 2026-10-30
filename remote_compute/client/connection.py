"""客户端连接层：TCP 帧收发、握手鉴权、心跳、超时、断线重连。

对应 ``docs/PROTOCOL.md`` 第 1、2 节：

- 帧格式由 ``common.protocol`` 实现（4B 大端长度 + 1B 类型 + payload）；
- 连接建立后第一条消息必须是 ``hello``，收到 ``hello_ok`` 之前不发任何其他消息；
- 客户端每 20 s 发 ``ping``，服务端回 ``pong``（读取时自动跳过）；
- socket 有连接超时 / IO 超时；
- 断线后 ``reconnect()`` 有限次重连，配合 runner 的 ``resume_job_id`` 续传。

线程模型：一个后台心跳线程只负责发 ``ping``；所有 socket 写入都经过同一把
可重入锁，保证「JSON 头 + BLOB」两帧之间不会插入心跳帧。读取只发生在调用方线程。
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from common import protocol
from common.config import ClientConfig

from . import __version__

CLIENT_VERSION = __version__
HEARTBEAT_INTERVAL_S = 20.0
MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_BASE_DELAY_S = 0.5

#: 客户端侧错误码（服务端错误码见 ``common.protocol``）。
ERR_CLIENT_CONNECTION = "CLIENT_CONNECTION"
ERR_CLIENT_PROTOCOL = "CLIENT_PROTOCOL"


class ClientError(Exception):
    """客户端异常基类；``exit_code`` 供 CLI 映射（0/1/2/3/4）。"""

    exit_code = 4
    code = "CLIENT_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = str(message)

    def __str__(self) -> str:
        return self.message


class ConnectFailed(ClientError):
    """建立连接/握手失败（可重试）。退出码 3。"""

    exit_code = 3
    code = ERR_CLIENT_CONNECTION


class ConnectionLost(ClientError):
    """会话中途断开或读取失败（可重连）。退出码 3。"""

    exit_code = 3
    code = ERR_CLIENT_CONNECTION


class RemoteError(ClientError):
    """服务端返回 ``error`` 消息。默认退出码 4；服务端超时映射为 1。"""

    exit_code = 4

    def __init__(self, code: str, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = str(code or protocol.ERR_INTERNAL)
        self.detail = dict(detail or {})

    @property
    def exit_code(self) -> int:  # type: ignore[override]
        if self.code == protocol.ERR_TIMEOUT:
            return 1
        return 4

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


MessageCallback = Callable[[dict], None]


class Connection:
    """一条到 RGPU 服务端的 TCP 会话（可重连）。"""

    def __init__(self, config: ClientConfig, client_version: str = CLIENT_VERSION):
        self.config = config
        self.client_version = client_version
        self.sock: Optional[socket.socket] = None
        self.reader: Optional[protocol.FrameReader] = None
        self.writer: Optional[protocol.FrameWriter] = None
        self.server_info: Dict[str, Any] = {}
        self.max_frame: int = protocol.MAX_FRAME_PAYLOAD
        self.connected = False

        self._send_lock = threading.RLock()
        self._hb_stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_broken = False

    # -- 连接生命周期 ------------------------------------------------------
    def connect(self) -> Dict[str, Any]:
        """建连并完成 hello 握手，返回 ``hello_ok`` 消息。"""
        self._open_socket()
        info = self._handshake()
        self._start_heartbeat()
        return info

    def reconnect(
        self,
        attempts: int = MAX_RECONNECT_ATTEMPTS,
        base_delay: float = RECONNECT_BASE_DELAY_S,
    ) -> Dict[str, Any]:
        """断开后按退避策略有限次重连；失败抛 ``ConnectFailed``。"""
        last_exc: Optional[ClientError] = None
        attempts = max(1, int(attempts))
        for attempt in range(attempts):
            if attempt:
                time.sleep(base_delay * (2 ** (attempt - 1)))
            self._stop_heartbeat()
            self._close_socket()
            try:
                self._open_socket()
                info = self._handshake()
                self._start_heartbeat()
                return info
            except ConnectFailed as exc:
                last_exc = exc
        raise ConnectFailed(
            f"重连失败（已尝试 {attempts} 次）: {last_exc.message if last_exc else 'unknown'}"
        )

    def close(self, send_bye: bool = True) -> None:
        """关闭会话（先停心跳；可选发 bye）。幂等。"""
        self._stop_heartbeat()
        if send_bye and self.connected and self.writer is not None:
            try:
                self.send_json({"kind": protocol.KIND_BYE})
            except ClientError:
                pass
        self._close_socket()

    def __enter__(self) -> "Connection":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- socket 与握手 ------------------------------------------------------
    def _open_socket(self) -> None:
        self._close_socket()
        try:
            sock = socket.create_connection(
                (self.config.host, self.config.port),
                timeout=float(self.config.connect_timeout),
            )
        except OSError as exc:
            raise ConnectFailed(f"无法连接 {self.config.address}: {exc}") from exc
        sock.settimeout(float(self.config.io_timeout))
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self.sock = sock
        self.reader = protocol.FrameReader(sock)
        self.writer = protocol.FrameWriter(sock)
        self.connected = True

    def _handshake(self) -> Dict[str, Any]:
        hello = {
            "kind": protocol.KIND_HELLO,
            "protocol": protocol.PROTOCOL_VERSION,
            "role": "client",
            "client_version": self.client_version,
            "token": self.config.token,
        }
        self.send_json(hello)
        try:
            msg = self.recv_json()
        except RemoteError:
            raise
        except ConnectionLost as exc:
            raise ConnectFailed(f"握手时连接断开: {exc.message}") from exc
        kind = msg.get("kind")
        if kind != protocol.KIND_HELLO_OK:
            raise ConnectFailed(f"握手失败：期望 {protocol.KIND_HELLO_OK}，实际收到 {kind!r}")
        try:
            protocol.check_version(msg)
        except protocol.ProtocolError as exc:
            raise RemoteError(exc.code, f"协议版本不兼容: {exc.message}") from exc
        self.server_info = msg
        try:
            self.max_frame = int(msg.get("max_frame") or protocol.MAX_FRAME_PAYLOAD)
        except (TypeError, ValueError):
            self.max_frame = protocol.MAX_FRAME_PAYLOAD
        return msg

    # -- 心跳 --------------------------------------------------------------
    def _start_heartbeat(self) -> None:
        self._hb_stop.clear()
        self._hb_broken = False
        thread = threading.Thread(
            target=self._heartbeat_loop, name="rgpu-heartbeat", daemon=True
        )
        self._hb_thread = thread
        thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.wait(HEARTBEAT_INTERVAL_S):
            try:
                self.send_json({"kind": protocol.KIND_PING, "ts": time.time()})
            except ClientError:
                self._hb_broken = True
                return

    def _stop_heartbeat(self) -> None:
        self._hb_stop.set()
        thread = self._hb_thread
        self._hb_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._hb_stop.clear()

    def _close_socket(self) -> None:
        sock, self.sock = self.sock, None
        self.reader = None
        self.writer = None
        self.connected = False
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    # -- 发送 --------------------------------------------------------------
    def _send(self, method_name: str, payload: Any) -> None:
        with self._send_lock:
            writer = self.writer
            if writer is None:
                raise ConnectionLost("连接尚未建立或已关闭")
            try:
                getattr(writer, method_name)(payload)
            except protocol.ProtocolError as exc:
                raise ConnectionLost(f"协议层发送失败: {exc.message}") from exc
            except (ConnectionError, OSError) as exc:
                self.connected = False
                raise ConnectionLost(f"发送失败（连接可能已断开）: {exc}") from exc

    def send_json(self, obj: Dict[str, Any]) -> None:
        self._send("send_json", obj)

    def send_blob(self, data: bytes) -> None:
        self._send("send_blob", data)

    def send_chunk(self, header: Dict[str, Any], payload: bytes) -> None:
        """原子地发送「JSON 头 + BLOB 帧」，心跳帧不会插到两者之间。"""
        if self.writer is None:
            raise ConnectionLost("连接尚未建立或已关闭")
        with self._send_lock:
            try:
                self.writer.send_json(header)
                self.writer.send_blob(payload)
            except protocol.ProtocolError as exc:
                raise ConnectionLost(f"协议层发送失败: {exc.message}") from exc
            except (ConnectionError, OSError) as exc:
                self.connected = False
                raise ConnectionLost(f"发送失败（连接可能已断开）: {exc}") from exc

    # -- 接收 --------------------------------------------------------------
    def recv_json(self, skip_pong: bool = True) -> Dict[str, Any]:
        """读一条 JSON 消息；跳过 ``pong``；``error`` 抛 ``RemoteError``。"""
        if self.reader is None:
            raise ConnectionLost("连接尚未建立或已关闭")
        while True:
            try:
                msg = self.reader.read_json()
            except protocol.ProtocolError as exc:
                if exc.code == protocol.ERR_TIMEOUT:
                    raise ConnectionLost(f"等待服务端响应超时: {exc.message}") from exc
                raise ConnectionLost(f"读取协议错误: {exc.message}") from exc
            except ConnectionError as exc:
                self.connected = False
                raise ConnectionLost("连接已被服务端关闭") from exc
            except OSError as exc:
                self.connected = False
                raise ConnectionLost(f"读取失败（连接可能已断开）: {exc}") from exc
            kind = msg.get("kind")
            if skip_pong and kind == protocol.KIND_PONG:
                continue
            if kind == protocol.KIND_ERROR:
                raise RemoteError(
                    str(msg.get("code") or protocol.ERR_INTERNAL),
                    str(msg.get("message") or "服务端返回错误"),
                    msg.get("detail") if isinstance(msg.get("detail"), dict) else None,
                )
            return msg

    def recv_blob(self) -> bytes:
        """读取紧跟 JSON 头的 BLOB 分片帧。"""
        if self.reader is None:
            raise ConnectionLost("连接尚未建立或已关闭")
        try:
            return self.reader.read_blob()
        except protocol.ProtocolError as exc:
            if exc.code == protocol.ERR_TIMEOUT:
                raise ConnectionLost(f"等待分片数据超时: {exc.message}") from exc
            raise ConnectionLost(f"读取分片失败: {exc.message}") from exc
        except ConnectionError as exc:
            self.connected = False
            raise ConnectionLost("连接已被服务端关闭") from exc
        except OSError as exc:
            self.connected = False
            raise ConnectionLost(f"读取分片失败（连接可能已断开）: {exc}") from exc

    def wait_for(
        self,
        kinds: Iterable[str],
        on_message: Optional[MessageCallback] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """持续读取直到出现 ``kinds`` 之一；期间其他消息交给 ``on_message``。

        ``on_message`` 会收到除 pong 之外的所有消息（包括 ``job_event`` /
        ``job_progress``）。``error`` 消息在 ``recv_json`` 中已经转成异常。
        """
        wanted: Set[str] = {str(k) for k in kinds}
        deadline = time.monotonic() + timeout_s if timeout_s else None
        while True:
            msg = self.recv_json()
            kind = msg.get("kind")
            if kind in wanted:
                return msg
            if on_message is not None:
                on_message(msg)
            if deadline is not None and time.monotonic() > deadline:
                raise ConnectionLost(f"等待消息 {sorted(wanted)} 超时")

    def list_gpus(self) -> List[Dict[str, Any]]:
        gpus = self.server_info.get("gpus") or []
        return [g for g in gpus if isinstance(g, dict)]
