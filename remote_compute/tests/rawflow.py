"""原始协议流程封装（严格按 docs/PROTOCOL.md v1 的帧级语义）。

用途：
- ``tests/test_e2e_local.py`` 对真服务端跑哈希篡改 / 断线续传场景；
- ``tests/test_raw_protocol_fake.py`` 在真 server/ 落地前，先用
  ``tests/fake_server.py``（协议参考实现）验证这套流程自身正确。

所有函数在发现协议违例时抛 ``AssertionError``（附带原始消息），便于定位是
「测试逻辑错」还是「服务端实现错」。
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import helpers


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def connect(address: str, token: str = "e2e-token", *, timeout: float = 30.0):
    """建立原始连接并完成 hello 握手，返回 ``(client, hello_ok)``。"""
    client = helpers.RawClient(address, timeout=timeout, token=token)
    client.connect()
    hello = client.hello()
    _require(hello.get("kind") == "hello_ok", f"握手失败: {hello!r}")
    return client, hello


def submit(client: helpers.RawClient, spec, resume_job_id: Optional[str] = None) -> dict:
    """发送 job_submit 并返回 job_accepted。"""
    client.send_json({
        "kind": "job_submit",
        "spec": spec.to_dict(),
        "resume_job_id": resume_job_id,
    })
    msg = client.recv_json()
    _require(msg.get("kind") == "job_accepted",
             f"job_submit 未返回 job_accepted: {msg!r}")
    return msg


def missing_paths(accepted: dict) -> List[str]:
    return [str(item.get("path") if isinstance(item, dict) else item)
            for item in accepted.get("missing") or []]


def present_paths(accepted: dict) -> List[str]:
    return [str(item.get("path") if isinstance(item, dict) else item)
            for item in accepted.get("present") or []]


def chunk_size_of(accepted: dict, default: int = 1024 * 1024) -> int:
    try:
        return int(accepted.get("chunk_size") or default)
    except (TypeError, ValueError):
        return default


def upload_entry(client: helpers.RawClient, entry, project_dir: Path,
                 chunk_size: int = 1024 * 1024) -> dict:
    """按 PROTOCOL 3.3 上传一个文件；返回最后一条消息（正常是 file_ok done=true，
    失败是 ``{"kind":"error", ...}``，由调用方断言具体错误码）。"""
    data = (Path(project_dir) / entry.path).read_bytes()
    _require(len(data) == entry.size,
             f"{entry.path} 本地大小 {len(data)} 与清单 {entry.size} 不符")
    offset = 0
    while offset < len(data):
        chunk = data[offset:offset + chunk_size]
        client.send_json({
            "kind": "file_chunk",
            "path": entry.path,
            "offset": offset,
            "length": len(chunk),
            "sha256": hashlib.sha256(chunk).hexdigest(),
        })
        client.send_blob(chunk)
        resp = client.recv_json()
        if resp.get("kind") == "error":
            return resp
        _require(resp.get("kind") == "file_ok", f"分片未被确认: {resp!r}")
        expected = offset + len(chunk)
        _require(int(resp.get("offset", -1)) == expected,
                 f"file_ok.offset 期望 {expected}，实际 {resp!r}")
        offset = expected
    client.send_json({"kind": "file_done", "path": entry.path})
    return client.recv_json()


def wait_finished(client: helpers.RawClient, job_id: str, *,
                  timeout: float = 180.0) -> dict:
    """发送 job_start，收集 job_started / job_event 直到 job_finished。

    返回 job_finished 消息，并附加 ``_started``（job_started）与 ``_events``（日志拼接）。
    """
    client.send_json({"kind": "job_start", "job_id": job_id})
    events: List[str] = []
    started: Optional[dict] = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = client.recv_json()
        kind = msg.get("kind")
        if kind == "job_started":
            started = msg
        elif kind == "job_event":
            events.append(str(msg.get("data") or ""))
        elif kind == "job_progress":
            continue
        elif kind == "error":
            raise AssertionError(f"作业执行期间收到错误: {msg!r}")
        elif kind == "job_finished":
            msg["_started"] = started
            msg["_events"] = "".join(events)
            return msg
        else:
            raise AssertionError(f"作业执行期间收到未知消息: {msg!r}")
    raise AssertionError(f"等待 job_finished 超时（{timeout}s），job_id={job_id}")


def query_status(client: helpers.RawClient, job_id: str) -> dict:
    client.send_json({"kind": "job_status", "job_id": job_id})
    msg = client.recv_json()
    _require(msg.get("kind") == "job_status_ok", f"job_status 未返回 ok: {msg!r}")
    return msg


def download_outputs(client: helpers.RawClient, job_id: str,
                     manifest: Optional[List[dict]], dest_root: Path) -> Dict[str, dict]:
    """按 PROTOCOL 3.6 下载全部产物并逐片/整文件校验 sha256。

    ``manifest`` 为 job_finished.outputs（可为空），用于与 output_list 交叉校验。
    返回 ``{path: {"size":..., "sha256":...}}``。
    """
    client.send_json({"kind": "output_list", "job_id": job_id})
    listed = client.recv_json()
    _require(listed.get("kind") == "output_list_ok", f"output_list 失败: {listed!r}")
    files = [item for item in listed.get("outputs") or []
             if isinstance(item, dict) and str(item.get("kind", "file")) == "file"]
    manifest_by_path = {str(item.get("path")): item
                        for item in (manifest or []) if isinstance(item, dict)}
    dest_root = Path(dest_root)
    downloaded: Dict[str, dict] = {}
    for item in files:
        path = str(item["path"])
        size = int(item["size"])
        expected_sha = str(item["sha256"])
        parts: List[bytes] = []
        offset = 0
        while offset < size:
            client.send_json({"kind": "output_chunk", "job_id": job_id,
                              "path": path, "offset": offset})
            head = client.recv_json()
            _require(head.get("kind") == "output_chunk", f"产物分片头异常: {head!r}")
            blob = client.recv_blob()
            length = int(head.get("length", -1))
            _require(len(blob) == length,
                     f"{path} 分片长度不符: 头 {length}，实际 {len(blob)}")
            chunk_sha = str(head.get("sha256") or "")
            if chunk_sha:
                _require(hashlib.sha256(blob).hexdigest() == chunk_sha,
                         f"{path} offset={offset} 分片 sha256 不符")
            parts.append(blob)
            offset += len(blob)
            if head.get("eof"):
                break
        data = b"".join(parts)
        _require(len(data) == size, f"{path} 产物大小不符: 期望 {size}，实际 {len(data)}")
        actual_sha = hashlib.sha256(data).hexdigest()
        _require(actual_sha == expected_sha, f"{path} 整文件 sha256 不符")
        if manifest_by_path:
            _require(path in manifest_by_path, f"{path} 不在 job_finished 清单中")
            _require(expected_sha == manifest_by_path[path].get("sha256"),
                     f"{path} 与 job_finished 清单 sha256 不一致")
        local = dest_root.joinpath(*path.split("/"))
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        downloaded[path] = {"size": size, "sha256": expected_sha}
    return downloaded
