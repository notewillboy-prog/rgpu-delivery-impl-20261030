"""分片上传：按服务端认可的 chunk_size 切片，逐片 sha256 校验、有限重传。

对应 ``docs/PROTOCOL.md`` 3.3：

1. 每个分片先发 ``file_chunk`` JSON 头（含 offset/length/sha256），紧跟一个 BLOB 帧；
2. 服务端每片回 ``file_ok``，``offset`` 表示下一个期望偏移（客户端据此推进/回退）；
3. 分片 sha256 不匹配 → 服务端 ``error HASH_MISMATCH`` → 客户端重发该分片（有限次）；
4. 所有分片发完发 ``file_done``，整文件 sha256 通过后服务端回 ``file_ok done=true``；
5. 整文件哈希不匹配 → 客户端从 0 重传整个文件（有限次）。

连接中途断开抛 ``connection.ConnectionLost``，由 ``runner`` 负责重连 +
``resume_job_id`` 续传（未完成文件按整文件重传，符合协议 3.3 的 demo 约定）。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from common import protocol
from common.hashing import sha256_file
from common.models import FileEntry

from .connection import ClientError, Connection, RemoteError

#: 每个分片最多重发次数（哈希不符）。
MAX_CHUNK_RETRIES = 3
#: 整文件最多重传次数（整文件哈希不符）。
MAX_FILE_RETRIES = 2

#: (path, done_bytes, total_bytes)
FileProgressCallback = Callable[[str, int, int], None]
#: (done_bytes, total_bytes) 聚合进度
ProgressCallback = Callable[[int, int], None]
#: (entry) 开始/完成一个文件
FileCallback = Callable[[FileEntry], None]


class UploadError(ClientError):
    """上传失败（本地文件变化、重试耗尽等）。"""

    def __init__(self, path: str, message: str, exit_code: int = 4):
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path
        self.exit_code = exit_code


def _read_exact(fh, offset: int, length: int, path: str) -> bytes:
    fh.seek(offset)
    data = fh.read(length)
    if len(data) != length:
        raise UploadError(
            path,
            f"本地文件读取长度不符（期望 {length}，实际 {len(data)}），文件可能在打包后被修改",
            exit_code=2,
        )
    return data


def upload_file(
    conn: Connection,
    entry: FileEntry,
    project_dir: str | os.PathLike,
    *,
    chunk_size: int = protocol.DEFAULT_CHUNK_SIZE,
    on_progress: Optional[FileProgressCallback] = None,
    max_chunk_retries: int = MAX_CHUNK_RETRIES,
    max_file_retries: int = MAX_FILE_RETRIES,
    start_offsets: Optional[Dict[str, int]] = None,
) -> int:
    """上传单个文件，返回上传的字节数。失败抛 ``UploadError``/``ConnectionLost``。"""
    root = Path(project_dir).expanduser()
    full = root / Path(*entry.path.split("/"))
    if not full.is_file():
        raise UploadError(entry.path, "本地文件不存在", exit_code=2)
    local_size = os.path.getsize(full)
    if local_size != entry.size:
        raise UploadError(
            entry.path,
            f"本地文件大小已变化（清单 {entry.size} B，当前 {local_size} B），请重新提交",
            exit_code=2,
        )

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        chunk_size = protocol.DEFAULT_CHUNK_SIZE
    chunk_size = min(chunk_size, protocol.MAX_CHUNK_SIZE)
    # 保证 BLOB 帧本身不超过单帧上限（帧头 5B 不计入 payload_len）
    chunk_size = max(1, min(chunk_size, protocol.MAX_FRAME_PAYLOAD))

    progress = on_progress or (lambda _path, _done, _total: None)
    resume_from = 0
    if start_offsets:
        try:
            resume_from = int(start_offsets.get(entry.path, 0) or 0)
        except (TypeError, ValueError):
            resume_from = 0
    resume_from = max(0, min(resume_from, entry.size))

    for file_attempt in range(max_file_retries + 1):
        # 整文件重传时绝不沿用上次的 start_offset：从头开始。
        start_offset = resume_from if file_attempt == 0 else 0
        # 每次整文件重传前再校验一次内容，避免传一份已经变化的文件
        try:
            current_sha = sha256_file(full)
        except OSError as exc:
            raise UploadError(entry.path, f"无法读取本地文件: {exc}", exit_code=2) from exc
        if current_sha != entry.sha256:
            raise UploadError(
                entry.path,
                f"本地文件 sha256 与清单不一致（清单 {entry.sha256[:12]}…，当前 {current_sha[:12]}…），请重新提交",
                exit_code=2,
            )

        retry_whole_file = False
        with open(full, "rb") as fh:
            offset = start_offset
            progress(entry.path, offset, entry.size)
            while offset < entry.size:
                length = min(chunk_size, entry.size - offset)
                data = _read_exact(fh, offset, length, entry.path)
                chunk_sha = hashlib.sha256(data).hexdigest()
                header: Dict[str, object] = {
                    "kind": protocol.KIND_FILE_CHUNK,
                    "path": entry.path,
                    "offset": offset,
                    "length": length,
                    "sha256": chunk_sha,
                }
                for chunk_attempt in range(max_chunk_retries + 1):
                    conn.send_chunk(header, data)
                    try:
                        resp = conn.wait_for([protocol.KIND_FILE_OK])
                    except RemoteError as exc:
                        if exc.code == protocol.ERR_HASH_MISMATCH and chunk_attempt < max_chunk_retries:
                            continue  # 重发当前分片
                        if exc.code == protocol.ERR_HASH_MISMATCH:
                            raise UploadError(
                                entry.path,
                                f"分片 offset={offset} 哈希不匹配，重试 {max_chunk_retries} 次后仍失败",
                            ) from exc
                        raise
                    resp_path = resp.get("path")
                    if resp_path is not None and resp_path != entry.path:
                        raise UploadError(entry.path, f"服务端确认了错误的文件: {resp_path}")
                    if resp.get("done"):
                        # 正常情况下只有 file_done 才会 done=true；提前收到视为整文件已就绪
                        progress(entry.path, entry.size, entry.size)
                        return entry.size
                    try:
                        next_offset = int(resp.get("offset", offset + length))
                    except (TypeError, ValueError):
                        raise UploadError(entry.path, f"服务端 file_ok.offset 非法: {resp!r}")
                    if next_offset == offset + length:
                        offset = next_offset
                    elif 0 <= next_offset <= entry.size:
                        # 服务端要求从更早偏移继续（如分片丢失）
                        offset = next_offset
                    else:
                        raise UploadError(
                            entry.path, f"服务端返回的 offset 越界: {next_offset}"
                        )
                    progress(entry.path, offset, entry.size)
                    break

            # 所有分片已发送：请求整文件校验
            for done_attempt in range(max_file_retries + 1):
                conn.send_json({"kind": protocol.KIND_FILE_DONE, "path": entry.path})
                try:
                    resp = conn.wait_for([protocol.KIND_FILE_OK])
                except RemoteError as exc:
                    if exc.code == protocol.ERR_HASH_MISMATCH and done_attempt < max_file_retries:
                        retry_whole_file = True
                        break
                    raise UploadError(entry.path, f"整文件校验失败: {exc}") from exc
                if resp.get("path") not in (None, entry.path):
                    raise UploadError(entry.path, f"服务端确认了错误的文件: {resp.get('path')}")
                if not resp.get("done"):
                    raise UploadError(entry.path, f"服务端未确认文件完成: {resp!r}")
                progress(entry.path, entry.size, entry.size)
                return entry.size

        if retry_whole_file:
            continue
        raise UploadError(entry.path, "未知的上传状态")

    raise UploadError(entry.path, f"整文件重传 {max_file_retries} 次后仍失败")


def upload_files(
    conn: Connection,
    entries: Sequence[FileEntry],
    project_dir: str | os.PathLike,
    *,
    chunk_size: int = protocol.DEFAULT_CHUNK_SIZE,
    on_progress: Optional[ProgressCallback] = None,
    on_file_start: Optional[FileCallback] = None,
    on_file_done: Optional[FileCallback] = None,
    start_offsets: Optional[Dict[str, int]] = None,
) -> int:
    """批量上传；返回总字节数。``on_progress`` 参数为累计已传/总字节。"""
    entries = list(entries)
    total = sum(int(e.size) for e in entries)
    done_before = 0
    overall = on_progress or (lambda _done, _total: None)

    def _file_progress(_path: str, done: int, _file_total: int) -> None:
        overall(done_before + max(0, min(done, _file_total)), total)

    for entry in entries:
        if on_file_start is not None:
            on_file_start(entry)
        upload_file(
            conn,
            entry,
            project_dir,
            chunk_size=chunk_size,
            on_progress=_file_progress,
            start_offsets=start_offsets,
        )
        done_before += int(entry.size)
        overall(done_before, total)
        if on_file_done is not None:
            on_file_done(entry)
    return total


def verify_local_entries(
    entries: Sequence[FileEntry], project_dir: str | os.PathLike
) -> List[str]:
    """上传前快速核对本地文件是否仍与清单一致，返回问题列表（空表示一致）。"""
    root = Path(project_dir).expanduser()
    problems: List[str] = []
    for entry in entries:
        full = root / Path(*entry.path.split("/"))
        if not full.is_file():
            problems.append(f"{entry.path}: 文件不存在")
            continue
        size = os.path.getsize(full)
        if size != entry.size:
            problems.append(f"{entry.path}: 大小 {entry.size} → {size}")
            continue
        try:
            if sha256_file(full) != entry.sha256:
                problems.append(f"{entry.path}: sha256 已变化")
        except OSError as exc:
            problems.append(f"{entry.path}: 读取失败 {exc}")
    return problems
