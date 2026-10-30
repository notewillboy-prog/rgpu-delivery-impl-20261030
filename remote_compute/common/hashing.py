"""哈希与环境指纹。

环境指纹 = sha256(mode + python 版本 + 归一化后的 requirements)。
要求：同一份依赖清单在不同机器/不同顺序/不同注释下必须得到同一指纹，
这样服务端才能「先检查环境是否存在，存在就复用」。
"""

from __future__ import annotations

import hashlib
import io
import os
import re
from typing import Iterable, List, Union

_CHUNK = 1024 * 1024
_INLINE_COMMENT = re.compile(r"\s+#.*$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_stream(stream: io.BufferedIOBase, chunk_size: int = _CHUNK) -> str:
    digest = hashlib.sha256()
    while True:
        block = stream.read(chunk_size)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Union[str, os.PathLike], chunk_size: int = _CHUNK) -> str:
    with open(path, "rb") as fh:
        return sha256_stream(fh, chunk_size)


def normalize_requirements(text_or_lines: Union[str, Iterable[str], None]) -> List[str]:
    """去注释、去空行、去首尾空白、去重、排序。

    保留 pip 的 ``-r`` / ``--index-url`` 等选项行（原样保留、去重排序），
    但忽略以 ``#`` 开头的整行注释。
    """
    if text_or_lines is None:
        return []
    if isinstance(text_or_lines, str):
        raw_lines = text_or_lines.splitlines()
    else:
        raw_lines = []
        for item in text_or_lines:
            raw_lines.extend(str(item).splitlines())
    cleaned = set()
    for raw in raw_lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # 行尾注释：pip 把「空白 + #」之后视为注释（"pkg==1.0 # note" / "pkg==1.0  # note"）
        line = _INLINE_COMMENT.sub("", line).strip()
        if not line:
            continue
        cleaned.add(line)
    return sorted(cleaned)


def env_fingerprint(
    requirements: Union[str, Iterable[str], None],
    python_version: str,
    mode: str = "venv",
    extra: str = "",
) -> str:
    """环境指纹（十六进制 sha256）。

    ``mode``：``venv`` 或 ``docker``；``extra`` 可放镜像基础层、GPU 架构等附加信息。
    """
    mode = (mode or "venv").strip().lower()
    if mode not in ("venv", "docker"):
        raise ValueError(f"unknown env mode: {mode}")
    reqs = normalize_requirements(requirements)
    payload = "\n".join(
        [
            "rgpu-env-v1",
            f"mode={mode}",
            f"python={python_version.strip()}",
            f"extra={extra.strip()}",
            "requirements:",
            *reqs,
        ]
    )
    return sha256_bytes(payload.encode("utf-8"))


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"
