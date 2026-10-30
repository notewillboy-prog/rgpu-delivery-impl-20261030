"""数据模型：任务描述、文件条目、环境描述、资源需求、GPU 信息。

这些 dataclass 是客户端与服务端之间唯一的任务描述格式；
所有 ``to_dict()`` 输出都必须能 JSON 序列化，``from_dict()`` 必须能反向还原。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .hashing import sha256_bytes

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_WIN_DRIVE = re.compile(r"^[A-Za-z]:")


class JobState(str, Enum):
    QUEUED = "queued"
    UPLOADING = "uploading"
    ENV_BUILDING = "env_building"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.TIMEOUT,
            JobState.CANCELLED,
        )


@dataclass
class FileEntry:
    """一个待上传/待回传文件。``path`` 一律用 POSIX 相对路径。"""

    path: str
    size: int
    sha256: str
    executable: bool = False

    def __post_init__(self) -> None:
        self.path = self.path.replace("\\", "/")
        if (
            not self.path
            or self.path.startswith("/")
            or _WIN_DRIVE.match(self.path)
            or ".." in self.path.split("/")
        ):
            raise ValueError(f"unsafe file path: {self.path!r}")
        if self.size < 0:
            raise ValueError(f"negative size for {self.path}")
        if not _HEX64.match(self.sha256):
            raise ValueError(f"bad sha256 for {self.path}: {self.sha256}")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"path": self.path, "size": self.size, "sha256": self.sha256}
        if self.executable:
            d["executable"] = True
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FileEntry":
        return cls(
            path=str(d["path"]),
            size=int(d["size"]),
            sha256=str(d["sha256"]),
            executable=bool(d.get("executable", False)),
        )


@dataclass
class EnvSpec:
    """远端环境描述。``fingerprint`` 可由客户端预填，服务端会重新计算并校验。"""

    mode: str = "venv"                 # venv | docker
    python_version: str = "3.11"
    requirements: List[str] = field(default_factory=list)
    fingerprint: str = ""
    env_vars: Dict[str, str] = field(default_factory=dict)
    docker_image: str = ""             # docker 模式下可选的自定义基础镜像
    reuse: bool = True                 # 允许服务端复用同指纹环境

    def compute_fingerprint(self) -> str:
        from .hashing import env_fingerprint

        return env_fingerprint(
            self.requirements,
            self.python_version,
            self.mode,
            extra=self.docker_image,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "python_version": self.python_version,
            "requirements": list(self.requirements),
            "fingerprint": self.fingerprint,
            "env_vars": dict(self.env_vars),
            "docker_image": self.docker_image,
            "reuse": self.reuse,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnvSpec":
        return cls(
            mode=str(d.get("mode", "venv")),
            python_version=str(d.get("python_version", "3.11")),
            requirements=[str(x) for x in d.get("requirements", [])],
            fingerprint=str(d.get("fingerprint", "")),
            env_vars={str(k): str(v) for k, v in d.get("env_vars", {}).items()},
            docker_image=str(d.get("docker_image", "")),
            reuse=bool(d.get("reuse", True)),
        )


@dataclass
class Resources:
    gpu: bool = False
    gpu_count: int = 1
    timeout_s: int = 3600
    cpus: int = 0                      # 0 = 不限制
    memory_gb: float = 0.0             # 0 = 不限制（仅记录，demo 不强制）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu": self.gpu,
            "gpu_count": self.gpu_count,
            "timeout_s": self.timeout_s,
            "cpus": self.cpus,
            "memory_gb": self.memory_gb,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Resources":
        return cls(
            gpu=bool(d.get("gpu", False)),
            gpu_count=int(d.get("gpu_count", 1)),
            timeout_s=int(d.get("timeout_s", 3600)),
            cpus=int(d.get("cpus", 0)),
            memory_gb=float(d.get("memory_gb", 0.0)),
        )


@dataclass
class JobSpec:
    """客户端提交的完整任务描述（环境 + 任务分离）。"""

    job_name: str
    entry_argv: List[str]
    inputs: List[FileEntry] = field(default_factory=list)
    env: EnvSpec = field(default_factory=EnvSpec)
    resources: Resources = field(default_factory=Resources)
    outputs: List[str] = field(default_factory=lambda: ["outputs/**"])
    workdir: str = "."
    meta: Dict[str, Any] = field(default_factory=dict)
    client_version: str = "0.1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_name": self.job_name,
            "entry_argv": list(self.entry_argv),
            "inputs": [f.to_dict() for f in self.inputs],
            "env": self.env.to_dict(),
            "resources": self.resources.to_dict(),
            "outputs": list(self.outputs),
            "workdir": self.workdir,
            "meta": dict(self.meta),
            "client_version": self.client_version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobSpec":
        return cls(
            job_name=str(d["job_name"]),
            entry_argv=[str(x) for x in d["entry_argv"]],
            inputs=[FileEntry.from_dict(x) for x in d.get("inputs", [])],
            env=EnvSpec.from_dict(d.get("env", {})),
            resources=Resources.from_dict(d.get("resources", {})),
            outputs=[str(x) for x in d.get("outputs", ["outputs/**"])],
            workdir=str(d.get("workdir", ".")),
            meta=dict(d.get("meta", {})),
            client_version=str(d.get("client_version", "0.1.0")),
        )

    def total_input_bytes(self) -> int:
        return sum(f.size for f in self.inputs)

    def input_index(self) -> Dict[str, FileEntry]:
        return {f.path: f for f in self.inputs}


@dataclass
class OutputEntry:
    path: str
    size: int
    sha256: str
    kind: str = "file"                 # file | dir_entry

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256, "kind": self.kind}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputEntry":
        return cls(
            path=str(d["path"]),
            size=int(d["size"]),
            sha256=str(d["sha256"]),
            kind=str(d.get("kind", "file")),
        )


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_total_mb: int = 0
    memory_free_mb: int = 0
    driver: str = ""
    cuda: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "memory_total_mb": self.memory_total_mb,
            "memory_free_mb": self.memory_free_mb,
            "driver": self.driver,
            "cuda": self.cuda,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GpuInfo":
        return cls(
            index=int(d["index"]),
            name=str(d["name"]),
            memory_total_mb=int(d.get("memory_total_mb", 0)),
            memory_free_mb=int(d.get("memory_free_mb", 0)),
            driver=str(d.get("driver", "")),
            cuda=str(d.get("cuda", "")),
        )


def validate_job_spec(spec: JobSpec) -> None:
    """提交前的本地校验；错误信息直接面向用户。"""
    if not spec.job_name.strip():
        raise ValueError("job_name 不能为空")
    if not spec.entry_argv:
        raise ValueError("entry_argv 不能为空，例如 ['python', 'train.py', '--epochs', '3']")
    if not spec.inputs:
        raise ValueError("inputs 为空：没有可上传的文件")
    seen = set()
    for entry in spec.inputs:
        if entry.path in seen:
            raise ValueError(f"inputs 中有重复路径: {entry.path}")
        seen.add(entry.path)
    if spec.env.mode not in ("venv", "docker"):
        raise ValueError(f"不支持的 env.mode: {spec.env.mode}")
    if not spec.env.fingerprint:
        spec.env.fingerprint = spec.env.compute_fingerprint()
    elif spec.env.fingerprint != spec.env.compute_fingerprint():
        raise ValueError("env.fingerprint 与本地 requirements 计算值不一致")
    if spec.resources.timeout_s <= 0:
        raise ValueError("resources.timeout_s 必须大于 0")
    for pattern in spec.outputs:
        if pattern.startswith("/") or ".." in pattern.replace("\\", "/").split("/"):
            raise ValueError(f"非法的产物匹配模式: {pattern}")


def dir_manifest_entry(path: str, size: int, sha256: str) -> Dict[str, Any]:
    """目录条目的 JSON 形式，仅用于产物清单。"""
    return {"path": path, "size": size, "sha256": sha256}


def fingerprint_of_manifest(entries: List[FileEntry]) -> str:
    """文件清单级指纹（用于判断两个任务的输入是否完全一致）。"""
    payload = "\n".join(f"{e.path}\t{e.size}\t{e.sha256}" for e in sorted(entries, key=lambda x: x.path))
    return sha256_bytes(payload.encode("utf-8"))


def normalize_relpath(path: str) -> str:
    """统一成 POSIX 相对路径，供跨平台使用。"""
    return os.path.normpath(path).replace("\\", "/")
