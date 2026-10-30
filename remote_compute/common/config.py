"""默认配置、客户端配置文件读写、token 生成。

配置优先级（客户端）：命令行参数 > 配置文件 > 默认值。
客户端配置默认位置：
- Windows: ``%APPDATA%\\rgpu\\client.json``
- Linux/macOS: ``~/.config/rgpu/client.json``
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .protocol import DEFAULT_CHUNK_SIZE

DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"
DEFAULT_SERVER_HOST = "0.0.0.0"
CLIENT_CONFIG_NAME = "client.json"
SERVER_DATA_DIR_NAME = ".rgpu-server"


def generate_token(nbytes: int = 24) -> str:
    """生成 URL 安全 token（默认 32 字符）。"""
    return secrets.token_urlsafe(nbytes)


def _config_root() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "rgpu"
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "rgpu"


def default_client_config_path() -> Path:
    return _config_root() / CLIENT_CONFIG_NAME


def default_server_data_dir() -> Path:
    env = os.environ.get("RGPU_DATA_DIR")
    if env:
        return Path(env)
    return Path.home() / SERVER_DATA_DIR_NAME


def parse_host_port(value: str, default_port: int = DEFAULT_PORT) -> tuple[str, int]:
    """解析 ``host:port`` / ``host`` / ``:port``。"""
    text = (value or "").strip()
    if not text:
        return DEFAULT_HOST, default_port
    if text.startswith("["):  # IPv6 [::1]:8765
        host, _, rest = text[1:].partition("]")
        if rest.startswith(":"):
            return host, int(rest[1:])
        return host, default_port
    if text.count(":") == 1:
        host, port = text.rsplit(":", 1)
        return host or DEFAULT_HOST, int(port)
    return text, default_port


@dataclass
class ClientConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    token: str = ""
    chunk_size: int = DEFAULT_CHUNK_SIZE
    connect_timeout: float = 10.0
    io_timeout: float = 300.0
    download_dir: str = "outputs"
    name: str = ""

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "token": self.token,
            "chunk_size": self.chunk_size,
            "connect_timeout": self.connect_timeout,
            "io_timeout": self.io_timeout,
            "download_dir": self.download_dir,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClientConfig":
        base = cls()
        return cls(
            host=str(d.get("host", base.host)),
            port=int(d.get("port", base.port)),
            token=str(d.get("token", base.token)),
            chunk_size=int(d.get("chunk_size", base.chunk_size)),
            connect_timeout=float(d.get("connect_timeout", base.connect_timeout)),
            io_timeout=float(d.get("io_timeout", base.io_timeout)),
            download_dir=str(d.get("download_dir", base.download_dir)),
            name=str(d.get("name", base.name)),
        )


@dataclass
class ServerConfig:
    host: str = DEFAULT_SERVER_HOST
    port: int = DEFAULT_PORT
    token: str = ""
    data_dir: str = ""
    max_parallel_jobs: int = 1
    python_bin: str = sys.executable
    enable_docker: bool = False
    strict_gpu: bool = False
    strict_recovery: bool = False
    recover_interrupted: bool = True
    min_gpu_free_mb: int = 16384
    weight_cache_bytes: int = 0
    max_upload_bytes: int = 50 * 1024 ** 3   # 50 GiB
    idle_timeout_s: int = 300
    keep_finished_jobs: int = 50
    env_cache_size: int = 20
    extra: Dict[str, Any] = field(default_factory=dict)


def load_client_config(path: Optional[str] = None) -> ClientConfig:
    """读取客户端配置；文件不存在时返回默认配置（不抛错）。"""
    config_path = Path(path) if path else default_client_config_path()
    if not config_path.is_file():
        return ClientConfig()
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"客户端配置文件无法解析: {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"客户端配置文件格式错误（应为 JSON 对象）: {config_path}")
    return ClientConfig.from_dict(data)


def save_client_config(config: ClientConfig, path: Optional[str] = None) -> Path:
    """保存客户端配置（含 token），文件权限尽量收紧到 600。"""
    config_path = Path(path) if path else default_client_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, config_path)
    return config_path


def server_config_from_args(args: Any) -> ServerConfig:
    """从 argparse 命名空间构造 ServerConfig（server 包使用，避免重复字段映射）。"""
    data_dir = args.data_dir or str(default_server_data_dir())
    token = args.token or os.environ.get("RGPU_TOKEN", "")
    return ServerConfig(
        host=args.host,
        port=args.port,
        token=token,
        data_dir=data_dir,
        max_parallel_jobs=max(1, int(args.max_parallel)),
        python_bin=args.python_bin,
        enable_docker=bool(args.enable_docker),
        strict_gpu=bool(getattr(args, 'strict_gpu', False)),
        strict_recovery=bool(getattr(args, 'strict_recovery', False)),
        recover_interrupted=bool(getattr(args, 'recover_interrupted', True)),
        min_gpu_free_mb=int(getattr(args, 'min_gpu_free_mb', 16384)),
        weight_cache_bytes=max(0, int(getattr(args, 'weight_cache_bytes', 0) or 0)),
        idle_timeout_s=int(args.idle_timeout),
    )


def server_config_to_dict(cfg: ServerConfig) -> Dict[str, Any]:
    return {
        "host": cfg.host,
        "port": cfg.port,
        "token": cfg.token,
        "data_dir": cfg.data_dir,
        "max_parallel_jobs": cfg.max_parallel_jobs,
        "python_bin": cfg.python_bin,
        "enable_docker": cfg.enable_docker,
        "strict_gpu": cfg.strict_gpu,
        "strict_recovery": cfg.strict_recovery,
        "recover_interrupted": cfg.recover_interrupted,
        "min_gpu_free_mb": cfg.min_gpu_free_mb,
        "weight_cache_bytes": cfg.weight_cache_bytes,
        "max_upload_bytes": cfg.max_upload_bytes,
        "idle_timeout_s": cfg.idle_timeout_s,
    }
