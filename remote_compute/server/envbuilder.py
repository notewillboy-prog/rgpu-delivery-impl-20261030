"""环境构建：`EnvSpec` → 环境指纹 → 目录复用 / venv 重建 / docker 镜像。

设计（``docs/ARCHITECTURE.md`` 3.2）：

- 目录 ``<data_dir>/envs/<fingerprint>/``；构建成功后写 ``.ready`` 标记。
- ``.ready`` 存在且解释器还在 → ``state="cached"``，不重复构建。
- venv 模式：
  - ``requirements`` 为空 → ``python -m venv --without-pip``（零依赖也能隔离环境，
    不需要联网；本机 Windows 自测走这条路径）；
  - 非空 → ``python -m venv`` + ``<venv>/bin/python -m pip install -r requirements.txt``。
- docker 模式：按指纹生成 ``Dockerfile`` + ``requirements.txt``，构建
  ``rgpu-env:<fingerprint 前 16 位>``；服务端未启用 docker 时抛 ``ENV_BUILD_FAILED``。
- 构建失败抛 :class:`EnvBuildError`，由执行器转成
  ``job_finished(state=failed, exit_code=-1, log_tail=[...])``。
- 构建输出通过 ``on_log`` 回调进作业日志，进度通过 ``progress`` 回调转成
  ``job_progress``。

本文件中的 docker 分支在 Windows 开发机上**未实测**（只能在 Linux + nvidia-container
环境验证），见 ``deploy/README.md`` 的「未验证」清单。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from common.models import EnvSpec

DEFAULT_DOCKER_BASE = "nvidia/cuda:12.4.1-runtime-ubuntu22.04"
BUILD_TIMEOUT_S = 1800
_TAIL_LINES = 20

ProgressCB = Callable[[str, float], None]
LogCB = Callable[[str], None]


class EnvBuildError(Exception):
    """环境构建失败（协议错误码 ``ENV_BUILD_FAILED``）。"""

    def __init__(self, message: str, log_lines: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.log_lines = list(log_lines or [])

    @property
    def tail(self) -> List[str]:
        return self.log_lines[-_TAIL_LINES:]


class EnvResult:
    """环境准备好的结果。``state`` 为 ``cached`` 或 ``built``。"""

    def __init__(
        self,
        mode: str,
        fingerprint: str,
        env_dir: Path,
        state: str,
        python_bin: str = "",
        docker_image: str = "",
        note: str = "",
    ) -> None:
        self.mode = mode
        self.fingerprint = fingerprint
        self.env_dir = env_dir
        self.state = state
        self.python_bin = python_bin
        self.docker_image = docker_image
        self.note = note

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"EnvResult(mode={self.mode!r}, fingerprint={self.fingerprint[:8]}…, "
            f"state={self.state!r}, python_bin={self.python_bin!r})"
        )


def venv_python(venv_dir: Path) -> Path:
    """venv 内解释器路径（Windows ``Scripts/python.exe``，POSIX ``bin/python``）。"""
    venv_dir = Path(venv_dir)
    candidates = (
        venv_dir / "Scripts" / "python.exe",
        venv_dir / "bin" / "python",
        venv_dir / "bin" / "python3",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # 尚未创建时给出「预期路径」，便于调用方报错
    return candidates[1] if os.name != "nt" else candidates[0]


def _run_capture(
    argv: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: float = BUILD_TIMEOUT_S,
    on_log: Optional[LogCB] = None,
) -> Tuple[int, List[str]]:
    """跑一个构建命令，按行回调日志，返回 ``(returncode, lines)``。"""
    lines: List[str] = []
    try:
        proc = subprocess.run(
            [str(x) for x in argv],
            cwd=str(cwd) if cwd else None,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        message = f"命令不存在: {argv[0]} ({exc})"
        if on_log:
            on_log(message)
        return 127, [message]
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or "") + (exc.stderr or "")
        message = f"构建命令超时（{timeout}s）: {' '.join(map(str, argv))}"
        if on_log:
            on_log(message)
            for line in str(text).splitlines():
                on_log(line)
        return 124, [message]
    raw = (proc.stdout or "") + (proc.stderr or "")
    for line in raw.splitlines():
        lines.append(line)
        if on_log:
            on_log(line)
    return proc.returncode, lines


class EnvBuilder:
    """按环境指纹构建/复用远端执行环境。"""

    def __init__(
        self,
        envs_dir: str | os.PathLike,
        python_bin: Optional[str] = None,
        enable_docker: bool = False,
        logger: Optional[Any] = None,
    ) -> None:
        self.envs_dir = Path(envs_dir)
        self.envs_dir.mkdir(parents=True, exist_ok=True)
        self.python_bin = python_bin or sys.executable
        self.enable_docker = bool(enable_docker)
        self.logger = logger
        # 同一环境指纹的构建必须串行：两个作业同时首次提交同一指纹时，
        # 不能让两个线程同时往 envs/<fingerprint>/ 里写（会互相 rmtree）。
        self._fp_locks_guard = threading.Lock()
        self._fp_locks: Dict[str, threading.Lock] = {}

    def _fingerprint_lock(self, fingerprint: str) -> threading.Lock:
        with self._fp_locks_guard:
            lock = self._fp_locks.get(fingerprint)
            if lock is None:
                lock = threading.Lock()
                self._fp_locks[fingerprint] = lock
            return lock

    # -- 对外入口 ----------------------------------------------------------
    def prepare(
        self,
        env: EnvSpec,
        progress: Optional[ProgressCB] = None,
        on_log: Optional[LogCB] = None,
    ) -> EnvResult:
        fingerprint = env.fingerprint or env.compute_fingerprint()
        mode = (env.mode or "venv").strip().lower()
        requested_py = (env.python_version or "").strip()
        actual_py = f"{sys.version_info.major}.{sys.version_info.minor}"
        note = ""
        if requested_py and requested_py != actual_py:
            note = (
                f"客户端 Python {requested_py} / 服务端 {actual_py}；"
                f"按服务端解释器（{self.python_bin}）构建环境"
            )
            self._log(on_log, f"[env] 注意：{note}")

        if mode == "docker":
            with self._fingerprint_lock(fingerprint):
                return self._prepare_docker(env, fingerprint, progress, on_log, note)
        with self._fingerprint_lock(fingerprint):
            return self._prepare_venv(env, fingerprint, progress, on_log, note)

    # -- venv --------------------------------------------------------------
    def _prepare_venv(
        self,
        env: EnvSpec,
        fingerprint: str,
        progress: Optional[ProgressCB],
        on_log: Optional[LogCB],
        note: str,
    ) -> EnvResult:
        env_dir = self.envs_dir / fingerprint
        venv_dir = env_dir / "venv"
        ready = env_dir / ".ready"
        python = venv_python(venv_dir)

        if env.reuse and ready.is_file() and python.is_file():
            self._log(on_log, f"[env] 复用已有环境: {env_dir}")
            return EnvResult("venv", fingerprint, env_dir, "cached", str(python), note=note)

        # 半成品目录 / 明确要求重建：先清掉
        if env_dir.exists():
            self._log(on_log, f"[env] 清理旧的半成品环境目录: {env_dir}")
            shutil.rmtree(env_dir, ignore_errors=True)
        env_dir.mkdir(parents=True, exist_ok=True)

        requirements = [str(item).strip() for item in env.requirements if str(item).strip()]
        self._progress(progress, "创建 venv（python -m venv）", 0.1)
        argv = [self.python_bin, "-m", "venv"]
        if not requirements:
            argv.append("--without-pip")
        argv.append(str(venv_dir))
        self._log(on_log, f"[env] 执行: {' '.join(argv)}")
        rc, lines = _run_capture(argv, on_log=on_log)
        if rc != 0:
            shutil.rmtree(env_dir, ignore_errors=True)
            raise EnvBuildError(
                f"创建 venv 失败（exit={rc}），命令: {' '.join(argv)}", lines
            )
        python = venv_python(venv_dir)
        if not python.is_file():
            shutil.rmtree(env_dir, ignore_errors=True)
            raise EnvBuildError(f"venv 创建后找不到解释器: {python}", lines)

        if requirements:
            req_path = env_dir / "requirements.txt"
            req_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")
            self._progress(progress, "pip install -r requirements.txt", 0.3)
            pip_argv = [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "-r",
                str(req_path),
            ]
            self._log(on_log, f"[env] 执行: {' '.join(pip_argv)}")
            rc, lines = _run_capture(pip_argv, on_log=on_log, env=self._child_env())
            if rc != 0:
                shutil.rmtree(env_dir, ignore_errors=True)
                raise EnvBuildError(
                    f"pip install 失败（exit={rc}）: {' '.join(requirements)}", lines
                )

        self._write_ready(env_dir, {
            "mode": "venv",
            "fingerprint": fingerprint,
            "python_version": env.python_version,
            "requirements": requirements,
            "python_bin": str(python),
            "built_at": time.time(),
            "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", ""),
        })
        self._progress(progress, "环境构建完成（venv）", 0.95)
        return EnvResult("venv", fingerprint, env_dir, "built", str(python), note=note)

    # -- docker ------------------------------------------------------------
    def _prepare_docker(
        self,
        env: EnvSpec,
        fingerprint: str,
        progress: Optional[ProgressCB],
        on_log: Optional[LogCB],
        note: str,
    ) -> EnvResult:
        if not self.enable_docker:
            raise EnvBuildError(
                "服务端未启用 docker 环境模式：请用 --enable-docker 启动，"
                "或客户端改用 --env-mode venv"
            )
        if shutil.which("docker") is None:
            raise EnvBuildError("服务端找不到 docker CLI，无法使用 docker 环境模式")
        tag = f"rgpu-env:{fingerprint[:16]}"
        env_dir = self.envs_dir / fingerprint
        ready = env_dir / ".ready"
        if env.reuse and ready.is_file() and self._docker_image_exists(tag, on_log):
            self._log(on_log, f"[env] 复用 docker 镜像: {tag}")
            return EnvResult("docker", fingerprint, env_dir, "cached", "", tag, note=note)

        if env_dir.exists():
            shutil.rmtree(env_dir, ignore_errors=True)
        env_dir.mkdir(parents=True, exist_ok=True)

        requirements = [str(item).strip() for item in env.requirements if str(item).strip()]
        base_image = (env.docker_image or DEFAULT_DOCKER_BASE).strip()
        (env_dir / "requirements.txt").write_text(
            "\n".join(requirements) + ("\n" if requirements else ""), encoding="utf-8"
        )
        dockerfile = [
            f"FROM {base_image}",
            "ENV DEBIAN_FRONTEND=noninteractive",
            "RUN apt-get update \\",
            " && apt-get install -y --no-install-recommends python3 python3-pip \\",
            " && rm -rf /var/lib/apt/lists/*",
            "WORKDIR /work",
        ]
        if requirements:
            dockerfile += [
                "COPY requirements.txt /tmp/rgpu-requirements.txt",
                "RUN python3 -m pip install --no-cache-dir --disable-pip-version-check "
                "-r /tmp/rgpu-requirements.txt",
            ]
        (env_dir / "Dockerfile").write_text("\n".join(dockerfile) + "\n", encoding="utf-8")

        self._progress(progress, f"docker build -t {tag}", 0.2)
        argv = ["docker", "build", "-t", tag, "-f", str(env_dir / "Dockerfile"), str(env_dir)]
        self._log(on_log, f"[env] 执行: {' '.join(argv)}")
        rc, lines = _run_capture(argv, on_log=on_log)
        if rc != 0:
            raise EnvBuildError(f"docker build 失败（exit={rc}），镜像标签 {tag}", lines)

        self._write_ready(env_dir, {
            "mode": "docker",
            "fingerprint": fingerprint,
            "requirements": requirements,
            "base_image": base_image,
            "docker_image": tag,
            "built_at": time.time(),
        })
        self._progress(progress, "环境构建完成（docker）", 0.95)
        return EnvResult("docker", fingerprint, env_dir, "built", "", tag, note=note)

    def _docker_image_exists(self, tag: str, on_log: Optional[LogCB]) -> bool:
        try:
            proc = subprocess.run(
                ["docker", "image", "inspect", tag],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        if proc.returncode != 0:
            self._log(on_log, f"[env] 镜像不存在: {tag}")
        return proc.returncode == 0

    # -- 工具 --------------------------------------------------------------
    @staticmethod
    def _write_ready(env_dir: Path, payload: Dict[str, Any]) -> None:
        ready = env_dir / ".ready"
        ready.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def read_ready(env_dir: Path) -> Dict[str, Any]:
        ready = Path(env_dir) / ".ready"
        try:
            data = json.loads(ready.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _progress(progress: Optional[ProgressCB], message: str, value: float) -> None:
        if progress:
            try:
                progress(message, value)
            except Exception:
                pass

    @staticmethod
    def _log(on_log: Optional[LogCB], line: str) -> None:
        if on_log:
            try:
                on_log(str(line))
            except Exception:
                pass

    @staticmethod
    def _child_env() -> Dict[str, str]:
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        return env


__all__ = ["EnvBuilder", "EnvBuildError", "EnvResult", "venv_python", "DEFAULT_DOCKER_BASE"]
