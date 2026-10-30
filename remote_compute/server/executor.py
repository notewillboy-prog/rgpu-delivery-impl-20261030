"""作业执行：``subprocess.Popen(argv)`` + 流式日志 + 超时 kill + 产物收集。

执行流程（与 ``docs/PROTOCOL.md`` 3.4 对齐）：

1. ``job_progress(state=env_building)`` → ``envbuilder.prepare()``（复用或构建）；
2. 环境就绪 → ``job_started(env={fingerprint, state})``；
3. ``Popen`` 作业进程，cwd = ``jobs/<id>/inputs``（再叠加 ``spec.workdir``），
   stdout/stderr 合并后逐行广播 ``job_event``，同时写 ``logs/job.log``；
4. 超时 → kill 进程树，``state=timeout, exit_code=-1``；
5. 按 ``spec.outputs`` glob 收集产物到 ``jobs/<id>/outputs/`` 并算 sha256；
6. ``job_finished(state, exit_code, duration_s, outputs, log_tail)`` —— 成功/失败都发。

注：协议错误码表里有 ``JOB_FAILED``/``TIMEOUT``，但 3.4 明确「结束（成功或失败都会发）
job_finished」；执行期不再额外发 ``error``，避免客户端「等 job_finished 却先收到 error」
的分支分歧。客户端按 ``state`` / ``exit_code`` 映射退出码。
"""

from __future__ import annotations

import glob as globmod
import os
import shutil
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from common.hashing import sha256_file
from common.models import JobState

from .envbuilder import EnvBuildError, EnvBuilder, EnvResult
from .jobstore import JobStore, safe_join
from .runtime import JobRuntime

DEFAULT_OUTPUT_PATTERNS = ["outputs/**"]
_PYTHON_NAMES = {"python", "python3", "python.exe", "py", "python3.exe"}


class Executor:
    """执行作业；由调度器工作线程调用，同一作业只会被执行一次。"""

    def __init__(
        self,
        store: JobStore,
        envbuilder: EnvBuilder,
        python_bin: str = "",
        enable_docker: bool = False,
        logger: Optional[Callable[[str], None]] = None,
        gpu_prelaunch_check: Optional[Callable[[List[int]], None]] = None,
    ) -> None:
        self.store = store
        self.envbuilder = envbuilder
        self.python_bin = python_bin
        self.enable_docker = bool(enable_docker)
        self.logger = logger
        self.gpu_prelaunch_check = gpu_prelaunch_check
        self._procs: Dict[str, subprocess.Popen] = {}
        self._procs_lock = threading.Lock()

    # -- 对外 --------------------------------------------------------------
    def run(self, runtime: JobRuntime, gpu_ids: Optional[List[int]] = None) -> None:
        """执行一个作业（内部吞掉所有异常，保证一定发 ``job_finished``）。"""
        gpu_ids = list(gpu_ids or [])
        try:
            self._run_inner(runtime, gpu_ids)
        except Exception as exc:  # 兜底：服务端内部错误也要让客户端拿到终态
            self._log(f"[exec] 作业 {runtime.job_id} 执行异常: {exc!r}")
            try:
                runtime.append_log_line(f"[rgpu] 服务端内部错误: {exc!r}\n", stream="stderr")
            except Exception:
                pass
            runtime.mark_finished(JobState.FAILED, -1)

    def terminate_all(self) -> None:
        """服务端退出时杀掉在跑的作业进程（避免孤儿进程）。"""
        with self._procs_lock:
            procs = list(self._procs.values())
        for proc in procs:
            kill_process_tree(proc)

    # -- 内部 --------------------------------------------------------------
    def _run_inner(self, runtime: JobRuntime, gpu_ids: List[int]) -> None:
        spec = runtime.spec
        job_id = runtime.job_id
        runtime.set_state(JobState.ENV_BUILDING)
        runtime.set_progress(0.1)
        runtime.emit_progress("env_building", 0.1, "检查/构建远端环境")

        def on_progress(message: str, value: float) -> None:
            runtime.emit_progress("env_building", 0.1 + 0.3 * float(value), message)

        def on_log(line: str) -> None:
            # envbuilder 自己的消息已带 "[env] " 前缀；pip/docker 原始输出补上前缀
            text = str(line)
            prefix = "" if text.startswith("[env]") else "[env] "
            runtime.append_log_line(f"{prefix}{text}\n", stream="stderr")

        try:
            env_result = self.envbuilder.prepare(
                spec.env, progress=on_progress, on_log=on_log
            )
        except EnvBuildError as exc:
            runtime.set_env(spec.env.fingerprint or "", "failed")
            runtime.append_log_line(f"[env] 环境构建失败: {exc.message}\n", stream="stderr")
            for line in exc.tail:
                runtime.append_raw_log(line + "\n")
            runtime.set_state(JobState.FAILED)
            runtime.mark_finished(JobState.FAILED, -1)
            return

        runtime.set_env(env_result.fingerprint, env_result.state)
        if env_result.note:
            runtime.note = env_result.note
        runtime.emit_progress(
            "env_building",
            0.4,
            f"环境就绪（{env_result.state}）",
        )

        # -- 工作目录 ------------------------------------------------------
        workdir = self._workdir(job_id, spec.workdir)
        if workdir is None or not workdir.is_dir():
            message = f"作业工作目录不存在: {spec.workdir!r}（inputs 内）"
            runtime.append_log_line(f"[rgpu] {message}\n", stream="stderr")
            runtime.set_state(JobState.FAILED)
            runtime.mark_finished(JobState.FAILED, -1)
            return

        # -- 组装命令 ------------------------------------------------------
        try:
            argv, proc_env, cwd = self._build_command(runtime, env_result, workdir, gpu_ids)
        except ValueError as exc:
            runtime.append_log_line(f"[rgpu] 入口命令非法: {exc}\n", stream="stderr")
            runtime.set_state(JobState.FAILED)
            runtime.mark_finished(JobState.FAILED, -1)
            return

        # Environment setup can be slow. Recheck immediately before execution;
        # this is still a snapshot, not an inter-process exclusive reservation.
        if spec.resources.gpu and self.gpu_prelaunch_check is not None:
            self.gpu_prelaunch_check(gpu_ids)
        runtime.set_state(JobState.RUNNING)
        runtime.set_progress(0.5)
        runtime.emit(runtime.job_started_message())

        # -- 启动进程 ------------------------------------------------------
        self._log(f"[exec] 启动作业 {job_id}: {argv[:4]}… cwd={cwd}")
        try:
            proc = subprocess.Popen(
                [str(x) for x in argv],
                cwd=str(cwd),
                env=proc_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=(os.name != "nt"),
            )
        except (OSError, ValueError) as exc:
            message = f"启动作业进程失败: {exc}"
            runtime.append_log_line(f"[rgpu] {message}\n", stream="stderr")
            runtime.set_state(JobState.FAILED)
            runtime.mark_finished(JobState.FAILED, -1)
            return

        with self._procs_lock:
            self._procs[job_id] = proc

        reader_done = threading.Event()

        def pump() -> None:
            stream = proc.stdout
            try:
                if stream is not None:
                    for line in stream:
                        runtime.append_log_line(line)
            finally:
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
                reader_done.set()

        reader = threading.Thread(target=pump, name=f"rgpu-log-{job_id}", daemon=True)
        reader.start()

        timeout_s = int(getattr(spec.resources, "timeout_s", 0) or 0)
        timed_out = False
        try:
            proc.wait(timeout=timeout_s if timeout_s > 0 else None)
        except subprocess.TimeoutExpired:
            timed_out = True
            runtime.append_log_line(
                f"[rgpu] 作业超时（{timeout_s}s），终止进程树\n", stream="stderr"
            )
            kill_process_tree(proc)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - 极端情况
                pass
        finally:
            reader_done.wait(timeout=15)
            try:
                reader.join(timeout=5)
            except RuntimeError:  # pragma: no cover
                pass
            with self._procs_lock:
                self._procs.pop(job_id, None)

        exit_code = proc.returncode if proc.returncode is not None else -1
        outputs = collect_outputs(
            workdir,
            spec.outputs or DEFAULT_OUTPUT_PATTERNS,
            self.store.outputs_dir(job_id),
            on_log=runtime.append_raw_log,
        )
        if timed_out:
            runtime.set_state(JobState.TIMEOUT)
            runtime.mark_finished(JobState.TIMEOUT, -1, outputs=outputs)
        elif exit_code == 0:
            runtime.set_state(JobState.SUCCEEDED)
            runtime.mark_finished(JobState.SUCCEEDED, 0, outputs=outputs)
        else:
            runtime.append_log_line(
                f"[rgpu] 作业退出码 {exit_code}\n", stream="stderr"
            )
            runtime.set_state(JobState.FAILED)
            runtime.mark_finished(JobState.FAILED, exit_code, outputs=outputs)

    # -- 辅助 --------------------------------------------------------------
    def _workdir(self, job_id: str, workdir_rel: str) -> Optional[Path]:
        inputs = self.store.inputs_dir(job_id)
        rel = str(workdir_rel or ".").replace("\\", "/").strip()
        if rel in ("", ".", "./"):
            return inputs
        try:
            return safe_join(inputs, rel)
        except ValueError:
            return None

    def _build_command(
        self,
        runtime: JobRuntime,
        env_result: EnvResult,
        workdir: Path,
        gpu_ids: List[int],
    ):
        spec = runtime.spec
        job_id = runtime.job_id
        entry = [str(x) for x in spec.entry_argv]
        if not entry:
            raise ValueError("entry_argv 为空")
        env_vars = {str(k): str(v) for k, v in (spec.env.env_vars or {}).items()}
        inputs_dir = self.store.inputs_dir(job_id)

        if env_result.mode == "docker":
            argv, proc_env = self._build_docker_command(
                runtime, env_result, workdir, inputs_dir, entry, env_vars, gpu_ids
            )
            return argv, proc_env, workdir

        argv = list(entry)
        if _basename(argv[0]).lower() in _PYTHON_NAMES and env_result.python_bin:
            argv[0] = str(env_result.python_bin)
        proc_env = dict(os.environ)
        proc_env.update(env_vars)
        proc_env["PYTHONUNBUFFERED"] = "1"
        proc_env.setdefault("PYTHONIOENCODING", "utf-8")
        proc_env.setdefault("PYTHONUTF8", "1")
        proc_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        proc_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids) if gpu_ids else ""
        venv_bin = Path(env_result.python_bin).parent if env_result.python_bin else None
        if venv_bin is not None and venv_bin.is_dir():
            proc_env["PATH"] = str(venv_bin) + os.pathsep + proc_env.get("PATH", "")
        return argv, proc_env, workdir

    def _build_docker_command(
        self,
        runtime: JobRuntime,
        env_result: EnvResult,
        workdir: Path,
        inputs_dir: Path,
        entry: List[str],
        env_vars: Dict[str, str],
        gpu_ids: List[int],
    ) -> tuple[list, dict]:
        """docker 环境模式命令（本机无 Linux，未实测，见 deploy/README.md）。"""
        rel = "."
        try:
            rel = workdir.resolve().relative_to(inputs_dir.resolve()).as_posix()
        except ValueError:  # pragma: no cover
            rel = "."
        argv: list = [
            "docker", "run", "--rm",
            "--name", f"rgpu-{runtime.job_id}",
            "-v", f"{inputs_dir}:/work",
            "-w", f"/work/{rel}",
            "--ipc=host",
        ]
        if gpu_ids:
            argv += ["--gpus", f"device={','.join(str(x) for x in gpu_ids)}"]
        for key, value in env_vars.items():
            argv += ["-e", f"{key}={value}"]
        argv += ["-e", "PYTHONUNBUFFERED=1"]
        argv.append(env_result.docker_image)
        argv += [str(x) for x in entry]
        proc_env = dict(os.environ)
        proc_env["PYTHONUNBUFFERED"] = "1"
        return argv, proc_env

    def _log(self, message: str) -> None:
        if self.logger:
            try:
                self.logger(message)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _basename(value: str) -> str:
    text = str(value).replace("\\", "/")
    return text.rsplit("/", 1)[-1]


def kill_process_tree(proc: subprocess.Popen) -> None:
    """杀掉进程树：Windows 用 ``taskkill /T``，POSIX 用进程组 SIGKILL。"""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
            pass
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover
        try:
            proc.kill()
        except OSError:
            pass


def collect_outputs(
    workdir: Path,
    patterns: List[str],
    outputs_dir: Path,
    on_log: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """按 glob 收集产物：复制到 ``outputs_dir`` 并返回协议清单。

    注意 ``pathlib.Path.glob('outputs/**')`` 只匹配目录、不匹配目录里的文件，
    所以这里用 :func:`glob.glob(..., recursive=True)`，并对以 ``/**`` 结尾的
    模式特殊处理（= 目录下所有文件，递归），行为与用户直觉一致。
    """
    workdir = Path(workdir)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    matched: Dict[str, Path] = {}
    for pattern in patterns or DEFAULT_OUTPUT_PATTERNS:
        for candidate in _iter_output_matches(workdir, str(pattern)):
            try:
                if not candidate.is_file():
                    continue
                rel = candidate.resolve().relative_to(workdir.resolve()).as_posix()
            except (ValueError, OSError):
                continue
            matched.setdefault(rel, candidate)

    manifest: List[Dict[str, Any]] = []
    for rel in sorted(matched):
        source = matched[rel]
        dest = outputs_dir.joinpath(*rel.split("/"))
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            size = dest.stat().st_size
            digest = sha256_file(dest)
        except OSError as exc:
            if on_log:
                on_log(f"[rgpu] 收集产物失败: {rel}: {exc}\n")
            continue
        manifest.append({"path": rel, "size": size, "sha256": digest, "kind": "file"})
    if not manifest and on_log:
        on_log(
            "[rgpu] 警告：未匹配到任何产物（检查 outputs 模式 "
            f"{', '.join(map(str, patterns))} 与脚本输出目录）\n"
        )
    return manifest


def _iter_output_matches(workdir: Path, pattern: str) -> List[Path]:
    """把一个 outputs 模式展开成候选路径（文件与目录都可能，调用方筛文件）。"""
    text = pattern.replace("\\", "/").strip()
    if not text or text.startswith("/") or ".." in text.split("/"):
        return []
    if text in ("**", "**/*"):
        return sorted(workdir.rglob("*"))
    if text.endswith("/**"):
        base = text[:-3].strip("/")
        root = workdir
        if base:
            root = workdir.joinpath(*base.split("/"))
        if not root.is_dir():
            return []
        return sorted(root.rglob("*"))
    try:
        names = globmod.glob(text, root_dir=str(workdir), recursive=True)
    except (ValueError, OSError):
        return []
    results: List[Path] = []
    for name in names:
        results.append(workdir.joinpath(*str(name).replace("\\", "/").split("/")))
    return results


__all__ = [
    "Executor",
    "collect_outputs",
    "kill_process_tree",
    "DEFAULT_OUTPUT_PATTERNS",
]
