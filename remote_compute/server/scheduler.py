"""GPU 探测与作业调度（FIFO 队列 + max_parallel + CUDA_VISIBLE_DEVICES 分配）。

- :func:`detect_gpus` 调 ``nvidia-smi`` 解析 GPU 列表；**任何失败都降级为空列表**
  （Windows 无驱动、无 ``nvidia-smi``、输出格式变化都不能让服务端起不来）。
- 解析函数拆成纯函数 :func:`parse_gpus_csv`，便于离线单测。
- :class:`Scheduler` 用 ``max_parallel_jobs`` 个工作线程消费 FIFO 队列；
  每个作业执行前按 ``resources.gpu_count`` 从「空闲 GPU 集合」里分配索引，
  通过 ``CUDA_VISIBLE_DEVICES=0,1`` 注入给作业进程，结束后释放。
- 没有可用 GPU 但作业要求 GPU 时**不阻塞**，以 CPU 继续执行并在
  ``job_started.note`` 里说明（demo 取舍，见 ARCHITECTURE 3.4）。
"""

from __future__ import annotations

import queue
import re
import shutil
import subprocess
import threading
from typing import Any, Callable, Dict, List, Optional

from common.models import GpuInfo

NVIDIA_SMI = "nvidia-smi"
QUERY_FIELDS = "index,name,memory.total,memory.free,driver_version"
QUERY_CMD = [
    NVIDIA_SMI,
    f"--query-gpu={QUERY_FIELDS}",
    "--format=csv,noheader,nounits",
]
_CUDA_VERSION_RE = re.compile(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)")


def strict_gpu_snapshot(timeout: float = 10.0) -> Dict[int, int]:
    """Return idle physical GPU index -> free MiB; fail closed on probe errors.

    This is a point-in-time check, not an exclusive lease against other users.
    """
    def query(fields: str) -> str:
        result = subprocess.run([NVIDIA_SMI, f'--query-{fields}', '--format=csv,noheader,nounits'],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=timeout)
        if result.returncode:
            raise RuntimeError('GPU 资源检查失败，拒绝执行')
        return result.stdout or ''
    try:
        rows = query('gpu=index,uuid,memory.free')
        processes = query('compute-apps=gpu_uuid,pid')
        occupied = set()
        for line in processes.splitlines():
            fields = [part.strip() for part in line.split(',')]
            if len(fields) != 2 or not fields[0].startswith('GPU-') or not fields[1].isdigit():
                raise ValueError('GPU 计算进程列表格式无效')
            occupied.add(fields[0])
        result = {}
        seen, seen_indices = set(), set()
        for line in rows.splitlines():
            fields = [part.strip() for part in line.split(',')]
            if (len(fields) != 3 or not fields[0].isdigit() or not fields[1].startswith('GPU-')
                    or not fields[2].isdigit() or fields[1] in seen or int(fields[0]) in seen_indices):
                raise ValueError('GPU 状态格式无效')
            seen.add(fields[1])
            seen_indices.add(int(fields[0]))
            if fields[1] not in occupied:
                result[int(fields[0])] = int(fields[2])
        if not rows.strip() or occupied - seen:
            raise ValueError('GPU 状态不完整')
        return result
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise RuntimeError('GPU 资源检查失败，拒绝执行') from exc


def _to_int(text: str) -> int:
    try:
        return int(float(str(text).strip()))
    except (TypeError, ValueError):
        return 0


def parse_gpus_csv(text: str, cuda: str = "") -> List[GpuInfo]:
    """解析 ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` 输出。

    每行形如 ``0, NVIDIA GeForce RTX 4050 Laptop GPU, 6141, 4639, 581.08``。
    GPU 名字里即使有逗号也能正确处理（前 1 段 index，后 3 段是数字字段）。
    任何解析不了的行直接跳过。
    """
    gpus: List[GpuInfo] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("Field ") or "not a valid field" in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        index_text, driver = parts[0], parts[-1]
        memory_free = parts[-2]
        memory_total = parts[-3]
        name = ", ".join(parts[1:-3]).strip()
        if not name:
            continue
        try:
            index = int(index_text)
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                index=index,
                name=name,
                memory_total_mb=_to_int(memory_total),
                memory_free_mb=_to_int(memory_free),
                driver=driver if driver.upper() != "N/A" else "",
                cuda=cuda,
            )
        )
    gpus.sort(key=lambda item: item.index)
    return gpus


def parse_cuda_version(text: str) -> str:
    """从 ``nvidia-smi`` 默认输出里抠 ``CUDA Version: 13.0``。"""
    match = _CUDA_VERSION_RE.search(str(text or ""))
    return match.group(1) if match else ""


def detect_cuda_version(timeout: float = 10.0) -> str:
    try:
        proc = subprocess.run(
            [NVIDIA_SMI],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return parse_cuda_version((proc.stdout or "") + (proc.stderr or ""))


def detect_gpus(
    query_timeout: float = 10.0,
    cuda_version: Optional[str] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> List[GpuInfo]:
    """探测本机 GPU；失败/无 GPU 时返回 ``[]``（绝不抛异常）。"""
    if shutil.which(NVIDIA_SMI) is None:
        _log(logger, "[gpu] 未找到 nvidia-smi，以 CPU 模式运行")
        return []
    try:
        proc = subprocess.run(
            QUERY_CMD,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=query_timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _log(logger, f"[gpu] nvidia-smi 调用失败（{exc}），以 CPU 模式运行")
        return []
    if proc.returncode != 0:
        _log(logger, f"[gpu] nvidia-smi 退出码 {proc.returncode}，以 CPU 模式运行")
        return []
    cuda = detect_cuda_version() if cuda_version is None else cuda_version
    gpus = parse_gpus_csv(proc.stdout or "", cuda=cuda)
    if gpus:
        _log(
            logger,
            "[gpu] 探测到 {} 张 GPU: {}".format(
                len(gpus),
                "; ".join(f"#{g.index} {g.name} {g.memory_total_mb}MiB" for g in gpus),
            ),
        )
    else:
        _log(logger, "[gpu] nvidia-smi 无可用 GPU 输出，以 CPU 模式运行")
    return gpus


def _log(logger: Optional[Callable[[str], None]], message: str) -> None:
    if logger:
        try:
            logger(message)
        except Exception:
            pass


class Scheduler:
    """FIFO 作业队列 + GPU 分配。

    ``runner`` 是执行回调，签名 ``runner(runtime, gpu_indices)``；由 ``executor``
    提供。调度器自己不关心执行细节。
    """

    def __init__(
        self,
        max_parallel: int = 1,
        gpus: Optional[List[GpuInfo]] = None,
        runner: Optional[Callable[[Any, List[int]], None]] = None,
        logger: Optional[Callable[[str], None]] = None,
        max_queue: Optional[int] = None,
        strict_gpu: bool = False,
        min_gpu_free_mb: int = 16384,
        strict_probe: Optional[Callable[[], Dict[int, int]]] = None,
    ) -> None:
        self.max_parallel = max(1, int(max_parallel))
        self.gpus = list(gpus or [])
        self.runner = runner
        self.logger = logger
        self.strict_gpu = bool(strict_gpu)
        self.min_gpu_free_mb = int(min_gpu_free_mb)
        if self.strict_gpu and self.min_gpu_free_mb <= 0:
            raise ValueError('严格 GPU 模式必须指定正数显存下限')
        self.strict_probe = strict_probe or strict_gpu_snapshot
        self.max_queue = int(max_queue) if max_queue else max(8, self.max_parallel * 4)
        self._queue: "queue.Queue[Optional[Any]]" = queue.Queue(maxsize=self.max_queue)
        self._workers: List[threading.Thread] = []
        self._stop = threading.Event()
        self._started = False
        self._gpu_busy: set = set()
        self._gpu_cv = threading.Condition()
        self._running: Dict[str, Any] = {}
        self._running_guard = threading.Lock()

    # -- 生命周期 ----------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for index in range(self.max_parallel):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"rgpu-worker-{index}",
                daemon=True,
            )
            thread.start()
            self._workers.append(thread)
        _log(self.logger, f"[sched] 启动 {self.max_parallel} 个工作线程，GPU 数 {len(self.gpus)}")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._gpu_cv:
            self._gpu_cv.notify_all()
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        for thread in self._workers:
            thread.join(timeout=timeout)
        self._workers.clear()
        self._started = False

    # -- 入队 --------------------------------------------------------------
    def submit(self, runtime: Any) -> bool:
        """入队；队列满返回 ``False``（协议错误码 ``BUSY``）。"""
        try:
            self._queue.put_nowait(runtime)
        except queue.Full:
            return False
        runtime.enqueued = True
        _log(self.logger, f"[sched] 入队 {runtime.job_id}（队列长度 {self._queue.qsize()}）")
        return True

    def queue_size(self) -> int:
        return self._queue.qsize()

    def gpu_state(self) -> Dict[str, Any]:
        with self._gpu_cv:
            return {
                "total": len(self.gpus),
                "busy": sorted(self._gpu_busy),
                "idle": [g.index for g in self.gpus if g.index not in self._gpu_busy],
            }

    # -- 内部 --------------------------------------------------------------
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                self._queue.task_done()
                continue
            runtime = item
            gpu_ids: List[int] = []
            try:
                gpu_ids = self.acquire_gpus(runtime)
                with self._running_guard:
                    self._running[runtime.job_id] = runtime
                if self.runner is not None:
                    self.runner(runtime, gpu_ids)
                else:  # pragma: no cover - 配置错误
                    raise RuntimeError("Scheduler 未配置 runner")
            except Exception as exc:  # 执行器崩了也不能让作业永远挂 running
                _log(self.logger, f"[sched] 作业 {runtime.job_id} 执行异常: {exc!r}")
                try:
                    try:
                        from common.models import JobState
                    except Exception:  # pragma: no cover
                        JobState = None  # type: ignore
                    if JobState is not None and not runtime.state.terminal:
                        runtime.append_log_line(f"[rgpu] 服务端执行异常: {exc!r}\n", stream="stderr")
                        runtime.mark_finished(JobState.FAILED, -1)
                except Exception:
                    pass
            finally:
                with self._running_guard:
                    self._running.pop(runtime.job_id, None)
                self.release_gpus(gpu_ids)
                self._queue.task_done()

    def acquire_gpus(self, runtime: Any) -> List[int]:
        """按作业需求分配 GPU 索引；无 GPU 时返回 ``[]``（CPU 模式）。"""
        resources = getattr(runtime.spec, "resources", None)
        want = 0
        needs_gpu = False
        if resources is not None:
            needs_gpu = bool(resources.gpu)
            want = max(0, int(resources.gpu_count or 0)) if needs_gpu else 0
        if self.strict_gpu and needs_gpu:
            want = max(1, want)
            with self._gpu_cv:
                if self._stop.is_set():
                    raise RuntimeError('调度器正在停止，拒绝 GPU 作业')
                live = self.strict_probe()
                expected = {g.index for g in self.gpus}
                eligible = [index for index, free in sorted(live.items())
                            if index in expected and index not in self._gpu_busy
                            and free >= self.min_gpu_free_mb]
                if len(eligible) < want:
                    raise RuntimeError(f'GPU 不足、显存不足或被其他计算进程占用：需要 {want} 张且每张至少 {self.min_gpu_free_mb} MiB；未执行')
                chosen = eligible[:want]
                self._gpu_busy.update(chosen)
                runtime.set_gpus(chosen)
                return chosen
        if not needs_gpu or not self.gpus:
            if needs_gpu and not self.gpus:
                runtime.note = "服务端无可用 GPU，以 CPU 模式执行"
                _log(self.logger, f"[sched] {runtime.job_id} 要求 GPU 但无可用 GPU，CPU 执行")
            return []
        want = max(1, min(want, len(self.gpus)))
        with self._gpu_cv:
            while not self._stop.is_set():
                idle = [g.index for g in self.gpus if g.index not in self._gpu_busy]
                if len(idle) >= want:
                    chosen = idle[:want]
                    self._gpu_busy.update(chosen)
                    runtime.set_gpus(chosen)
                    _log(
                        self.logger,
                        f"[sched] {runtime.job_id} 分配 GPU {chosen}",
                    )
                    return chosen
                self._gpu_cv.wait(timeout=0.5)
        return []

    def release_gpus(self, gpu_ids: List[int]) -> None:
        if not gpu_ids:
            return
        with self._gpu_cv:
            for index in gpu_ids:
                self._gpu_busy.discard(index)
            self._gpu_cv.notify_all()

    def validate_prelaunch(self, gpu_ids: List[int]) -> None:
        if not self.strict_gpu:
            return
        if not gpu_ids or self._stop.is_set():
            raise RuntimeError('GPU 分配无效或服务正在停止；未执行')
        live = self.strict_probe()
        if any(live.get(index, 0) < self.min_gpu_free_mb for index in gpu_ids):
            raise RuntimeError('GPU 状态在准备环境时改变：被占用或显存不足；未执行')


__all__ = [
    "GpuInfo",
    "Scheduler",
    "detect_gpus",
    "detect_cuda_version",
    "parse_gpus_csv",
    "parse_cuda_version",
]
