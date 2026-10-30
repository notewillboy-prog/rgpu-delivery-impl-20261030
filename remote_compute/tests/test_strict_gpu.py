"""Strict GPU scheduling is opt-in and must never fall back to CPU."""
from types import SimpleNamespace
import subprocess
import pytest

from common.models import GpuInfo
from server.scheduler import Scheduler, strict_gpu_snapshot


def runtime(want=1, gpu=True):
    result = SimpleNamespace(spec=SimpleNamespace(resources=SimpleNamespace(gpu=gpu, gpu_count=want)),
                             job_id='test', assigned=None)
    result.set_gpus = lambda value: setattr(result, 'assigned', value)
    return result


@pytest.mark.parametrize('gpus,live,want', [
    ([], {}, 1),
    ([GpuInfo(index=0, name='test')], {0: 8000}, 1),
    ([GpuInfo(index=0, name='test')], {}, 1),
    ([GpuInfo(index=0, name='test')], {0: 24000}, 2),
])
def test_strict_refuses_missing_busy_or_insufficient_gpu(gpus, live, want):
    sched = Scheduler(gpus=gpus, strict_gpu=True, strict_probe=lambda: live, min_gpu_free_mb=16000)
    item = runtime(want)
    with pytest.raises(RuntimeError, match='GPU'):
        sched.acquire_gpus(item)
    assert item.assigned is None


def test_strict_allocation_and_internal_contention():
    sched = Scheduler(gpus=[GpuInfo(index=0, name='test')], strict_gpu=True, strict_probe=lambda: {0: 20000})
    item = runtime()
    assert sched.acquire_gpus(item) == [0]
    with pytest.raises(RuntimeError, match='GPU'):
        sched.acquire_gpus(runtime())
    sched.release_gpus([0])
    assert sched.acquire_gpus(runtime()) == [0]


def test_strict_rechecks_after_environment_setup():
    state = {0: 20000}
    sched = Scheduler(gpus=[GpuInfo(index=0, name='test')], strict_gpu=True, strict_probe=lambda: state)
    gpu_ids = sched.acquire_gpus(runtime())
    sched.validate_prelaunch(gpu_ids)
    state.clear()
    with pytest.raises(RuntimeError, match='改变'):
        sched.validate_prelaunch(gpu_ids)


def test_strict_failure_and_shutdown_never_call_runner():
    sched = Scheduler(gpus=[], strict_gpu=True, strict_probe=lambda: {})
    sched._stop.set()
    with pytest.raises(RuntimeError, match='停止'):
        sched.acquire_gpus(runtime())
    assert sched.acquire_gpus(runtime(gpu=False)) == []


def test_snapshot_filters_external_processes_and_fails_closed(monkeypatch):
    def run(command, **kwargs):
        if 'gpu=index,uuid,memory.free' in command[1]:
            return SimpleNamespace(returncode=0, stdout='0, GPU-aaa, 24000\n1, GPU-bbb, 20000\n')
        return SimpleNamespace(returncode=0, stdout='GPU-aaa, 1234\n')
    monkeypatch.setattr(subprocess, 'run', run)
    assert strict_gpu_snapshot() == {1: 20000}
    def broken(*args, **kwargs):
        raise subprocess.TimeoutExpired('nvidia-smi', 1)
    monkeypatch.setattr(subprocess, 'run', broken)
    with pytest.raises(RuntimeError, match='拒绝'):
        strict_gpu_snapshot()


def test_demo_default_still_supports_cpu_fallback():
    item = runtime()
    assert Scheduler(gpus=[]).acquire_gpus(item) == []
    assert 'CPU' in item.note

