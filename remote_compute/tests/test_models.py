"""数据模型单元测试：FileEntry 路径安全、JobSpec 往返、validate_job_spec、
OutputEntry / GpuInfo / EnvSpec / Resources 往返、清单指纹。

契约来源：``docs/PROTOCOL.md`` 3.1 节 + ``common/models.py``。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers  # noqa: E402

from common.hashing import env_fingerprint, sha256_bytes  # noqa: E402
from common.models import (  # noqa: E402
    EnvSpec,
    FileEntry,
    GpuInfo,
    JobSpec,
    JobState,
    OutputEntry,
    Resources,
    fingerprint_of_manifest,
    normalize_relpath,
    validate_job_spec,
)

HEX_A = "a" * 64


def make_entry(path: str = "train.py", data: bytes = b"print('hi')\n",
               executable: bool = False) -> FileEntry:
    return FileEntry(path=path, size=len(data), sha256=sha256_bytes(data),
                     executable=executable)


def make_env(requirements=("numpy==2.1.0",), python_version: str = "3.12") -> EnvSpec:
    env = EnvSpec(requirements=list(requirements), python_version=python_version)
    env.fingerprint = env.compute_fingerprint()
    return env


def make_spec(**overrides) -> JobSpec:
    kwargs = {
        "job_name": "incremental-train-demo",
        "entry_argv": ["python", "train.py", "--epochs", "2"],
        "inputs": [make_entry("train.py"), make_entry("data.csv", b"x,y\n1,2\n")],
        "env": make_env(),
        "resources": Resources(gpu=False, timeout_s=180),
        "outputs": ["outputs/**"],
        "workdir": ".",
        "meta": {"mode": "incremental_training", "base_model": "model.json"},
    }
    kwargs.update(overrides)
    return JobSpec(**kwargs)


class FileEntryTest(unittest.TestCase):
    def test_valid_entry_and_to_dict(self) -> None:
        entry = make_entry("train.py")
        self.assertEqual(entry.to_dict()["path"], "train.py")
        self.assertEqual(entry.to_dict()["size"], len(b"print('hi')\n"))
        self.assertEqual(entry.to_dict()["sha256"], sha256_bytes(b"print('hi')\n"))
        self.assertNotIn("executable", entry.to_dict())  # False 时不写，保持 JSON 精简

    def test_executable_flag_roundtrip(self) -> None:
        entry = make_entry("run.sh", b"#!/bin/sh\n", executable=True)
        self.assertTrue(entry.to_dict()["executable"])
        self.assertEqual(FileEntry.from_dict(entry.to_dict()), entry)
        # 缺省 executable -> False
        plain = FileEntry.from_dict({"path": "a.py", "size": 1, "sha256": HEX_A})
        self.assertFalse(plain.executable)

    def test_windows_separators_normalized_to_posix(self) -> None:
        entry = make_entry(r"pkg\subdir\file.py")
        self.assertEqual(entry.path, "pkg/subdir/file.py")
        self.assertEqual(FileEntry.from_dict(entry.to_dict()).path, "pkg/subdir/file.py")

    def test_nested_and_dotted_relative_paths_allowed(self) -> None:
        for path in ("a", "a/b/c.py", "outputs/model.json", "./a.txt"):
            with self.subTest(path=path):
                entry = make_entry(path)
                self.assertTrue(entry.path)

    def test_rejects_parent_traversal(self) -> None:
        for path in ("..", "../etc/passwd", "a/../b", "a/..", "a/b/../../c", "..\\windows\\x"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError) as ctx:
                    make_entry(path)
                self.assertIn("unsafe", str(ctx.exception))

    def test_rejects_absolute_paths(self) -> None:
        # 契约：inputs[].path 一律 POSIX 相对路径。这里覆盖 POSIX 绝对路径
        # 与 Windows 反斜杠绝对路径（实现会把反斜杠转成 "/"，因此同样被拦）。
        for path in ("/etc/passwd", "/", "\\windows\\system32"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError) as ctx:
                    make_entry(path)
                self.assertIn("unsafe", str(ctx.exception))

    def test_rejects_windows_drive_and_unc_absolute_paths(self) -> None:
        # 契约（PROTOCOL 3.1）：inputs[].path 一律 POSIX 相对路径。
        # Windows 盘符绝对路径与 UNC 共享路径同样必须拒绝（Lead 已补 _WIN_DRIVE 校验）。
        for path in ("C:/tmp/x.py", "c:\\tmp\\x.py", "Z:/", "//server/share/x.py"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError) as ctx:
                    make_entry(path)
                self.assertIn("unsafe", str(ctx.exception))

    def test_rejects_empty_path(self) -> None:
        with self.assertRaises(ValueError):
            make_entry("")

    def test_rejects_negative_size(self) -> None:
        with self.assertRaises(ValueError):
            FileEntry(path="a.py", size=-1, sha256=HEX_A)

    def test_rejects_bad_sha256(self) -> None:
        for bad in ("", "abc", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "z" * 64):
            with self.subTest(sha=bad):
                with self.assertRaises(ValueError):
                    FileEntry(path="a.py", size=1, sha256=bad)

    def test_from_dict_requires_core_fields(self) -> None:
        for missing in ("path", "size", "sha256"):
            data = {"path": "a.py", "size": 1, "sha256": HEX_A}
            del data[missing]
            with self.subTest(missing=missing):
                with self.assertRaises(KeyError):
                    FileEntry.from_dict(data)


class OutputEntryGpuInfoTest(unittest.TestCase):
    def test_output_entry_roundtrip(self) -> None:
        entry = OutputEntry(path="outputs/model.json", size=612, sha256=HEX_A)
        self.assertEqual(entry.kind, "file")
        self.assertEqual(OutputEntry.from_dict(entry.to_dict()), entry)
        dir_entry = OutputEntry(path="outputs/sub/", size=0, sha256=HEX_A, kind="dir_entry")
        self.assertEqual(OutputEntry.from_dict(dir_entry.to_dict()).kind, "dir_entry")
        # kind 缺省 -> file
        self.assertEqual(
            OutputEntry.from_dict({"path": "a", "size": 1, "sha256": HEX_A}).kind, "file"
        )

    def test_output_entry_json_serializable(self) -> None:
        entry = OutputEntry(path="outputs/metrics.json", size=3, sha256=HEX_A)
        self.assertEqual(json.loads(json.dumps(entry.to_dict())), entry.to_dict())

    def test_gpu_info_roundtrip(self) -> None:
        gpu = GpuInfo(index=0, name="NVIDIA A100", memory_total_mb=40960,
                      memory_free_mb=40000, driver="550.54", cuda="12.4")
        self.assertEqual(GpuInfo.from_dict(gpu.to_dict()), gpu)
        minimal = GpuInfo(index=1, name="CPU")
        restored = GpuInfo.from_dict(minimal.to_dict())
        self.assertEqual(restored.memory_total_mb, 0)
        self.assertEqual(restored.driver, "")
        self.assertEqual(restored, minimal)


class EnvResourcesTest(unittest.TestCase):
    def test_env_spec_roundtrip(self) -> None:
        env = EnvSpec(mode="docker", python_version="3.11",
                      requirements=["numpy==2.1.0"], fingerprint=HEX_A,
                      env_vars={"CUDA_VISIBLE_DEVICES": "0"},
                      docker_image="pytorch/pytorch:2.4", reuse=False)
        self.assertEqual(EnvSpec.from_dict(env.to_dict()), env)

    def test_env_spec_from_empty_dict_uses_defaults(self) -> None:
        env = EnvSpec.from_dict({})
        self.assertEqual(env.mode, "venv")
        self.assertEqual(env.python_version, "3.11")
        self.assertEqual(env.requirements, [])
        self.assertTrue(env.reuse)

    def test_env_spec_compute_fingerprint(self) -> None:
        env = EnvSpec(requirements=["numpy==2.1.0"], python_version="3.12", mode="venv")
        self.assertEqual(env.compute_fingerprint(),
                         env_fingerprint(["numpy==2.1.0"], "3.12", "venv"))
        docker_env = EnvSpec(requirements=["numpy==2.1.0"], python_version="3.12",
                             mode="docker", docker_image="nvidia/cuda:12.4-base")
        self.assertEqual(
            docker_env.compute_fingerprint(),
            env_fingerprint(["numpy==2.1.0"], "3.12", "docker",
                            extra="nvidia/cuda:12.4-base"),
        )

    def test_resources_defaults_and_roundtrip(self) -> None:
        res = Resources()
        self.assertFalse(res.gpu)
        self.assertEqual(res.gpu_count, 1)
        self.assertEqual(res.timeout_s, 3600)
        self.assertEqual(res.cpus, 0)
        self.assertEqual(res.memory_gb, 0.0)
        custom = Resources(gpu=True, gpu_count=2, timeout_s=600, cpus=8, memory_gb=32.5)
        self.assertEqual(Resources.from_dict(custom.to_dict()), custom)
        self.assertEqual(Resources.from_dict({}), Resources())


class JobSpecTest(unittest.TestCase):
    def test_full_roundtrip_through_json(self) -> None:
        spec = make_spec()
        restored = JobSpec.from_dict(json.loads(json.dumps(spec.to_dict(), ensure_ascii=False)))
        self.assertEqual(restored, spec)
        self.assertEqual([e.path for e in restored.inputs], ["train.py", "data.csv"])
        self.assertEqual(restored.env.fingerprint, spec.env.fingerprint)
        self.assertEqual(restored.meta["mode"], "incremental_training")

    def test_total_input_bytes_and_index(self) -> None:
        spec = make_spec()
        self.assertEqual(spec.total_input_bytes(), 12 + 8)
        index = spec.input_index()
        self.assertEqual(set(index), {"train.py", "data.csv"})
        self.assertEqual(index["data.csv"].size, 8)

    def test_from_dict_defaults_for_optional_sections(self) -> None:
        spec = JobSpec.from_dict({"job_name": "j", "entry_argv": ["python", "-V"]})
        self.assertEqual(spec.inputs, [])
        self.assertEqual(spec.workdir, ".")
        self.assertEqual(spec.outputs, ["outputs/**"])
        self.assertEqual(spec.client_version, "0.1.0")

    def test_manifest_fingerprint_order_insensitive(self) -> None:
        entries = [make_entry("b.py", b"b"), make_entry("a.py", b"a")]
        self.assertEqual(fingerprint_of_manifest(entries),
                         fingerprint_of_manifest(list(reversed(entries))))
        changed = [make_entry("b.py", b"b2"), make_entry("a.py", b"a")]
        self.assertNotEqual(fingerprint_of_manifest(entries),
                            fingerprint_of_manifest(changed))

    def test_normalize_relpath(self) -> None:
        self.assertEqual(normalize_relpath(r"a\b\c.py"), "a/b/c.py")
        self.assertEqual(normalize_relpath("a/b/./c.py"), "a/b/c.py")
        self.assertEqual(normalize_relpath("a/b/../c.py"), "a/c.py")


class ValidateJobSpecTest(unittest.TestCase):
    def test_valid_spec_fills_fingerprint(self) -> None:
        env = EnvSpec(requirements=["numpy==2.1.0"], python_version="3.12")
        self.assertEqual(env.fingerprint, "")
        spec = make_spec(env=env)
        validate_job_spec(spec)
        self.assertEqual(env.fingerprint, env.compute_fingerprint())
        self.assertEqual(len(env.fingerprint), 64)

    def test_matching_fingerprint_passes(self) -> None:
        validate_job_spec(make_spec())  # 不抛

    def test_blank_job_name_rejected(self) -> None:
        for name in ("", "   ", "\t\n"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as ctx:
                    validate_job_spec(make_spec(job_name=name))
                self.assertIn("job_name", str(ctx.exception))

    def test_empty_entry_argv_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_job_spec(make_spec(entry_argv=[]))
        self.assertIn("entry_argv", str(ctx.exception))

    def test_empty_inputs_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_job_spec(make_spec(inputs=[]))
        self.assertIn("inputs", str(ctx.exception))

    def test_duplicate_input_paths_rejected(self) -> None:
        duplicated = [make_entry("a.py", b"one"), make_entry("a.py", b"two")]
        with self.assertRaises(ValueError) as ctx:
            validate_job_spec(make_spec(inputs=duplicated))
        self.assertIn("重复", str(ctx.exception))
        # 跨平台写法归一化后重复也必须识别
        dup2 = [make_entry("a/b.py", b"one"), make_entry(r"a\b.py", b"two")]
        with self.assertRaises(ValueError):
            validate_job_spec(make_spec(inputs=dup2))

    def test_unsupported_env_mode_rejected(self) -> None:
        env = make_env()
        env.mode = "conda"
        with self.assertRaises(ValueError) as ctx:
            validate_job_spec(make_spec(env=env))
        self.assertIn("env.mode", str(ctx.exception))

    def test_fingerprint_mismatch_rejected(self) -> None:
        env = make_env()
        env.fingerprint = "0" * 64
        with self.assertRaises(ValueError) as ctx:
            validate_job_spec(make_spec(env=env))
        self.assertIn("fingerprint", str(ctx.exception))

    def test_nonpositive_timeout_rejected(self) -> None:
        for timeout in (0, -1):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError) as ctx:
                    validate_job_spec(make_spec(resources=Resources(timeout_s=timeout)))
                self.assertIn("timeout_s", str(ctx.exception))

    def test_illegal_output_patterns_rejected(self) -> None:
        for pattern in ("/abs/outputs/**", "../escape/**", "a/../../b/**"):
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError) as ctx:
                    validate_job_spec(make_spec(outputs=[pattern]))
                self.assertIn("产物", str(ctx.exception))

    def test_legal_output_patterns_accepted(self) -> None:
        for patterns in (["outputs/**"], ["*.json", "outputs/**"], []):
            with self.subTest(patterns=patterns):
                validate_job_spec(make_spec(outputs=list(patterns)))


class JobStateTest(unittest.TestCase):
    def test_terminal_states(self) -> None:
        terminal = {JobState.SUCCEEDED, JobState.FAILED, JobState.TIMEOUT, JobState.CANCELLED}
        for state in JobState:
            with self.subTest(state=state):
                self.assertEqual(state.terminal, state in terminal)
        self.assertFalse(JobState.RUNNING.terminal)
        self.assertFalse(JobState.QUEUED.terminal)

    def test_state_values_frozen(self) -> None:
        self.assertEqual({s.value for s in JobState},
                         {"queued", "uploading", "env_building", "running",
                          "succeeded", "failed", "timeout", "cancelled"})
        self.assertEqual(JobState("succeeded"), JobState.SUCCEEDED)
        self.assertEqual(JobState.SUCCEEDED, "succeeded")  # str 枚举，可直接序列化


if __name__ == "__main__":
    unittest.main(verbosity=2)
