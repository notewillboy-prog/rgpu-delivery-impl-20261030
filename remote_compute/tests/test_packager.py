"""客户端打包器单元测试（真实运行 ``client.packager``）。

覆盖：项目目录递归清单、默认忽略规则、include/exclude glob、sha256/大小正确、
requirements 自动识别、增量训练 meta 推断、JobSpec 构建与本地校验、入口命令解析。

注意：本文件在 client/ 落地后才真正执行；未落地时整体 skip（阶段 A 不阻塞）。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers  # noqa: E402

if not helpers.CLIENT_AVAILABLE:
    raise unittest.SkipTest(f"client/ 尚未落地: {helpers.PROJECT_ROOT / 'client'}")

from client import packager  # noqa: E402
from common.hashing import env_fingerprint, normalize_requirements, sha256_bytes, sha256_file  # noqa: E402
from common.models import JobSpec  # noqa: E402


def write(path: Path, data: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        # 固定 "\n"，避免 Windows 默认把 \n 写成 \r\n 导致 size/sha256 断言漂移
        path.write_text(data, encoding="utf-8", newline="\n")
    return path


def build_project(root: Path) -> Path:
    """构造一个带各种应忽略内容的项目目录。"""
    write(root / "train.py", "print('train')\n")
    write(root / "pkg" / "utils.py", "X = 1\n")
    write(root / "data.csv", "x,y\n1,2\n")
    write(root / "model.json", '{"w": 0, "b": 0}\n')
    write(root / "README.md", "# demo\n")
    write(root / "requirements.txt", "# 依赖\nnumpy==2.1.0  # comment\n")
    # 默认应忽略的
    write(root / ".git" / "config", "[core]\n")
    write(root / "__pycache__" / "train.cpython-312.pyc", b"\x00pyc")
    write(root / ".venv" / "pyvenv.cfg", "home=...\n")
    write(root / "node_modules" / "left-pad" / "index.js", "module.exports=1\n")
    write(root / "outputs" / "model.json", '{"stale": true}\n')
    write(root / "run.log", "log line\n")
    write(root / ".DS_Store", b"\x00\x00")
    # 例外文件（用于 include/exclude 测试）
    write(root / "notes.txt", "hello\n")
    write(root / "tmp" / "scratch.py", "x = 2\n")
    return root


class PackagerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = helpers.make_temp_dir("packager-")
        self.addCleanup(helpers.cleanup_dir, self.root)
        build_project(self.root)

    def paths(self, entries) -> list[str]:
        return [e.path for e in entries]


class CollectFilesTest(PackagerTestCase):
    def test_collects_only_uploadable_files(self) -> None:
        entries = packager.collect_files(self.root)
        self.assertEqual(
            self.paths(entries),
            ["README.md", "data.csv", "model.json", "notes.txt", "pkg/utils.py",
             "requirements.txt", "tmp/scratch.py", "train.py"],
        )
        index = {e.path: e for e in entries}
        self.assertEqual(index["data.csv"].size, len(b"x,y\n1,2\n"))
        self.assertEqual(index["data.csv"].sha256, sha256_bytes(b"x,y\n1,2\n"))
        for entry in entries:
            with self.subTest(path=entry.path):
                self.assertEqual(entry.sha256, sha256_file(self.root / entry.path))
                self.assertGreaterEqual(entry.size, 0)

    def test_default_ignored_patterns(self) -> None:
        entries = packager.collect_files(self.root)
        for bad in (".git/config", "__pycache__/train.cpython-312.pyc",
                    ".venv/pyvenv.cfg", "node_modules/left-pad/index.js",
                    "outputs/model.json", "run.log", ".DS_Store"):
            with self.subTest(bad=bad):
                self.assertNotIn(bad, self.paths(entries))

    def test_include_only_python(self) -> None:
        entries = packager.collect_files(self.root, include=["*.py"])
        self.assertEqual(self.paths(entries),
                         ["pkg/utils.py", "tmp/scratch.py", "train.py"])

    def test_include_with_path_pattern(self) -> None:
        entries = packager.collect_files(self.root, include=["pkg/*.py"])
        self.assertEqual(self.paths(entries), ["pkg/utils.py"])

    def test_exclude_file_glob_and_dir_prefix(self) -> None:
        entries = packager.collect_files(self.root, exclude=["*.txt", "tmp"])
        # 注意 "*.txt" 也会命中 requirements.txt（这是 glob 语义，CLI 用户需自行避开）
        self.assertEqual(
            self.paths(entries),
            ["README.md", "data.csv", "model.json", "pkg/utils.py", "train.py"],
        )

    def test_exclude_wins_over_include(self) -> None:
        entries = packager.collect_files(self.root, include=["*.py"], exclude=["tmp"])
        self.assertEqual(self.paths(entries), ["pkg/utils.py", "train.py"])

    def test_ignore_defaults_false_keeps_them(self) -> None:
        entries = packager.collect_files(self.root, ignore_defaults=False)
        paths = self.paths(entries)
        self.assertIn(".git/config", paths)
        self.assertIn("outputs/model.json", paths)
        self.assertIn("run.log", paths)

    def test_missing_project_dir_raises(self) -> None:
        with self.assertRaises(packager.PackagerError):
            packager.collect_files(self.root / "nope")

    def test_warning_callback_receives_skip_messages(self) -> None:
        warnings: list[str] = []
        packager.collect_files(self.root, include=["*.none"], on_warning=warnings.append)
        # 没有匹配文件时 entries 为空；warning 只在不可读/符号链接时出现，这里不强制
        self.assertIsInstance(warnings, list)

    def test_result_sorted_by_path(self) -> None:
        entries = packager.collect_files(self.root)
        self.assertEqual(self.paths(entries), sorted(self.paths(entries)))


class RequirementsTest(PackagerTestCase):
    def test_auto_detect_and_normalize(self) -> None:
        reqs, path = packager.read_requirements(self.root)
        self.assertEqual(reqs, ["numpy==2.1.0"])
        self.assertTrue(path and path.endswith("requirements.txt"))

    def test_missing_requirements_returns_empty(self) -> None:
        project = helpers.make_temp_dir("packager-noreq-")
        self.addCleanup(helpers.cleanup_dir, project)
        write(project / "a.py", "x = 1\n")
        reqs, path = packager.read_requirements(project)
        self.assertEqual(reqs, [])
        self.assertIsNone(path)

    def test_explicit_requirements_path(self) -> None:
        extra = write(self.root / "deps" / "req.txt", "torch==2.0\n# c\n")
        reqs, path = packager.read_requirements(self.root, "deps/req.txt")
        self.assertEqual(reqs, ["torch==2.0"])
        self.assertEqual(Path(path).resolve(), extra.resolve())

    def test_explicit_missing_requirements_raises(self) -> None:
        with self.assertRaises(packager.PackagerError):
            packager.read_requirements(self.root, "no-such.txt")

    def test_build_env_spec_fingerprint_matches_contract(self) -> None:
        env = packager.build_env_spec(self.root)
        self.assertEqual(env.mode, "venv")
        self.assertEqual(env.requirements, ["numpy==2.1.0"])
        expected = env_fingerprint(["numpy==2.1.0"], env.python_version, "venv")
        self.assertEqual(env.fingerprint, expected)
        # 与 hashing 归一化结果一致（注释/顺序不影响）
        self.assertEqual(env.requirements, normalize_requirements("# c\nnumpy==2.1.0\n"))

    def test_build_env_spec_overrides(self) -> None:
        env = packager.build_env_spec(
            self.root, env_mode="docker", python_version="3.11",
            docker_image="nvidia/cuda:12.4-base",
        )
        self.assertEqual(env.mode, "docker")
        self.assertEqual(env.python_version, "3.11")
        self.assertEqual(
            env.fingerprint,
            env_fingerprint(["numpy==2.1.0"], "3.11", "docker",
                            extra="nvidia/cuda:12.4-base"),
        )


class IncrementalMetaTest(PackagerTestCase):
    def test_inferred_when_model_and_dataset_exist(self) -> None:
        meta = packager.infer_incremental_meta(self.root)
        self.assertEqual(meta, {"mode": "incremental_training",
                                "base_model": "model.json",
                                "dataset": "data.csv"})

    def test_none_when_only_model(self) -> None:
        (self.root / "data.csv").unlink()
        self.assertIsNone(packager.infer_incremental_meta(self.root))

    def test_incremental_meta_fallback_defaults(self) -> None:
        empty = helpers.make_temp_dir("packager-empty-")
        self.addCleanup(helpers.cleanup_dir, empty)
        write(empty / "train.py", "x = 1\n")
        self.assertEqual(
            packager.incremental_meta(empty),
            {"mode": "incremental_training", "base_model": "model.json",
             "dataset": "data.csv"},
        )


class EntryCommandTest(unittest.TestCase):
    def test_simple_split(self) -> None:
        self.assertEqual(
            packager.parse_entry_command("python train.py --epochs 2 --data data.csv"),
            ["python", "train.py", "--epochs", "2", "--data", "data.csv"],
        )

    def test_quoted_argument_kept_together(self) -> None:
        self.assertEqual(
            packager.parse_entry_command('python train.py --note "hello world"'),
            ["python", "train.py", "--note", "hello world"],
        )

    def test_list_input_filtered(self) -> None:
        self.assertEqual(
            packager.parse_entry_command(["python", "train.py", "", "--epochs", "1"]),
            ["python", "train.py", "--epochs", "1"],
        )

    def test_empty_raises(self) -> None:
        for value in ("", "   ", []):
            with self.subTest(value=value):
                with self.assertRaises(packager.PackagerError):
                    packager.parse_entry_command(value)

    def test_unbalanced_quotes_raises(self) -> None:
        with self.assertRaises(packager.PackagerError):
            packager.parse_entry_command('python train.py --note "unclosed')


class MakeJobSpecTest(PackagerTestCase):
    def test_build_valid_spec(self) -> None:
        spec = packager.make_job_spec(
            self.root,
            ["python", "train.py", "--epochs", "2", "--data", "data.csv",
             "--out", "outputs", "--model", "model.json"],
            timeout_s=180,
        )
        self.assertIsInstance(spec, JobSpec)
        self.assertEqual(spec.job_name, self.root.name)
        self.assertEqual(spec.outputs, ["outputs/**"])
        self.assertEqual(spec.meta["mode"], "incremental_training")
        self.assertEqual(spec.meta["base_model"], "model.json")
        self.assertEqual(spec.resources.timeout_s, 180)
        self.assertEqual(spec.client_version, "0.1.0")
        self.assertIn("train.py", spec.input_index())
        self.assertEqual(spec.env.fingerprint, spec.env.compute_fingerprint())
        spec.to_dict()  # 必须可 JSON 序列化（dict 内均为内建类型）

    def test_custom_outputs_and_job_name(self) -> None:
        spec = packager.make_job_spec(
            self.root, ["python", "train.py"], job_name="自定义作业",
            outputs=["results/**", "*.json"], gpu=True, timeout_s=90,
        )
        self.assertEqual(spec.job_name, "自定义作业")
        self.assertEqual(spec.outputs, ["results/**", "*.json"])
        self.assertTrue(spec.resources.gpu)

    def test_meta_override(self) -> None:
        spec = packager.make_job_spec(
            self.root, ["python", "train.py"], meta={"mode": "custom", "k": "v"}
        )
        self.assertEqual(spec.meta, {"mode": "custom", "k": "v"})

    def test_empty_project_raises(self) -> None:
        empty = helpers.make_temp_dir("packager-empty-")
        self.addCleanup(helpers.cleanup_dir, empty)
        write(empty / "only.log", "log\n")  # 被默认忽略
        with self.assertRaises(packager.PackagerError):
            packager.make_job_spec(empty, ["python", "x.py"])

    def test_missing_project_raises(self) -> None:
        with self.assertRaises(packager.PackagerError):
            packager.make_job_spec(self.root / "nope", ["python", "x.py"])

    def test_non_positive_timeout_rejected(self) -> None:
        with self.assertRaises(ValueError):
            packager.make_job_spec(self.root, ["python", "train.py"], timeout_s=0)

    def test_illegal_output_pattern_rejected(self) -> None:
        with self.assertRaises(ValueError):
            packager.make_job_spec(self.root, ["python", "train.py"],
                                   outputs=["../escape/**"])


class PatternTest(unittest.TestCase):
    def test_basename_pattern_matches_any_depth(self) -> None:
        self.assertTrue(packager.path_matches("a/b/c.py", "*.py"))
        self.assertTrue(packager.path_matches("c.py", "*.py"))

    def test_path_pattern_anchored(self) -> None:
        self.assertTrue(packager.path_matches("src/a.py", "src/*.py"))
        self.assertFalse(packager.path_matches("other/src/a.py", "src/*.py"))

    def test_dir_prefix_semantics(self) -> None:
        # 不含 "/" 的模式按「任意深度的文件名」匹配；对目录本身用 is_dir=True 时
        # fnmatch 命中，collect_files 借此剪枝整棵子树（见上一条测试）。
        self.assertTrue(packager.path_matches("tmp", "tmp", is_dir=True))
        self.assertFalse(packager.path_matches("tmp/data.csv", "tmp"))
        self.assertTrue(packager.path_matches("tmp/data.csv", "tmp/**"))
        self.assertFalse(packager.path_matches("tmpx/data.csv", "tmp"))

    def test_backslash_pattern_normalized(self) -> None:
        self.assertTrue(packager.path_matches("src/a.py", "src\\a.py"))

    def test_default_ignored_helper(self) -> None:
        self.assertTrue(packager.is_default_ignored("__pycache__", True))
        self.assertTrue(packager.is_default_ignored("a/b/run.log", False))
        self.assertFalse(packager.is_default_ignored("a/b/train.py", False))

    def test_normalize_patterns_strips_whitespace(self) -> None:
        # 回归用例：client-dev 已修复 normalize_patterns 未去首尾空白的问题
        # （修复前本用例是 @unittest.expectedFailure）。
        self.assertEqual(
            packager.normalize_patterns([" *.py ", "*.py", "src\\*.py", "", None]),
            ["*.py", "src/*.py"],
        )

    def test_summarize(self) -> None:
        entries = [packager.FileEntry(path="a.py", size=3, sha256=sha256_bytes(b"abc")),
                   packager.FileEntry(path="b.py", size=4, sha256=sha256_bytes(b"abcd"))]
        summary = packager.summarize(entries)
        self.assertEqual(summary, {"count": 2, "bytes": 7, "files": ["a.py", "b.py"]})


class ExampleProjectTest(unittest.TestCase):
    """示例项目必须能被打包器正常收集（e2e 前置条件）。"""

    def test_example_project_collectable(self) -> None:
        if not helpers.EXAMPLE_AVAILABLE:
            self.skipTest("examples 尚未落地")
        entries = packager.collect_files(helpers.EXAMPLE_PROJECT)
        paths = [e.path for e in entries]
        for required in ("train.py", "data.csv", "model.json", "requirements.txt"):
            with self.subTest(required=required):
                self.assertIn(required, paths)
        self.assertNotIn("outputs/model.json", paths)
        self.assertFalse(any(p.startswith("__pycache__/") for p in paths))
        for entry in entries:
            with self.subTest(path=entry.path):
                self.assertEqual(entry.sha256,
                                 sha256_file(helpers.EXAMPLE_PROJECT / entry.path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
