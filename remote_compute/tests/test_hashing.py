"""哈希与环境指纹单元测试。

覆盖：sha256 已知向量、stream/file 一致性、requirements 归一化（注释/空行/多行字符串/
排序去重/选项行）、环境指纹稳定性与敏感性、human_size。

回归背景：``normalize_requirements`` 曾把 ``["#c\\nnumpy==2.1.0\\n"]`` 这种
「列表元素里含多行」的输入当成一行处理（Lead 已在 ``common/hashing.py`` 修复），
本文件把它固化成回归用例。
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers  # noqa: E402

from common.hashing import (  # noqa: E402
    env_fingerprint,
    human_size,
    normalize_requirements,
    sha256_bytes,
    sha256_file,
    sha256_stream,
)


class Sha256Test(unittest.TestCase):
    def test_known_vectors(self) -> None:
        self.assertEqual(
            sha256_bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        self.assertEqual(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        self.assertEqual(
            sha256_bytes(b"The quick brown fox jumps over the lazy dog"),
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
        )

    def test_sha256_stream_matches_bytes(self) -> None:
        data = bytes(range(256)) * 100
        self.assertEqual(sha256_stream(io.BytesIO(data)), sha256_bytes(data))
        # 任意 chunk 大小结果一致（分块读不能改变哈希）
        for chunk in (1, 7, 1024, len(data)):
            with self.subTest(chunk=chunk):
                self.assertEqual(sha256_stream(io.BytesIO(data), chunk_size=chunk),
                                 sha256_bytes(data))

    def test_sha256_file_matches_bytes(self) -> None:
        tmp = helpers.make_temp_dir("hash-")
        self.addCleanup(helpers.cleanup_dir, tmp)
        target = tmp / "blob.bin"
        data = b"rgpu\x00payload\xff" * 1000
        target.write_bytes(data)
        self.assertEqual(sha256_file(target), sha256_bytes(data))
        for chunk in (1, 13, 4096):
            with self.subTest(chunk=chunk):
                self.assertEqual(sha256_file(target, chunk_size=chunk), sha256_bytes(data))

    def test_sha256_file_missing_raises(self) -> None:
        tmp = helpers.make_temp_dir("hash-")
        self.addCleanup(helpers.cleanup_dir, tmp)
        with self.assertRaises(FileNotFoundError):
            sha256_file(tmp / "nope.bin")

    def test_hash_is_64_lowercase_hex(self) -> None:
        digest = sha256_bytes(b"hello")
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, digest.lower())
        int(digest, 16)  # 合法十六进制


class NormalizeRequirementsTest(unittest.TestCase):
    def test_none_and_empty(self) -> None:
        self.assertEqual(normalize_requirements(None), [])
        self.assertEqual(normalize_requirements(""), [])
        self.assertEqual(normalize_requirements([]), [])
        self.assertEqual(normalize_requirements("\n\n  \n\t\n"), [])

    def test_strip_blank_and_full_line_comments(self) -> None:
        text = """
        # 这是注释
        numpy==2.1.0

        # torch 注释
          # 缩进注释
        torch==2.0.1
        """
        self.assertEqual(normalize_requirements(text), ["numpy==2.1.0", "torch==2.0.1"])

    def test_inline_comment_removed(self) -> None:
        text = "numpy==2.1.0  # pinned for cuda\nscipy>=1.0  # 数值计算\n"
        self.assertEqual(normalize_requirements(text), ["numpy==2.1.0", "scipy>=1.0"])

    def test_single_space_inline_comment_also_removed(self) -> None:
        # pip 语义：空白后的 # 即注释。单空格与多空格写法必须归一化成同一指纹输入。
        text = "numpy==2.1.0 # pinned\nscipy>=1.0\t# tab comment\n"
        self.assertEqual(normalize_requirements(text), ["numpy==2.1.0", "scipy>=1.0"])
        self.assertEqual(
            env_fingerprint("numpy==2.1.0 # pinned\n", "3.12"),
            env_fingerprint("numpy==2.1.0\n", "3.12"),
        )

    def test_inline_comment_only_whitespace_left(self) -> None:
        self.assertEqual(normalize_requirements("pkg==1.0  # note\n"), ["pkg==1.0"])

    def test_dedupe_and_sort(self) -> None:
        lines = ["torch==2.0", " numpy==2.1.0 ", "numpy==2.1.0", "a==1", "torch==2.0"]
        self.assertEqual(normalize_requirements(lines), ["a==1", "numpy==2.1.0", "torch==2.0"])

    def test_multiline_string_input(self) -> None:
        text = "numpy==2.1.0\n# comment\ntorch==2.0\n\npandas==2.2.0\n"
        self.assertEqual(
            normalize_requirements(text),
            ["numpy==2.1.0", "pandas==2.2.0", "torch==2.0"],
        )

    def test_list_item_containing_multiple_lines_is_split(self) -> None:
        # 回归用例：元素内部的多行必须逐行归一化
        items = ["#c\nnumpy==2.1.0\n", "torch==2.0  # note\npandas==2.2.0"]
        self.assertEqual(
            normalize_requirements(items),
            ["numpy==2.1.0", "pandas==2.2.0", "torch==2.0"],
        )

    def test_option_lines_preserved(self) -> None:
        text = "--index-url https://pypi.example/simple\n-r base.txt\nnumpy==2.1.0\n"
        self.assertEqual(
            normalize_requirements(text),
            ["--index-url https://pypi.example/simple", "-r base.txt", "numpy==2.1.0"],
        )

    def test_duplicate_with_and_without_comment_collapses(self) -> None:
        text = "numpy==2.1.0\nnumpy==2.1.0  # same\n"
        self.assertEqual(normalize_requirements(text), ["numpy==2.1.0"])

    def test_output_is_sorted_and_deterministic(self) -> None:
        text = "z==1\na==1\nm==1\n"
        first = normalize_requirements(text)
        self.assertEqual(first, ["a==1", "m==1", "z==1"])
        self.assertEqual(normalize_requirements(text), first)


class EnvFingerprintTest(unittest.TestCase):
    def test_deterministic_hex(self) -> None:
        fp1 = env_fingerprint(["numpy==2.1.0"], "3.12", "venv")
        fp2 = env_fingerprint(["numpy==2.1.0"], "3.12", "venv")
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)
        self.assertEqual(fp1, fp1.lower())
        int(fp1, 16)

    def test_order_duplicates_comments_do_not_change_fingerprint(self) -> None:
        base = env_fingerprint(["numpy==2.1.0", "torch==2.0"], "3.12")
        variants = [
            ["torch==2.0", "numpy==2.1.0"],
            "numpy==2.1.0\ntorch==2.0\n",
            "# deps\n\n torch==2.0  # gpu \nnumpy==2.1.0\n",
            ["numpy==2.1.0", "torch==2.0", "numpy==2.1.0"],
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertEqual(env_fingerprint(variant, "3.12"), base)

    def test_mode_case_and_whitespace_normalized(self) -> None:
        self.assertEqual(
            env_fingerprint(["numpy==2.1.0"], "3.12", "VENV"),
            env_fingerprint(["numpy==2.1.0"], "3.12", " venv "),
        )
        self.assertEqual(
            env_fingerprint(["numpy==2.1.0"], " 3.12 "),
            env_fingerprint(["numpy==2.1.0"], "3.12"),
        )

    def test_sensitivity_to_requirements(self) -> None:
        base = env_fingerprint(["numpy==2.1.0"], "3.12")
        self.assertNotEqual(base, env_fingerprint(["numpy==2.1.1"], "3.12"))
        self.assertNotEqual(base, env_fingerprint(["numpy==2.1.0", "torch==2.0"], "3.12"))
        self.assertNotEqual(base, env_fingerprint([], "3.12"))

    def test_sensitivity_to_python_version(self) -> None:
        self.assertNotEqual(
            env_fingerprint(["numpy==2.1.0"], "3.11"),
            env_fingerprint(["numpy==2.1.0"], "3.12"),
        )

    def test_sensitivity_to_mode(self) -> None:
        self.assertNotEqual(
            env_fingerprint(["numpy==2.1.0"], "3.12", "venv"),
            env_fingerprint(["numpy==2.1.0"], "3.12", "docker"),
        )

    def test_sensitivity_to_extra(self) -> None:
        self.assertNotEqual(
            env_fingerprint(["numpy==2.1.0"], "3.12", extra=""),
            env_fingerprint(["numpy==2.1.0"], "3.12", extra="cuda12.4"),
        )
        self.assertEqual(
            env_fingerprint(["numpy==2.1.0"], "3.12", extra=" cuda12.4 "),
            env_fingerprint(["numpy==2.1.0"], "3.12", extra="cuda12.4"),
        )

    def test_none_requirements_equals_empty(self) -> None:
        self.assertEqual(
            env_fingerprint(None, "3.12"),
            env_fingerprint([], "3.12"),
        )

    def test_unknown_mode_raises_value_error(self) -> None:
        for bad in ("conda", "virtualenv", "k8s"):
            with self.subTest(mode=bad):
                with self.assertRaises(ValueError):
                    env_fingerprint(["numpy==2.1.0"], "3.12", bad)

    def test_empty_mode_falls_back_to_venv(self) -> None:
        # 实现：mode 为空时按默认 venv 处理（不是报错）
        self.assertEqual(
            env_fingerprint(["numpy==2.1.0"], "3.12", ""),
            env_fingerprint(["numpy==2.1.0"], "3.12", "venv"),
        )

    def test_matches_envspec_compute_fingerprint(self) -> None:
        from common.models import EnvSpec

        env = EnvSpec(requirements=["numpy==2.1.0"], python_version="3.12", mode="venv")
        self.assertEqual(env.compute_fingerprint(), env_fingerprint(["numpy==2.1.0"], "3.12"))
        docker_env = EnvSpec(requirements=["numpy==2.1.0"], python_version="3.12",
                             mode="docker", docker_image="pytorch/pytorch:2.4-cuda12.1")
        self.assertEqual(
            docker_env.compute_fingerprint(),
            env_fingerprint(["numpy==2.1.0"], "3.12", "docker",
                            extra="pytorch/pytorch:2.4-cuda12.1"),
        )


class HumanSizeTest(unittest.TestCase):
    def test_known_values(self) -> None:
        cases = [
            (0, "0 B"),
            (1, "1 B"),
            (512, "512 B"),
            (1023, "1023 B"),
            (1024, "1.0 KiB"),
            (1536, "1.5 KiB"),
            (1024 ** 2, "1.0 MiB"),
            (2 * 1024 ** 3, "2.0 GiB"),
            (1024 ** 4, "1.0 TiB"),
            (1024 ** 5, "1024.0 TiB"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(human_size(value), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
