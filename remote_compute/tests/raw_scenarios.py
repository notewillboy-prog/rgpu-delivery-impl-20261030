"""原始协议验收场景（混入类）：哈希篡改、断线续传。

同一套场景对两种服务端复用：
- ``tests/test_e2e_local.py``：对真 ``server/``（验收用）；
- ``tests/test_raw_protocol_fake.py``：对 ``tests/fake_server.py``（协议参考实现，
  在真 server 落地前先证明测试客户端逻辑本身正确）。

子类需提供 ``server_address()``，并在 ``setUpClass`` 里设置 ``cls.tmp``。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import helpers
import rawflow

ENTRY_TRAIN = (
    "python train.py --epochs 2 --data data.csv --out outputs --model model.json"
)


class RawFlowScenarios:
    token = "e2e-token"

    # -- 子类实现 ----------------------------------------------------------
    def server_address(self) -> str:  # pragma: no cover - 抽象
        raise NotImplementedError

    # -- 工具 --------------------------------------------------------------
    def _connect(self, **kwargs):
        return rawflow.connect(self.server_address(), token=self.token, **kwargs)

    def _spec(self, project: Path):
        from client import packager  # 延迟导入：假服务端自测不依赖 client 落地

        return packager.make_job_spec(
            project,
            ["python", "train.py", "--epochs", "2", "--data", "data.csv",
             "--out", "outputs", "--model", "model.json"],
            timeout_s=120,
        )

    def _project(self, name: str) -> Path:
        return helpers.copy_example_project(Path(self.tmp) / name / "proj")

    # -- 场景 1：哈希篡改 → HASH_MISMATCH ----------------------------------
    def test_01_hash_mismatch(self) -> None:
        project = self._project("tamper")
        spec = self._spec(project)
        target = spec.inputs[0]
        target.sha256 = "0" * 64  # 内容不变，清单哈希错误
        tampered = target.path

        client, _hello = self._connect()
        try:
            accepted = rawflow.submit(client, spec)
            missing = rawflow.missing_paths(accepted)
            self.assertTrue(missing, f"应缺文件: {accepted!r}")
            chunk_size = rawflow.chunk_size_of(accepted)
            index = spec.input_index()
            mismatch_path = None
            for path in missing:
                entry = index[path]
                resp = rawflow.upload_entry(client, entry, project, chunk_size)
                if resp.get("kind") == "error":
                    self.assertEqual(resp.get("code"), "HASH_MISMATCH", resp)
                    mismatch_path = path
                    break
                self.assertTrue(resp.get("done"), f"文件未完成: {resp!r}")
            self.assertIsNotNone(mismatch_path, "上传全部文件后服务端未报 HASH_MISMATCH")
            self.assertEqual(mismatch_path, tampered,
                             f"应是篡改文件 {tampered} 触发哈希错误，实际 {mismatch_path}")
        finally:
            client.close()

    # -- 场景 2：上传中断 → resume_job_id 续传 -----------------------------
    def test_02_resume_after_upload_interruption(self) -> None:
        project = self._project("resume")
        spec = self._spec(project)

        # 第一次连接：提交并完整上传第一个文件，然后直接 close 模拟掉线
        client, _hello = self._connect()
        try:
            accepted = rawflow.submit(client, spec)
            job_id = str(accepted["job_id"])
            missing = rawflow.missing_paths(accepted)
            self.assertGreaterEqual(len(missing), 2,
                                    f"续传测试至少需要 2 个待传文件: {missing}")
            chunk_size = rawflow.chunk_size_of(accepted)
            first_path = missing[0]
            resp = rawflow.upload_entry(client, spec.input_index()[first_path],
                                        project, chunk_size)
            self.assertTrue(resp.get("done"), f"首个文件应上传成功: {resp!r}")
        finally:
            client.close()

        # 第二次连接：hello → 带 resume_job_id 重新提交
        client, _hello = self._connect()
        try:
            resumed = rawflow.submit(client, spec, resume_job_id=job_id)
            self.assertEqual(resumed.get("job_id"), job_id, resumed)
            present = rawflow.present_paths(resumed)
            missing2 = rawflow.missing_paths(resumed)
            self.assertIn(first_path, present, f"已完成文件应出现在 present: {present}")
            self.assertNotIn(first_path, missing2, "已完成文件不应再次上传")
            self.assertTrue(missing2, "续传应只剩未完成文件")

            chunk_size = rawflow.chunk_size_of(resumed, default=chunk_size)
            index = spec.input_index()
            for path in missing2:
                resp = rawflow.upload_entry(client, index[path], project, chunk_size)
                self.assertEqual(resp.get("kind"), "file_ok", resp)
                self.assertTrue(resp.get("done"), f"{path} 第二轮上传未完成: {resp!r}")

            finished = rawflow.wait_finished(client, job_id)
            self.assertEqual(finished.get("state"), "succeeded", finished)
            self.assertEqual(int(finished.get("exit_code", -1)), 0)
            self.assertIn("epoch 2/2", finished.get("_events") or "")
            started = finished.get("_started") or {}
            self.assertIn((started.get("env") or {}).get("state"), ("built", "cached"))

            manifest = finished.get("outputs") or []
            self.assertTrue(manifest, "job_finished 应带产物清单")
            downloaded = rawflow.download_outputs(
                client, job_id, manifest, Path(self.tmp) / "resume" / "dl")
            self.assertIn("outputs/model.json", downloaded)
            self.assertIn("outputs/metrics.json", downloaded)

            status = rawflow.query_status(client, job_id)
            self.assertEqual(status.get("state"), "succeeded")
        finally:
            client.close()
