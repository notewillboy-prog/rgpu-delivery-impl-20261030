"""端到端验收：真起服务端 + 真跑客户端 CLI / 原始协议。

覆盖（对应 TASK_TREE 验收项 2/3/4 与 PROTOCOL.md 3.2–3.6）：

1. 成功链路：client CLI 提交 ``examples/incremental_train`` → 退出码 0 → 流式 epoch 日志 →
   下载 outputs/model.json + metrics.json → sha256 与远端清单一致 → 增量语义成立
   （远端加载本地训练后的基座模型，runs/epochs_total/loss 接续）。
2. 环境复用：二次提交同一 requirements，``job_started.env.state == "cached"``。
3. 失败场景：错误 token（退出码 4 / BAD_AUTH）、非零退出码（退出码 1 / failed）、
   任务超时（退出码 1 / timeout）、哈希篡改（HASH_MISMATCH）。
4. 断线续传：上传中断 → 重连 → ``resume_job_id`` → present 跳过已完成文件 →
   续传剩余文件 → 作业成功 → 产物 sha256 与 ``job_finished`` 清单一致。

运行前提：``server/``、``client/``、``examples/`` 已落地。任一缺失或设置
``RGPU_SKIP_E2E=1`` 时整模块 skip（不会让阶段 A 变红）。
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers  # noqa: E402
from raw_scenarios import RawFlowScenarios  # noqa: E402

# ---------------------------------------------------------------------------
# 模块级开关与共享服务端
# ---------------------------------------------------------------------------
E2E_PORT = int(os.environ.get("RGPU_E2E_PORT", "18766"))
E2E_TOKEN = os.environ.get("RGPU_E2E_TOKEN", "e2e-token")
HEALTHY_TIMEOUT = 240.0
SERVER_READY_TIMEOUT = 20.0

E2E_SKIP_REASON = None
if not helpers.SERVER_AVAILABLE:
    E2E_SKIP_REASON = "server/ 尚未落地（等 server-dev）"
elif not helpers.CLIENT_AVAILABLE:
    E2E_SKIP_REASON = "client/ 尚未落地"
elif not helpers.EXAMPLE_AVAILABLE:
    E2E_SKIP_REASON = "examples/incremental_train 尚未落地"
elif os.environ.get("RGPU_SKIP_E2E", "").strip().lower() in ("1", "true", "yes", "on"):
    E2E_SKIP_REASON = "RGPU_SKIP_E2E=1（跳过端到端）"

if E2E_SKIP_REASON is None:
    from client import packager  # noqa: E402

_SERVER = None  # type: ignore[assignment]


def server() -> helpers.ServerProcess:
    if _SERVER is None:  # pragma: no cover - 只有 setUpModule 失败才会走到
        raise RuntimeError("e2e 服务端未启动")
    return _SERVER


def setUpModule() -> None:
    global _SERVER
    if E2E_SKIP_REASON is not None:
        return
    _SERVER = helpers.ServerProcess(
        port=E2E_PORT,
        token=E2E_TOKEN,
        data_dir=helpers.make_temp_dir("e2e-server-"),
        ready_timeout=SERVER_READY_TIMEOUT,
    )
    _SERVER.start()


def tearDownModule() -> None:
    global _SERVER
    if _SERVER is not None:
        _SERVER.stop(cleanup_data_dir=True)
        _SERVER = None


skip_e2e = unittest.skipIf(E2E_SKIP_REASON is not None, E2E_SKIP_REASON or "")

ENTRY_TRAIN = (
    "python train.py --epochs 2 --data data.csv --out outputs --model model.json"
)


# ---------------------------------------------------------------------------
# 通用断言 / CLI 封装
# ---------------------------------------------------------------------------
def result_json(proc) -> dict:
    """从 ``--json`` 模式的 stdout 解析结果；失败时带上完整进程输出。"""
    lines = [line for line in (proc.stdout or "").splitlines()
             if line.strip().startswith("{")]
    if not lines:
        raise AssertionError("client --json 未在 stdout 输出 JSON 结果\n"
                            + helpers.describe_process(proc))
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:  # pragma: no cover - 防御
        raise AssertionError(f"client JSON 结果无法解析: {exc}\n"
                             + helpers.describe_process(proc)) from exc


def run_submit(project_dir: Path, download_dir: Path, *, entry: str = ENTRY_TRAIN,
               timeout_s: int = 180, token: str = E2E_TOKEN,
               extra: tuple = ()) -> "tuple[dict, object]":
    proc = helpers.run_client(
        "submit", str(project_dir),
        "--entry", entry,
        "--server", f"127.0.0.1:{E2E_PORT}",
        "--token", token,
        "--timeout", str(timeout_s),
        "--download-dir", str(download_dir),
        "--json",
        *extra,
        timeout=HEALTHY_TIMEOUT,
    )
    return result_json(proc), proc


def prepare_example_project(parent: Path, *, local_epochs: int = 2) -> Path:
    """复制示例项目并在本地先训练一次（制造「已有基座权重」的增量起点）。"""
    project = helpers.copy_example_project(parent / "proj")
    proc = helpers.run_local_train(project, "--epochs", str(local_epochs))
    if proc.returncode != 0:
        raise AssertionError("本地预训练失败（e2e 前置条件）\n"
                             + helpers.describe_process(proc))
    model = helpers.read_json(project / "model.json")
    assert model["runs"] == 1, f"本地预训练后 runs 应为 1，实际 {model}"
    assert model["epochs_total"] == local_epochs
    return project


def chainlog(proc) -> str:
    return helpers.output_of(proc)


@skip_e2e
class E2ECliTest(unittest.TestCase):
    """客户端 CLI + 真实服务端的成功/失败链路。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = helpers.make_temp_dir("e2e-cli-")
        cls.addClassCleanup(helpers.cleanup_dir, cls.tmp)

    # -- 1. 成功链路 + 增量语义 + 产物下载校验 ------------------------------
    def test_01_success_incremental_and_download(self) -> None:
        project = prepare_example_project(self.tmp / "success")
        local_model = helpers.read_json(project / "model.json")
        download_dir = self.tmp / "success-dl"

        result, proc = run_submit(project, download_dir)

        self.assertEqual(proc.returncode, 0, helpers.describe_process(proc))
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("state"), "succeeded", result)
        self.assertEqual(result.get("client_exit"), 0)
        self.assertEqual(result.get("exit_code"), 0)
        job_id = result.get("job_id") or ""
        self.assertTrue(job_id.startswith("job-"), f"job_id 格式异常: {job_id!r}")

        # 流式日志：真实打印了每个 epoch
        logs = chainlog(proc)
        self.assertIn("epoch 1/2", logs)
        self.assertIn("epoch 2/2", logs)
        self.assertIn("加载基座模型", logs)

        # 产物已下载，且与客户端返回的远端清单 sha256 一致
        outputs = {item["path"]: item for item in result.get("outputs") or []}
        for rel in ("outputs/model.json", "outputs/metrics.json"):
            with self.subTest(path=rel):
                self.assertIn(rel, outputs)
                local_path = Path(outputs[rel]["local_path"])
                self.assertTrue(local_path.is_file(), f"产物未下载: {local_path}")
                self.assertEqual(local_path.resolve(),
                                 (download_dir / rel).resolve())
                actual = helpers.sha256_file(local_path)
                self.assertEqual(actual, outputs[rel]["sha256"],
                                 f"{rel} 下载后 sha256 与远端清单不一致")
                self.assertTrue(outputs[rel].get("verified"))

        # 增量语义：远端在本地训练后的权重上继续训练
        metrics = helpers.read_json(download_dir / "outputs" / "metrics.json")
        self.assertTrue(metrics["base_model_loaded"])
        self.assertEqual(metrics["runs"], 2)
        self.assertEqual(metrics["epochs_total"], 4)
        self.assertAlmostEqual(float(metrics["start_loss"]),
                               float(local_model["final_loss"]), places=5)
        self.assertLess(float(metrics["final_loss"]), float(metrics["start_loss"]))

        model = helpers.read_json(download_dir / "outputs" / "model.json")
        self.assertEqual(model["runs"], 2)
        self.assertEqual(model["epochs_total"], 4)
        self.assertEqual(model["weights"] if "weights" in model else
                         {"w": model["w"], "b": model["b"]}, metrics["weights"])

        # 服务端 state.json 里的产物清单也要与下载文件一致（若字段存在则强制校验）
        state_file = server().data_dir / "jobs" / job_id / "state.json"
        if state_file.is_file():
            state = helpers.read_json(state_file)
            manifest = state.get("outputs") or []
            if manifest:
                by_path = {str(item.get("path")): item for item in manifest
                           if isinstance(item, dict)}
                for rel, item in outputs.items():
                    with self.subTest(state_path=rel):
                        self.assertIn(rel, by_path)
                        self.assertEqual(item["sha256"], by_path[rel].get("sha256"))

        # 3.5 状态查询：作业已结束也能查到 succeeded
        status = helpers.run_client(
            "status", job_id,
            "--server", f"127.0.0.1:{E2E_PORT}", "--token", E2E_TOKEN, "--json",
            timeout=60,
        )
        self.assertEqual(status.returncode, 0, helpers.describe_process(status))
        status_data = result_json(status)
        self.assertEqual(status_data.get("state"), "succeeded", status_data)

    # -- 2. 环境复用 -------------------------------------------------------
    def test_02_env_reuse_cached(self) -> None:
        project = helpers.copy_example_project(self.tmp / "reuse" / "proj")

        first, first_proc = run_submit(project, self.tmp / "reuse" / "dl1")
        self.assertEqual(first_proc.returncode, 0, helpers.describe_process(first_proc))
        first_env = first.get("env") or {}
        self.assertIn(str(first_env.get("state")), ("built", "cached"),
                      f"第一次 job_started 缺少合法 env.state: {first_env}")

        second, second_proc = run_submit(project, self.tmp / "reuse" / "dl2")
        self.assertEqual(second_proc.returncode, 0,
                         helpers.describe_process(second_proc))
        second_env = second.get("env") or {}
        self.assertEqual(second_env.get("state"), "cached",
                         f"二次提交应复用环境（cached），实际 {second_env}")
        self.assertEqual(second_env.get("fingerprint"), first_env.get("fingerprint"))
        self.assertIn("cached", chainlog(second_proc))

    # -- 3. 错误 token ------------------------------------------------------
    def test_03_wrong_token_exit_4(self) -> None:
        project = helpers.copy_example_project(self.tmp / "badauth" / "proj")
        result, proc = run_submit(project, self.tmp / "badauth" / "dl",
                                  token="definitely-wrong-token")
        self.assertEqual(proc.returncode, 4, helpers.describe_process(proc))
        self.assertFalse(result.get("ok"))
        self.assertEqual((result.get("error") or {}).get("code"), "BAD_AUTH", result)
        self.assertIn("BAD_AUTH", chainlog(proc))

    # -- 4. 非零退出码 ------------------------------------------------------
    def test_04_nonzero_exit_code(self) -> None:
        project = self.tmp / "failjob" / "proj"
        project.mkdir(parents=True, exist_ok=True)
        (project / "fail.py").write_text(
            "import sys\nprint('about to fail', flush=True)\nsys.exit(3)\n",
            encoding="utf-8",
        )
        result, proc = run_submit(project, self.tmp / "failjob" / "dl",
                                  entry="python fail.py", timeout_s=60)
        self.assertEqual(proc.returncode, 1, helpers.describe_process(proc))
        self.assertEqual(result.get("state"), "failed", result)
        self.assertEqual(result.get("exit_code"), 3, result)
        self.assertEqual((result.get("error") or {}).get("code"), "JOB_FAILED")
        logs = chainlog(proc) + "\n" + "\n".join(result.get("log_tail") or [])
        self.assertIn("about to fail", logs)

    # -- 5. 任务超时 --------------------------------------------------------
    def test_05_timeout(self) -> None:
        project = helpers.copy_example_project(self.tmp / "timeout" / "proj")
        slow_entry = (
            "python train.py --epochs 50 --data data.csv --out outputs "
            "--model model.json --simulate-seconds 120"
        )
        started = time.time()
        result, proc = run_submit(project, self.tmp / "timeout" / "dl",
                                  entry=slow_entry, timeout_s=3)
        elapsed = time.time() - started
        self.assertEqual(proc.returncode, 1, helpers.describe_process(proc))
        self.assertEqual(result.get("state"), "timeout", result)
        self.assertEqual((result.get("error") or {}).get("code"), "TIMEOUT")
        self.assertLess(elapsed, 120, "超时任务应在远端被及时终止，而不是等它跑完")

    # -- 服务端 CLI 契约（顺带验收） ---------------------------------------
    def test_06_server_gpus_cli(self) -> None:
        proc = helpers.run_capture([helpers.PY, "-m", "server.main", "gpus"], timeout=60)
        self.assertEqual(proc.returncode, 0, helpers.describe_process(proc))
        data = json.loads((proc.stdout or "").strip() or "null")
        self.assertTrue(isinstance(data, (list, dict)), f"gpus 输出应为 JSON: {proc.stdout!r}")


# ---------------------------------------------------------------------------
# 原始协议测试场景（真服务端）：哈希篡改 → HASH_MISMATCH；断线续传 → resume 成功
# 同一套场景先在 test_raw_protocol_fake.py 对 fake_server 自测过。
# ---------------------------------------------------------------------------
@skip_e2e
class E2ERawProtocolTest(RawFlowScenarios, unittest.TestCase):
    """对真服务端执行原始协议验收场景。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = helpers.make_temp_dir("e2e-raw-")
        cls.addClassCleanup(helpers.cleanup_dir, cls.tmp)

    def server_address(self) -> str:
        return server().address


if __name__ == "__main__":
    unittest.main(verbosity=2)