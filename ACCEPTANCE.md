# 验收清单（CPU 演示 vs 真实 GPU）

> 2026-10-30 更新：用户已批准真实 GPU 训练；L0–L6 均已在 gpu-server 独立目录/端口执行并留证。详细结果见 `work/rgpu_live_20261030/LIVE_ACCEPTANCE_20261030.md`。

## L0 库/单元（已完成）

```powershell
cd D:\auto_label_workbench\work\rgpu_delivery_impl_20261030\remote_compute
$env:PYTHONUTF8='1'; $env:PYTHONDONTWRITEBYTECODE='1'
& D:\auto_label_workbench\.venv\Scripts\python.exe -m pytest tests/test_idempotency.py tests/test_cache_integration.py tests/test_resume_offsets.py tests/test_download_resume.py tests/test_receipt_wiring.py tests/test_status_payload.py -q
```

期望：全绿；覆盖同 key 复用、异 spec 冲突、缓存命中/损坏、上传偏移、`.part` 续传、账本禁止重投。

## L1 全量离线（已完成）

```powershell
& D:\auto_label_workbench\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

期望：**225 passed, 1 skipped, 110 subtests passed**。跳过项是既有的可选 E2E 开关，不是失败。

## L2 本地真服务端 + 缓存/幂等 E2E（已完成）

```powershell
& D:\auto_label_workbench\.venv\Scripts\python.exe -m pytest tests/test_idempotency.py tests/test_cache_e2e.py tests/test_server_received_offsets.py -q
```

期望：第二个作业 `cache_hits` 覆盖全部输入、`missing=[]`；重复提交同 key 返回同 `job_id`；部分上传后 `received[path] > 0`。

## L3 gpu-server CPU 预检（未执行）

只读/无训练：

```bash
ssh rgpu@gpu-server 'python3 -V; nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv; \
  ls -ld /mnt/b2730ee6-a71b-4fb0-8470-ae63626b6f38; \
  systemctl is-active rgpu-server 2>/dev/null || true'
```

验收点：

- 数据盘可写、剩余 ≥ 300 GiB；
- `/opt/rgpu-server` 与 8765 未被本实现改动；
- 两个 family 的解释器可按绝对路径找到。

## L4 真实 bundle 的“不训练”预检（未执行）

在 gpu-server 的独立目录（不要碰 8765）：

```bash
cd /home/rgpu/rgpu-training-20261030/bundle
export CUDA_VISIBLE_DEVICES=0
export INCREMENTAL_GPU=0
export WORKBENCH_SHARED_GPU=0
python -I -B -X utf8 run.py --manifest-sha256 "$MANIFEST_SHA" --help
# 只验证 import/参数；真正执行必须另经批准
```

更细的预检可以写一个 `--dry-run`（尚未实现）只做 `prepare`、`torch.cuda.is_available()`、父权重加载、显存释放，不启动 worker。建议在 L5 前补这个开关。

## L5 单模型 1 step（未执行，需批准）

前置：L4 通过、gpu-server 两块 3090 空闲、8765 无作业（`nvidia-smi --query-compute-apps` 空）、token 文件 0600、`max-parallel 1`。

```bash
# 服务端（示例；不要使用 8765）
python -m server.main serve --host 127.0.0.1 --port 18813 \
  --token "$(cat token)" --data-dir /mnt/.../rgpu-training-20261030/data \
  --max-parallel 1 --weight-cache-bytes 322122547200

# 客户端（另一台机器，SSH 隧道）
python -m client.main submit <locate-project> --server 127.0.0.1:18813 \
  --token "$TOKEN" --gpu --timeout 3600 \
  --receipt-dir ~/.rgpu-receipts --request-id locate-1step
```

验收点：

- `prepare.py` 通过父指纹/数据指纹/磁盘检查；
- `run.py` 写 `outputs/run_receipt.json`，`validation.ok=true`；
- `weights.zip` 的 `result_manifest.parent_model_version_id` 等于 prepare 的 `remote_parent_model_id`；
- `quality_status=needs_review`，不宣称质量；
- 失败时 `outputs/run_failure.json` + `logs/` 可定位，且不发布半成品权重。

## L6 双模型 + 恢复 + 缓存（未执行，需批准）

1. 先 Locate 1 step，再 SAM3 1 step；两个 family 使用各自的 venv（3.11/cu118、3.12/cu126）。
2. 第二次提交同数据：`cache_hits` 覆盖输入，上传字节为 0。
3. 上传中途 kill 客户端 → 重连带 `resume_job_id` → `received` 从断点继续 → 最终 `weights.zip` SHA 一致。
4. 服务端重启 → 默认启动恢复把非终态作业标 `FAILED`（`log_tail/error` 有原因、不自动重跑）；`--strict-recovery` 下未完成作业只能查、不能自动重跑；终态作业仍可 `job_status`。
5. 全程保留 `nvidia-smi` 前后快照和 8765 `systemctl show` 时间戳，证明未占用/未重启。

## 不能作为验收的东西

- 只看“进程退出 0”不等于训练成功；必须校验 `weights.zip`。
- 1 step/小数据只证明链路，不证明收敛/质量。
- CPU 模式跑出的权重不是 GPU 训练结果，不能用于业务验收。
- 缓存命中不等于输入正确；仍以整文件 SHA + `parent_fingerprint` 为准。

## 2026-10-30 Codex 复核后的更正（离线，未部署/未训练）

- **strict_recovery 旁路**：`attach_runtime` 现在也受 strict_recovery 约束；进程内未知的非终态磁盘作业拒绝 attach，只有本进程刚通过幂等路径创建的作业可用 `created_here=True` 注册。回归：`tests/test_strict_recovery_bypass.py`、`tests/test_strict_recovery.py`（wire 同 key 重启后 CONFLICT）。
- **幂等崩溃窗**：`IdempotencyIndex` 先持久化 `reserved` job_id，再以该 id 创建，再写 `committed`；重试只修复同一 id，不再 create 第二个 job。`JobStore.create_job(spec, job_id=...)` 支持显式 id 且 spec 冲突拒绝。回归：`tests/test_idempotency.py`（磁盘 job 目录始终 1）。
- **新类别误拒**：`run.py RealDeps.classes()` 合并父类与 `training.class_prompts`；`weights_validate.py` 比较完整 class->prompt 映射。回归覆盖两家族 base 空类/新增类/同类，以及 replay 旧类合并、重复类名、new_only 携带 replay 提示词拒绝。
- **父指纹/父 ID**：`run.py` 构造 local->remote 映射并写入 receipt；`weights_validate.py` 在指定 expected 时拒绝缺失/空/非法指纹，并校验映射。历史产物不补字段。
- **bundle 绑定**：`audit/verify_live.py` 对 manifest SHA 漂移/缺失始终 FAIL；`build_live.py` 只在 build 阶段签一次 manifest，拒绝消费后重算。旧 locate 证据保留，不重训/不覆盖。
- 全量测试：**241 passed, 1 skipped, 110 subtests passed**；审计自测 13/13 PASS。
- **仍未闭环**：真实大模型仍是 gpu-server 本地直跑 `run.py`；尚未完成 gpu-client 客户端经 RGPU 投递双模型的完整闭环；下一阶段测试数据用用户指定的 **car 或 Pothole**，不再用 fish。

> 更正上方“L5 审计 PASS”的旧口径：Locate 严格 bundle-binding 审计为 FAIL；SAM3 为 PASS。

## 2026-10-30 离线代码收尾第二轮

- `run.py` 的类别绑定支持 replay：合并父类 + `replay_class_prompts` + 新类；重复类名、new_only 携带回放提示词、提示词冲突均拒绝。
- strict recovery 增加 reserved 修复回归：修复已有预留保留 `created_here=False`，不得被当作本进程新作业放行。
- audit selftest 增加 bundle_manifest 缺失 FAIL 用例，14/14；build helper 5 项通过。
- 全量 **247 passed, 1 skipped, 110 subtests passed**；定向 **38 passed**。
- 仍未部署/未训练；历史 locate bundle-binding FAIL 不变。
