# RGPU 远程投送 · 可执行实现（2026-10-30）

本目录是 `D:\auto_label_workbench\remote_compute` 的**独立实现副本**，用于把 10-28/10-29 审查中的 P0 缺口落成可运行代码。原项目、`remote_compute` 并行工作副本、`prepare.py`、`rgpu_bundle.py` 和两台服务器均未修改。

## 1. 本轮完成

| 缺口 | 实现 | 位置 |
|---|---|---|
| 跨作业权重缓存 | `WeightCache` 接入服务端：提交时命中直接落到 `inputs/`，`file_done` 后写入缓存 | `remote_compute/server/app.py:647`、`:808`、`:827` |
| 缓存开关 | `--weight-cache-bytes`，默认 0（禁用） | `common/config.py:121`、`server/main.py` |
| 服务端幂等 | 文件索引 + 同 key 同 spec 复用、异 spec `CONFLICT` | `server/idempotency.py`、`app.py:335` |
| 幂等复用/终态查询 | `attach_runtime` 允许附着终态作业，客户端可查结果 | `server/app.py:808` |
| 上传断点 | `job_accepted.received` 返回部分文件偏移；上传器从该偏移继续 | `server/app.py:360`、`client/uploader.py` |
| 下载断点 | `.part` 保留，断线后从大小继续；前缀哈希错误自动从 0 重下 | `client/runner.py:_download_one` |
| 客户端投递账本 | `--receipt-dir/--request-id` opt-in，`begin_once` 后禁止自动重投 | `client/runner.py:187`、`client/main.py` |
| 真实执行入口 | `run.py`：调用 prepare → worker → 校验 weights.zip → 写 receipt | `server/rgpu_adapter/run.py` |
| 产物校验 | `weights_validate.py`：ZIP 路径/CRC/checksum/权重指纹/父指纹/质量口径 | `server/rgpu_adapter/weights_validate.py` |
| 状态可观测 | `updated_at/log_tail/error/gpus` 出现在 `job_status` | `server/runtime.py`、`server/jobstore.py` |

## 2. 测试

```powershell
cd D:\auto_label_workbench\work\rgpu_delivery_impl_20261030\remote_compute
$env:PYTHONUTF8='1'; $env:PYTHONDONTWRITEBYTECODE='1'
& D:\auto_label_workbench\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

结果：**224 passed, 1 skipped, 110 subtests passed**（2026-10-30 实测，Windows CPU）。

新增覆盖：

- `tests/test_idempotency.py`：索引并发/冲突/损坏/路径安全 + 真服务端重复提交复用。
- `tests/test_cache_integration.py`：缓存命中直接 materialize、状态记录、禁用透明。
- `tests/test_cache_e2e.py`：第二个作业 0 上传命中缓存。
- `tests/test_resume_offsets.py`：上传从 `received` 开始、整文件重试回 0、越界截断。
- `tests/test_server_received_offsets.py`：服务端真实回传部分偏移。
- `tests/test_download_resume.py`：`.part` 续传、断线保留、损坏前缀重下。
- `tests/test_receipt_wiring.py`：第一次提交成功、第二次无网络动作即拒绝。
- `tests/test_run_adapter.py`：`run.py` 成功/失败/二次执行/校验失败四条路径。
- `tests/test_status_payload.py`：状态诊断字段。

## 3. 仍未完成（真实 GPU 前必须知道）

1. **`run.py` 还没打进 bundle**：`src/label_workbench/rgpu_bundle.py` 当前只复制 `prepare.py`，且 entry 是 `prepare.py`。需按 `INTEGRATION.md` 的补丁点把 `run.py`、`weights_validate.py` 加入复制列表，并把 entry 改成 `run.py`。未改该文件，因为并行任务正在编辑它。
2. **`run.py` 的真实依赖未在 GPU 上跑过**：默认 `RealDeps` 依赖 bundle 内 `server/training_service`（或 `sam3_training_service`）、`workbench_server/gpu_lock.py`、torch/transformers；本轮只用 FakeDeps 测了编排。真实执行会写 `outputs/run_receipt.json` 或 `run_failure.json`。
3. **缓存无淘汰**：配额满会拒绝写入（非致命），没有 LRU/引用计数；gpu-server 需要人工/运维清理，或后续加 `prune()`。
4. **缓存每次命中全量重算 SHA**：安全但 12–18 GB 权重会慢；可接受“安全优先”，不要为了快而信任 sidecar。
5. **幂等索引 crash window**：`create_job` 成功、`record` 前进程崩溃会留下孤儿 job；客户端 receipt 仍是最后一道防线。
6. **上传续传的边界**：服务端只对“部分文件”返回 `received`；若服务端已把坏文件清空，客户端会从 0 重传。整文件 SHA 仍是最终裁判。
7. **下载续传的边界**：只处理 `.part` 与断线；服务端产物在中途被改写时，最终 SHA 会失败并触发一次从 0 重下，仍失败则任务失败。
8. **GPU 隔离不是本轮目标**：`strict_gpu` 仍是点快照 + prelaunch，不是跨进程租约；gpu-server 8765 不受本实现控制。真实训练前必须做维护窗口/操作员协调。
9. **真实训练未执行**：按用户约束，本轮没有启动任何训练或服务；所有验收均为 CPU/Windows。

## 4. 目录

```
rgpu_delivery_impl_20261030/
├── remote_compute/                 # 独立副本（含本轮修改）
│   ├── server/{app,jobstore,runtime,idempotency,weight_cache_prototype}.py
│   ├── client/{runner,uploader,main,durable_receipt}.py
│   └── tests/test_{idempotency,cache_integration,cache_e2e,resume_offsets,...}.py
├── server/rgpu_adapter/run.py      # 待合并进 bundle 的执行入口
├── server/rgpu_adapter/weights_validate.py
├── ACCEPTANCE.md                   # CPU / 真实 GPU 验收命令
├── INTEGRATION.md                  # 与现有项目的接线点与补丁
└── 开发流程.md                     # 做了什么/方法/坑/注意/交接
```

## 5. 结论

- 缓存、幂等、上传/下载续传、客户端账本、`run.py`、产物校验从“建议/原型”变成了**有测试的实现**；RGPU 全量测试 224 通过。
- 但“真实 GPU 双模型训练”仍被三件事阻塞：bundle 还没复制 `run.py`、真实 `RealDeps` 未在 GPU 验证、gpu-server GPU/8765 协调未获批准。
- 下一步见 `ACCEPTANCE.md` 的 L4–L6；在用户明确解除“不启动训练”限制前，不应执行真实训练。
