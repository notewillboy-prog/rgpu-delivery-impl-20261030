# 接线说明（不修改并行文件）

本文件给出“把实现副本合入现有项目”的最小补丁点。`rgpu_bundle.py`、`prepare.py`、`remote_compute` 原副本仍由并行任务所有；以下只描述改法，不在本轮改动它们。

## 1. 服务端：缓存 + 幂等

### 1.1 启动参数

```bash
python -m server.main serve \
  --host 127.0.0.1 --port 18813 \
  --token "$(cat /home/rgpu/rgpu-training-20261030/token)" \
  --data-dir /mnt/b2730ee6-a71b-4fb0-8470-ae63626b6f38/rgpu-training-20261030/data \
  --max-parallel 1 \
  --weight-cache-bytes 322122547200
```

`--weight-cache-bytes` 为 0 时行为与旧版完全一致；启用后缓存位于 `<data-dir>/weights-cache`，索引位于 `<data-dir>/idempotency`。默认启动会把上次遗留的非终态作业标 `FAILED`（不自动重跑）；要保留“归属不明”语义用 `--strict-recovery`。

### 1.2 客户端携带 idempotency key

```python
spec.meta["idempotency_key"] = "batch-20261030-locate-001"
```

服务端 `_on_job_submit` 的行为：

- 同 key、同 canonical spec fingerprint：返回原 `job_id`，`reused=true`，`state` 为当前状态（含终态）。
- 同 key、异 fingerprint：`error CONFLICT`，不创建新作业。
- 无 key：完全走旧路径。

### 1.3 缓存命中返回

`job_accepted` 新增可选字段：

```json
{
  "received": {"data/images/a.jpg": 12345},
  "cache_hits": ["approved_base.zip"],
  "reused": false,
  "state": "uploading"
}
```

老客户端忽略这些字段即可；新客户端用 `received` 从偏移续传，用 `cache_hits` 打日志/验收。

## 2. 客户端：上传/下载/账本

### 2.1 上传

`upload_file(..., start_offsets={path: offset})` 已实现：

- 第一次尝试从 `offset` 开始，且要求该偏移前内容已由服务端确认；
- 整文件 `file_done` 失败重试时强制从 0 开始，避免坏前缀被沿用；
- 服务端仍按分片 SHA 与整文件 SHA 双重复核。

`JobRunner._upload_with_resume` 已读取 `accepted["received"]` 并传入；无需额外接线。

### 2.2 下载

`_download_one` 已改为：

- `.part` 存在且大小 ≤ 目标大小时，从该大小继续；
- 先重算 `.part` 前缀 SHA，最终整文件再核对；
- `ConnectionLost` 保留 `.part`；协议/本地错误删除半成品；
- 最终 SHA 不符时自动从 0 重下一次，再失败才报错。

### 2.3 客户端账本（opt-in）

```bash
python -m client.main submit <project> \
  --server 127.0.0.1:18813 --token "$TOKEN" \
  --receipt-dir /home/rgpu-client/.rgpu-receipts \
  --request-id locate-20261030-001
```

语义：

- 第一次 `begin_once` 允许提交；之后同一 receipt 目录/request_id 一律拒绝，退出码 4；
- 服务端接受后写 `job_id`，再进入 `tracking`；
- 终态写 `succeeded/failed/cancelled`；进程崩溃时账本最多停在 `intent/accepted/tracking`，恢复动作是 `query_existing`，**不自动重投**。

## 3. Bundle：把 `run.py` 放进去

`src/label_workbench/rgpu_bundle.py` 当前复制清单包含 `prepare.py`，entry 是 `prepare.py`。需要加：

```python
# build_bundle() 内，准备复制 service 文件时
for name in ("prepare.py", "run.py", "weights_validate.py"):
    shutil.copyfile(ADAPTER_DIR / name, bundle / name)
```

并把 `bundle_manifest.json` / receipt / entry 改成：

```python
entry = ["python", "-I", "-B", "-X", "utf8", "run.py",
         "--manifest-sha256", manifest_sha]
```

`verify_trusted_entrypoint()` 也要把 `run.py`、`weights_validate.py` 纳入允许/必需的信任锚；否则执行端会拒绝。

## 4. `run.py` 依赖的 bundle 布局

```
bundle/
├── bundle_manifest.json
├── request.json
├── prepare.py                 # 已存在
├── run.py                     # 新增
├── weights_validate.py        # 新增
├── requirements.txt
├── server/
│   ├── training_service/      # 或 sam3_training_service
│   ├── workbench_server/gpu_lock.py
│   └── vendor_sam3/
└── inputs/...
```

`run.py` 通过 `server/` 在 `sys.path` 中导入 family 包；`worker.run_job` 需要 `workbench_server.gpu_lock.leased`，因此 `workbench_server/` 必须一起复制（当前 bundle 已复制 `gpu_lock.py`）。

## 5. 真实 GPU 执行前必须补的

1. `rgpu_bundle.py` 的 entry/trusted 校验（§3）。
2. `run.py` 的 `RealDeps` 在真实 bundle 上跑一次 L4（不训练，只 import/加载父权重/释放）。
3. gpu-server 上 `INCREMENTAL_GPU` 只允许 `0`/`1`；`run.py` 会把 `CUDA_VISIBLE_DEVICES=0,1` 收敛到第一个物理号，但服务器进程仍应显式设单卡。
4. 8765 的协调：本实现不能硬隔离 8765；真实训练前需要操作员确认 8765 无作业、GPU 空闲，并保留 `nvidia-smi` 前后证据。

## 6. 不建议做的事

- 不要把 `weight_cache_prototype.py` 改名/搬走后再 import：当前接线直接依赖该模块；合并时保持文件名或同步修改 `app.py` 的 import。
- 不要让 `WeightCache.materialize` 覆盖已存在文件；它故意 `FileExistsError`，调用方应先清理 0 字节占位。
- 不要信任 `file_checksums.json` 自身的哈希（它按设计不包含自己）；校验器会检查覆盖集合与每个成员的实际 SHA。
- 不要把 `quality_status` 改成 `approved`：当前 worker 只写 `needs_review`，没有业务阈值。
