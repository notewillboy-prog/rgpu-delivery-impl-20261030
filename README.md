# RGPU 远程算力投送

通过 Windows 或 Linux 客户端上传代码、数据和父权重，在远端执行任务，再下载日志与校验后的产物。仓库包含 TCP 客户端、服务端、GPU 调度、断点续传、环境缓存和增量训练示例。

## 远端 GPU 闭环

从仓库根目录运行一条命令，即可完成两次真实 CUDA 增量训练。

本机需要 Python 3.9+、OpenSSH，以及已配置公钥登录的 SSH 别名。远端需要 Linux、NVIDIA 驱动和一个已安装 CUDA PyTorch 的 Python 环境。本机入口使用 Python 标准库。

```bash
python tools/gpu_roundtrip.py --ssh-host gpu-host --remote-python /path/to/venv/bin/python
```

将 `gpu-host` 与解释器路径替换为自己的配置。可先检查远端环境。

```bash
ssh gpu-host '/path/to/venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"'
```

入口依次完成以下步骤。

1. 在远端 `~/.rgpu-roundtrip/<run-id>/` 部署独立服务端代码。
2. 启动回环地址上的随机端口服务和 SSH 隧道，使用本次会话生成的随机鉴权令牌。
3. 通过 RGPU 上传训练代码、CSV 数据和初始权重，由严格 GPU 调度执行第一轮训练。
4. 下载 `model.json` 与 `metrics.json`，逐文件核验大小和 SHA-256。
5. 将下载的权重作为下一轮输入重新上传，执行第二轮增量训练。
6. 验证父权重 SHA、数据 SHA、训练计数、loss 接续、环境缓存、CUDA 张量位置、显存分配及 `nvidia-smi` 中的训练 PID。
7. 停止本次创建的服务端和隧道，保存报告与产物。

两次任务均通过现有 `client.runner.submit_project` 进入正常上传、调度、执行和下载流程。远端已有的 PyTorch 环境通过任务参数复用，服务端仍为任务创建隔离的启动环境。

## 查看结果

本地每次运行生成独立目录。

```text
_rgpu_data/gpu_roundtrip/<run-id>/
  report.json
  project/
  receipts/
  round-1/outputs/model.json
  round-1/outputs/metrics.json
  round-2/outputs/model.json
  round-2/outputs/metrics.json
  tunnel.stderr.log
```

`report.json` 的 `status` 为 `passed` 时，检查项全部通过，且本次服务端与隧道已停止。失败报告包含错误、已获得的作业回执、服务端日志和清理结果。远端任务目录保留输入、日志及产物以供复核。

可通过 `--work-dir` 指定本地报告父目录，使用 `--steps` 调整每轮训练步数，或通过 `--timeout` 设置单个任务的超时秒数。每次调用均生成新的运行 ID。

## 已实测的 GPU 链路

2026-11-12，在远端 RTX 3090、Python 3.10、PyTorch 2.5.1+cu121 上完成两次实际任务，每次 128 步。

| 验证项 | 结果 |
|---|---|
| 第一轮 loss | 1.02980649 → 0.00167739 |
| 第二轮 loss | 0.00167739 → 0.00035984 |
| 权重接续 | 第二轮父权重 SHA 等于第一轮下载权重 SHA |
| 产物下载 | 两轮共 4 个 JSON 文件，大小和 SHA-256 全部一致 |
| 计算位置 | 张量和参数均为 `cuda:0`，记录 CUDA 时间及实际显存分配 |
| 外部观测 | 两轮训练 PID 均出现在相应物理 GPU 的 `nvidia-smi` 采样中 |
| 环境 | 第一轮 built，第二轮 cached |
| 清理 | 独立服务端及 SSH 隧道已停止 |

当前示例使用小型线性回归模型验证传输、GPU 执行和权重接续。LocateAnything/SAM3 业务训练适配的接线与验收材料见 [INTEGRATION.md](INTEGRATION.md) 和 [ACCEPTANCE.md](ACCEPTANCE.md)。

## 使用现有服务端

常驻服务与普通客户端入口位于 `remote_compute/`。

```bash
cd remote_compute
python -m server.main serve --host 127.0.0.1 --port 8765 --token YOUR_SESSION_TOKEN --data-dir ./_rgpu_data/server --strict-gpu
```

在另一个终端连接已配置的服务。

```bash
cd remote_compute
python -m client.main submit examples/incremental_train --entry "python train.py --epochs 3 --data data.csv --out outputs --model model.json" --server 127.0.0.1:8765 --token YOUR_SESSION_TOKEN
```

常驻服务的鉴权令牌也可以通过 `RGPU_TOKEN` 环境变量设置。客户端配置、图形入口与部署说明见 [组件说明](remote_compute/README.md)。

## 开发与测试

开发测试使用 pytest。

2026-11-12 全量回归通过 283 项，跳过 2 项，另有 110 个子测试通过。

```bash
python -m pip install pytest
python -m pytest remote_compute/tests -q
```

协议、调度、断点续传和适配器测试位于 `remote_compute/tests/`。新增 GPU 闭环回归覆盖结果校验、CPU 计算位置拒绝、父权重绑定、文件篡改、GPU PID 匹配、客户端 API 参数顺序和失败清理。

实际 GPU 验收使用上方的 `tools/gpu_roundtrip.py` 命令，并保留生成的 `report.json`。

## 目录

```text
remote_compute/common/                 协议、配置、数据模型及哈希
remote_compute/client/                 打包、上传、状态跟踪及下载
remote_compute/server/                 GPU 调度、环境构建及执行
remote_compute/examples/gpu_train/     CUDA 增量训练任务
remote_compute/examples/incremental_train/  标准库训练示例及示例数据
remote_compute/tests/                  回归测试
server/rgpu_adapter/                   业务训练适配器
vendor/rgpu_worker_patch/              业务 worker 独立副本
tools/gpu_roundtrip.py                本地一键 GPU 闭环入口
tools/remote_gpu_session.py           独立远端会话管理
```
