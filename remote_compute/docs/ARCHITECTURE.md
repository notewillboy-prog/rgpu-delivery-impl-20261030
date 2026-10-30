# 架构设计 —— RGPU 远程算力调度 Demo

> 需求原文：`增量训练前置技术_云端算力快速部署.txt`
> 协议细节：`docs/PROTOCOL.md`（v1 已冻结）
> 任务分解与协作边界：`TASK_TREE.md`

## 1. 问题与目标

本地机器跑增量训练时：显存不够、训练把机器占满、或者本地根本没有 GPU。目标是把
**环境 + 任务**整包投递到临时租用的云端 Linux 服务器，云端执行，本地只看日志、拿产物；
服务器是可插拔的——换一台机器只需改地址和 token。

非目标（demo 明确不做）：

- 不做多租户计费、用户体系、Web 控制台；
- 不做跨节点分布式训练（单机多卡由 `max_parallel_jobs` + `CUDA_VISIBLE_DEVICES` 覆盖）；
- 不做增量传输（同一文件重复上传会重传，仅做断点续传）；
- 服务端不支持 Windows（需求指定 Linux）。

## 2. 组件与职责

```
┌──────────────────────────── 本地（Windows / Linux） ───────────────────────────┐
│  client.main (CLI)                 client.gui (tkinter 一键窗口)               │
│      │                                  │                                      │
│      ├── packager ── 扫描项目目录 → FileEntry[]（相对路径 + sha256）            │
│      ├── runner ──── 提交 / 跟随日志 / 下载产物 / 退出码映射                    │
│      ├── uploader ── 1 MiB 分片、逐片校验、断线续传                             │
│      └── connection ─ 握手鉴权、帧收发、心跳、重连                              │
└───────────────────────────────────┬────────────────────────────────────────────┘
                                    │ TCP：长度前缀帧（JSON 控制 + BLOB 分片）
┌───────────────────────────────────┴────────────────────────────────────────────┐
│  server.app      会话线程：鉴权 → 消息分发 → 上传/执行/下载                     │
│  server.jobstore jobs/<job_id>/{spec.json, inputs/, outputs/, logs/, state.json}│
│  server.envbuilder 指纹 → venv 复用/重建 或 docker 镜像                         │
│  server.scheduler nvidia-smi 探测、FIFO 队列、GPU 分配、并发度                  │
│  server.executor subprocess(argv) → 流式日志 → 超时 → 产物清单                  │
└───────────────────────────────────┬────────────────────────────────────────────┘
                                    │
                          作业进程（用户 train.py，CPU/GPU）
```

分层原则：**环境与任务分离**。

- 环境 = `EnvSpec`（mode/python/requirements/env_vars）→ 环境指纹 → 服务端缓存目录。
  同指纹直接复用，避免每次提交都装依赖。
- 任务 = `JobSpec`（entry_argv + inputs + outputs + resources + meta）→ 每次独立作业目录。
- 传输 = `common.protocol`：同一套分片循环用于上传和下载，天然支持续传与校验。

## 3. 关键流程

### 3.1 一键投递序列

```
client                          server
  │ hello(token) ───────────────▶│ 校验 token，回 hello_ok(gpus, 版本)
  │◀─────────────────────────────│
  │ job_submit(spec) ───────────▶│ 建 job 目录，校验指纹，回 missing[]
  │◀─────────────────────────────│
  │ file_chunk + blob ×N ───────▶│ 逐片校验 sha256，回 file_ok(offset)
  │ file_done ──────────────────▶│ 整文件校验
  │ job_start ──────────────────▶│ 入队 → 环境就绪/构建 → subprocess
  │◀── job_started(env.state) ───│
  │◀── job_event(stream) ×N ─────│ stdout/stderr 逐行推送
  │◀── job_finished(outputs) ────│ 退出码 + 产物 sha256 清单
  │ output_list / output_chunk ─▶│ 按 offset 回产物分片
  │◀── blob ×N ──────────────────│
```

### 3.2 环境指纹与复用

```
fingerprint = sha256("rgpu-env-v1\nmode=venv\npython=3.12\nextra=\nrequirements:\n" + sorted(归一化依赖))
```

- 服务端 `<data_dir>/envs/<fingerprint>/` 下建 venv（或 docker 镜像标签），完成后写 `.ready`。
- `.ready` 存在 → `env.state=cached`；否则构建并回传 `job_progress`。
- 构建失败 → `job_finished(state=failed, exit_code=-1, log_tail=[pip 错误])`，
  本地一眼看到是依赖问题，不用翻服务器。
- requirements 为空时用 `python -m venv --without-pip`：零依赖任务也能隔离环境，
  且不需要联网（这也是本机 Windows 无 pip 环境下 e2e 能跑通的原因）。

### 3.3 断点续传

- 分片级：服务端 `file_ok` 返回下一个期望 `offset`；网络抖动后从该 offset 续传。
- 文件级：整文件 sha256 不符 → 客户端从 0 重传（demo 取舍，避免复杂的部分文件合并）。
- 会话级：重连后 `job_submit` 带 `resume_job_id`，服务端校验 spec 指纹一致后复用 job，
  返回 `present`（已完整上传的文件）与 `missing`（还需上传的文件）。
- 客户端任务元数据落 `~/.rgpu/last_job.json`，GUI 崩了也能续。

### 3.4 GPU 调度

- `nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader`
  解析为 `GpuInfo[]`；解析失败/无 GPU 时降级为空列表并继续以 CPU 模式服务（Windows 自测路径）。
- 队列 FIFO；`max_parallel_jobs` 控制同时运行作业数（默认 1）。
- 每个作业按 `resources.gpu_count` 分配 GPU，注入 `CUDA_VISIBLE_DEVICES`；
  训练脚本里不写死 `cuda:0`，只认逻辑设备 0。
- 无 GPU 但 `resources.gpu=true` 的作业：仍执行，`job_started` 中说明以 CPU 运行（demo 不阻塞）。

### 3.5 失败语义

| 场景 | 服务端行为 | 客户端表现 |
|------|-----------|-----------|
| token 错 | `error BAD_AUTH` 后断开 | 退出码 4，提示检查 token |
| 上传哈希不符 | `error HASH_MISMATCH`，等待重传 | 自动重传有限次，仍失败退出码 1 |
| 环境构建失败 | `job_finished(state=failed, exit_code=-1, log_tail)` | 打印 pip 错误，退出码 1 |
| 任务非零退出 | `job_finished(state=failed, exit_code=N)` | 打印日志，退出码 1 |
| 超时 | kill 进程树，`job_finished(state=timeout)` | 退出码 1，提示加 `--timeout` |
| 断线 | 保留 job 目录，会话断开 | 重连续传，超次数退出码 3 |

## 4. 安全边界（demo 级）

- token 鉴权（`--token` 或 `RGPU_TOKEN`）；服务端无 token 拒绝启动。
- 上传路径统一 `POSIX 相对路径` 校验，拒绝绝对路径与 `..`（`FileEntry.__post_init__`）。
- 入口命令 argv 列表直接 `subprocess.Popen`，**不经过 shell**，避免注入。
- 作业相互隔离在各自目录；作业间不能互相覆盖。
- 传输为明文 TCP：生产使用务必套 SSH 隧道（`ssh -L 8765:127.0.0.1:8765 user@server`），
  或只监听内网/安全组白名单。`deploy/README.md` 有说明。
- 尚未实现（明确标注）：传输加密、用户隔离（chroot/容器）、磁盘配额。

## 5. 增量训练适配点

`examples/incremental_train` 是接入样例：本地项目目录里放

```
项目目录/
  train.py           # 入口，读 model.json（旧权重）继续训练
  model.json         # 当前模型/权重（增量训练的起点）
  data.csv           # 本轮新增数据
  requirements.txt   # 环境依赖
```

客户端把整个目录作为 `inputs` 上传，`meta.mode="incremental_training"` 标记语义。
真实项目接入只需保证：入口脚本能用相对路径读到权重与数据，产物写进 `outputs/`，
设备号用 `CUDA_VISIBLE_DEVICES` 注入而不是写死。

## 6. 扩展路线（demo 之后）

1. 环境镜像化：wheelhouse 离线包、conda、docker 镜像缓存，解决云端装包慢/无外网。
2. 增量传输：以文件 sha256 做内容寻址缓存，跨 job 复用未变化的权重/数据。
3. 断点训练：客户端把 checkpoint 路径注册给服务端，重试时从 checkpoint 继续。
4. 多服务器：客户端维护服务器池，按 GPU 空闲度选节点；服务端注册中心。
5. 安全：TLS/mTLS、每用户容器隔离、资源配额、审计日志。
6. 与标注软件集成：X-AnyLabeling 的「一键增量训练」按钮直接调 `client.gui` 背后的 API。

## 7. 平台与测试现实

- 开发机是 Windows 10/11 + Python 3.12.14（项目 venv 无 pip、无外网），有 `nvidia-smi` 与 Docker CLI。
- 服务端在 Windows 上以 CPU 模式跑通全链路（venv `--without-pip` + 纯标准库示例任务）。
- Linux 部署脚本与 GPU 实际调度在本机**无法实测**，文档中逐条标注「未验证」；
  `nvidia-smi` 解析逻辑用本机真实输出验证。
