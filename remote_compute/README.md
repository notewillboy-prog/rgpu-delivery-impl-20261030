# RGPU · 通用远程算力调度 Demo（增量训练场景）

> 本地写代码、用本地环境小数据跑通，然后**一键把算力任务抛到云端**：
> 代码 + 环境依赖 + 模型权重 + 增量数据整包上传，云端还原环境、排队执行、回传日志与产物。
> 本地机器继续干原来的活，服务器是可插拔的——换一台只需改地址和 token。

- 客户端：Windows / Linux（Python 标准库 + tkinter 一键窗口）
- 服务端：Linux（Python 标准库；CPU / NVIDIA GPU 均可）
- 传输：单端口 TCP，自制长度前缀帧（JSON 控制 + 二进制分片），token 鉴权，断点续传

完整远端 GPU 闭环与最新实测结果见 [仓库首页](../README.md)。

## 它解决什么问题

| 痛点 | RGPU 的做法 |
|------|-------------|
| 本地没有 GPU / 显存不够 | 任务抛到临时租用的云端 Linux 服务器执行 |
| 训练把本机占满、卡住其他服务 | 本地只负责打包上传和收结果，不参与计算 |
| 云端装环境麻烦、每次重装 | requirements 指纹化，同指纹环境只建一次，后续直接复用 |
| 大模型/大权重上传中断要重传 | 分片传输 + 服务端逐片确认，断线后从断点续传 |
| 换服务器就要改代码 | 服务器无状态可插拔，客户端只配置 `host:port + token` |
| 训练脚本写死 GPU 号 | 调度器统一注入 `CUDA_VISIBLE_DEVICES`，脚本里只认逻辑设备 0 |

## 5 分钟本机跑通（Windows，CPU 模拟云端）

前提为 Python 3.9+。该标准库示例使用本机 Python，服务端在当前目录创建独立数据目录。真实远端 GPU 验收入口见 [仓库首页](../README.md)。

```powershell
# 0) 进入项目根目录
cd remote_compute

# 1) 起服务端（本机模拟一台云端 Linux 服务器；CPU 模式）
python -m server.main serve --host 127.0.0.1 --port 8765 --token demo-token --data-dir .tmp_server
# 看到 RGPU_SERVER_READY host=127.0.0.1 port=8765 ... 即就绪

# 2) 另开一个终端，一键提交增量训练任务
python -m client.main submit examples\incremental_train `
    --entry "python train.py --epochs 3 --data data.csv --out outputs --model model.json" `
    --server 127.0.0.1:8765 --token demo-token --timeout 300

# 3) 看结果：日志流式打印，产物下载到 examples\incremental_train\outputs\
#    再提交一次，job_started 会显示 env.state=cached（环境被复用）
```

图形界面（一键增量训练）：

```powershell
python -m client.main gui
```

填服务器地址 / token / 项目目录 / 入口命令，点「开始增量训练」即可，日志和进度实时显示。

## 云端部署（Linux）

```bash
# 云服务器（Ubuntu 22.04+，有 NVIDIA 驱动即可）
git clone <本仓库> rgpu && cd rgpu
sudo bash deploy/install_server.sh --port 8765 --token "$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')"
```

详见 [deploy/README.md](deploy/README.md)：云主机选型、安全组、SSH 隧道、GPU 驱动/CUDA 兼容性、
systemd 服务管理、卸载与排错。（Linux 实测项与「未验证」项在该文档里逐条标注。）

## 工作原理

```
本地 client                                     云端 server (Linux)
  packager  项目目录 → 文件清单+sha256
  runner ── hello(token) ───────────────────────▶ 鉴权
          ── job_submit(spec) ──────────────────▶ 建作业目录 / 校验环境指纹
          ── file_chunk + blob ×N ──────────────▶ 逐片 sha256 校验（可续传）
          ── job_start ─────────────────────────▶ 排队 → 复用/构建环境 → subprocess
          ◀──────────────── job_started/event ─── 流式日志
          ◀──────────────── job_finished ─────── 退出码 + 产物清单
          ── output_chunk(offset) ─────────────▶ 分片回传产物
```

**环境与任务分离**：环境 = requirements + Python 版本 → 指纹；任务 = 代码 + 权重 + 数据 + 入口命令。
同一份依赖无论写多少遍注释、顺序如何，指纹不变，云端只建一次环境。

技术细节见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，协议见 [docs/PROTOCOL.md](docs/PROTOCOL.md)。

## 目录结构

```
common/      协议、数据模型、哈希/环境指纹、配置（客户端/服务端共用，契约冻结）
server/      服务端：会话、作业存储、环境构建、GPU 调度、执行器
client/      客户端：打包、上传、执行跟踪、CLI、tkinter GUI
examples/    示例增量训练任务（纯标准库，可直接投递）
tests/       单元测试 + 本地端到端测试（模拟云端）
deploy/      Linux 一键部署脚本、systemd、Dockerfile
docs/        协议、架构、使用文档
```

## 测试

```powershell
python -m unittest discover -s tests -v
# 或（PowerShell 默认执行策略会拦 .ps1，必须带 -ExecutionPolicy Bypass）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_checks.ps1
```

- `run_checks.ps1` 默认**包含端到端测试**（自动起 server 18766 + 真跑 client 提交示例任务）；
  只跑离线单元测试用 `-SkipE2E`，或设 `RGPU_SKIP_E2E=1`；e2e 端口可用 `RGPU_E2E_PORT` 覆盖。
- 仓库级完整测试从根目录执行 `python -m pytest remote_compute/tests -q`。2026-11-12 实测 283 项通过、2 项跳过、110 个子测试通过，包含真实本地服务端的端到端覆盖。
- 覆盖：帧协议、哈希与环境指纹、路径安全、打包清单、上传/下载、作业成功/失败/超时、
  错误 token、哈希不匹配、断线续传、二次提交环境复用。
- 测试临时目录放在 `<项目根>/_rgpu_data/tests/`（DSH 沙箱下系统 temp 目录写文件会被拒）。

## 安全说明（重要）

- 默认明文 TCP。**公网部署请只监听内网 + 安全组白名单**，或本地 `ssh -L 8765:127.0.0.1:8765 user@server`
  走 SSH 隧道（客户端连 `127.0.0.1:8765`）。
- 服务端必须设置 token（`--token` 或环境变量 `RGPU_TOKEN`），无 token 拒绝启动。
- 上传路径做了相对路径与 `..` 校验；入口命令不经 shell 执行。
- demo 未实现：TLS、多用户隔离、磁盘配额、计费审计。生产使用前请补齐。

## 已知限制

- 独立副本可使用 `--strict-recovery`：服务重启后，如果未完成作业不在本进程内，查询、恢复和启动都会返回 `CONFLICT`，提示核查原进程及作业，不改旧状态、不自动重跑。原进程内重连和已结束任务查询保留。该选项默认关闭；它无法判断旧进程是否仍在运行，也不提供跨进程独占、远端取消或优化器断点续训。

- 服务端只支持 Linux（需求约定）；Windows 仅作为客户端与本地测试环境。
- 没有跨作业的文件级去重：同一份大权重每次提交都会重传（断点续传只在单次任务内生效）。
- GPU 调度是单机 FIFO，不做跨节点分布式训练。
- 依赖安装走公网 PyPI；离线集群需要自备 wheelhouse（见 `deploy/README.md` 扩展说明）。

## License

MIT（demo 用途，可自由修改）。

## 严格 GPU 选项（独立项目副本）

服务端可设置 `--strict-gpu --min-gpu-free-mb 16384`：GPU 作业只分配当前无计算进程、空闲显存不低于门限的 GPU；资源不足或检查失败则作业失败，不退回 CPU。环境创建后、启动训练进程之前会再检查一次。默认不启用，保留原 demo 行为。该检查只反映查询瞬间，进程内分配不构成跨服务或跨用户 GPU 独占租约；不能防止外部进程随后抢占，实际训练仍需另行实现和验证跨进程资源协调。
