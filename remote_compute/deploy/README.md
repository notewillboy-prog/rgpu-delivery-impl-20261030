# RGPU 服务端部署指南（Linux / Ubuntu 22.04・24.04）

> 本目录只有部署相关文件；服务端代码在 `server/`，协议契约在 `docs/PROTOCOL.md`。
> 需求原文：`增量训练前置技术_云端算力快速部署.txt`（「临时借用云端服务器、装完服务端就走」）。
>
> **实测边界**：开发机是 Windows，没有 Linux 环境；本机 docker daemon 也未运行。
> 因此本目录的脚本/镜像只做过**静态审阅与语法检查**，真实 Linux 上的一次成功部署
> **尚未验证**。文末「未验证清单」逐条列出，请先在一台按量计费的临时机器上跑通。

## 1. 文件说明

| 文件 | 用途 |
|------|------|
| `install_server.sh` | 一键安装：apt 依赖 → `rgpu` 系统用户 → 拷贝代码 → venv → 生成 token → 装 systemd → 等端口 → 打印 token 与安全提示 |
| `rgpu-server.service` | systemd 单元模板，`install_server.sh` 会把路径/端口/并行数替换成实际值 |
| `Dockerfile` | 可选：把**服务端本体**跑在容器里（nvidia/cuda runtime + python3 + 代码） |
| `README.md` | 本文 |

## 2. 云服务器选购建议

| 项 | 建议 | 说明 |
|----|------|------|
| 镜像 | **Ubuntu 22.04 LTS / 24.04 LTS**（纯净版，不要带预装面板） | 本仓库脚本按 apt + systemd 设计；Debian 12 大致可用（未验证） |
| 规格 | 视任务：CPU 玩具任务 2C4G 起；真实增量训练按显存/带宽选 GPU 机型 | 服务端本身很轻（标准库 + 线程），瓶颈是作业 |
| GPU | 至少 1 张 NVIDIA 卡；如需并行，`--max-parallel` ≤ 显卡数更合理 | 服务端用 `nvidia-smi` 探测，把 `CUDA_VISIBLE_DEVICES` 分配给作业 |
| 磁盘 | 系统盘 ≥ 50 GB；数据盘挂到 `/var/lib/rgpu`（作业输入/产物/环境缓存） | 作业目录会累积：`jobs/`、`envs/`；demo 无自动清理（见第 8 节） |
| 网络 | 公网 IP + 安全组只开 **22**；**不要**把 8765 对 `0.0.0.0/0` 开放 | 明文 TCP + 共享 token，优先 SSH 隧道（第 5 节） |
| 计费 | 按量/竞价实例，用完即销毁 | 项目定位就是「可插拔、不绑定的临时算力」 |
| NVIDIA 驱动 | 云镜像通常自带驱动；自装请用官方 `.run` 或发行版仓库，**驱动版本要 ≥ 下文 CUDA 要求** | `nvidia-smi` 能列卡即可 |

> 买机器时先确认：`nvidia-smi` 能正常输出、`python3 --version` ≥ 3.9。

## 3. 一键部署（推荐）

### 3.1 上传代码

在**本地项目根目录**执行（把 `<SERVER_IP>`、`<USER>` 换成实际值）：

```bash
# 只传必要目录，别传 .venv / 本机数据目录
tar czf rgpu-server.tgz common server deploy
scp rgpu-server.tgz <USER>@<SERVER_IP>:/tmp/
ssh <USER>@<SERVER_IP>
```

服务器上：

```bash
mkdir -p ~/rgpu && tar xzf /tmp/rgpu-server.tgz -C ~/rgpu
cd ~/rgpu
sudo bash deploy/install_server.sh
```

脚本会打印 token 和后续命令，形如：

```
================= RGPU 服务端安装完成 =================
 服务名      : rgpu-server（systemd，开机自启）
 数据目录    : /var/lib/rgpu  （jobs/<id>/... 与 envs/<fingerprint>/...）
 监听        : 0.0.0.0:8765
 Token       : <32 字符 token>
 ...
```

### 3.2 自定义参数

```bash
sudo RGPU_PORT=9000 RGPU_DATA_DIR=/data/rgpu RGPU_MAX_PARALLEL=2 \
     bash deploy/install_server.sh
```

| 环境变量 | 默认 | 含义 |
|----------|------|------|
| `RGPU_PORT` | `8765` | 监听端口 |
| `RGPU_DATA_DIR` | `/var/lib/rgpu` | 作业/环境数据目录（建议数据盘） |
| `RGPU_MAX_PARALLEL` | `1` | 最大并行作业数（systemd 的 `--max-parallel`） |
| `RGPU_APP_DIR` | `/opt/rgpu-server` | 代码安装目录 |
| `RGPU_PYTHON_BIN` | `python3` | 建 venv 用的解释器 |
| `RGPU_ENABLE_DOCKER` | `0` | 设为 `1` 启用 docker 环境模式（需 docker + nvidia-container-toolkit） |

重复执行脚本是安全的：已有 token 保留，venv 复用，只更新代码并重启服务。

### 3.3 部署后自检

```bash
sudo systemctl status rgpu-server
sudo journalctl -u rgpu-server -n 50 --no-pager

# 服务端自带原始 socket 自测（在项目/代码目录跑）
cd /opt/rgpu-server
.venv/bin/python server/selftest.py --server 127.0.0.1:8765 --token "$(sudo awk -F= '/^RGPU_TOKEN=/{print $2}' /etc/rgpu/server.env)"
```

预期输出最后一行：

```
[selftest] PASS：提交 → 上传/断线续传 → 启动 → 流式日志 → 下载产物 → 环境复用 （2 个产物）
```

### 3.4 手动部署（不用脚本）

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv ca-certificates

sudo useradd --system --create-home --home-dir /var/lib/rgpu --shell /usr/sbin/nologin rgpu
sudo mkdir -p /opt/rgpu-server /var/lib/rgpu /etc/rgpu
sudo cp -a common server /opt/rgpu-server/
sudo python3 -m venv /opt/rgpu-server/.venv
sudo chown -R rgpu:rgpu /var/lib/rgpu

TOKEN="$(cd /opt/rgpu-server && sudo .venv/bin/python -m server.main token)"
echo "RGPU_TOKEN=$TOKEN" | sudo tee /etc/rgpu/server.env
sudo chmod 640 /etc/rgpu/server.env

sudo cp deploy/rgpu-server.service /etc/systemd/system/rgpu-server.service
# 按需修改端口/数据目录后：
sudo systemctl daemon-reload
sudo systemctl enable --now rgpu-server
sudo journalctl -u rgpu-server -f
```

## 4. GPU 驱动、CUDA 与两种环境模式

服务端只负责「按客户端环境指纹准备环境 + 调度 GPU」，不绑定 CUDA 版本；
具体兼容性由下面两条路径决定：

### 4.1 venv 环境模式（默认）

- 用**宿主机的 `python3`** 创建 venv，再 `pip install -r requirements.txt`。
- 适用于：作业依赖是纯 Python / 有 CPU 轮子，或 pip 能装到的 CUDA 轮子。
- **关键兼容性**：pip 装的深度学习框架轮子（如 `torch`）自带 CUDA runtime，
  但要求**宿主机 NVIDIA 驱动**足够新。判断方法：`nvidia-smi` 右上角的
  `CUDA Version` 是「该驱动支持的最高 CUDA runtime 版本」；框架要求的 CUDA
  不能高于它。典型：驱动 `550.x` → CUDA 12.4；驱动 `535.x` → CUDA 12.2。
- Python 小版本不一致（客户端 3.12 / 服务端 3.11）不会报错：服务端按**自己的**
  解释器构建环境，`job_started.env` 会回传实际指纹，日志里也有提示。
- 宿主无外网/无 pip 源时，`pip install` 会失败并回传
  `job_finished(state=failed, exit_code=-1, log_tail=[pip 错误])` —— 这是预期行为，
  客户端本地先看到依赖错误再排查，不必登服务器。

### 4.2 docker 环境模式（`--env-mode docker`）

- 需要宿主机：`docker` + `nvidia-container-toolkit`（`docker run --gpus all` 可用），
  且服务端以 `--enable-docker` 启动（`RGPU_ENABLE_DOCKER=1` 重跑安装脚本）。
- 服务端按环境指纹在 `<data_dir>/envs/<fingerprint>/` 生成 `Dockerfile`：
  基础镜像默认 `nvidia/cuda:12.4.1-runtime-ubuntu22.04`（客户端可用
  `env.docker_image` 覆盖），`apt` 装 `python3` + `python3-pip`，再按需
  `pip install -r requirements.txt`；构建镜像标签 `rgpu-env:<指纹前 16 位>`。
- 作业在容器里执行，作业目录挂载到 `/work`，GPU 通过
  `docker run --gpus device=<分配到的索引>` 限制。
- 同一指纹第二次提交直接复用镜像（`env.state=cached`）。
- **必须注意**：基础镜像 CUDA 版本要 ≤ 宿主驱动支持的上限，且宿主机
  `nvidia-container-toolkit` 已配置；否则容器内看不到 GPU。
- 服务端未启用 docker 时，客户端提交 docker 模式会立刻收到
  `error ENV_BUILD_FAILED`（在传大文件之前就失败，省流量）。

### 4.3 服务端容器化（可选，`deploy/Dockerfile`）

```bash
# 在项目根目录构建（构建上下文是仓库根）
docker build -t rgpu-server:local -f deploy/Dockerfile .
docker run -d --name rgpu-server --restart unless-stopped \
  --gpus all -p 8765:8765 \
  -e RGPU_TOKEN='<token>' \
  -v /srv/rgpu-data:/data \
  rgpu-server:local
```

容器内跑「服务端」，作业仍用容器内 python 创建 venv（docker 模式则由宿主 docker
再起容器，需要把宿主 docker socket 挂进容器——demo 未做，建议直接用 systemd 部署）。

## 5. 端口与安全（重要）

本协议是**明文 TCP + 共享 token**（见 `docs/PROTOCOL.md` 第 1 节），只保证
「不知道 token 的人连不上」，**不保证传输加密**。生产务必按下面做：

1. **首选 SSH 隧道**（不向公网开 8765）：

   ```bash
   # 本地（Windows PowerShell / Linux 都行）
   ssh -N -L 8765:127.0.0.1:8765 <USER>@<SERVER_IP>
   # 客户端这时连本机端口即可：
   python -m client.main submit <项目目录> --entry "python train.py" \
       --server 127.0.0.1:8765 --token <token>
   ```

   隧道里流量由 SSH 加密，服务端只监听回环或内网，安全组只开 22。

2. 若一定要直连公网端口：
   - 安全组/ufw 只放行**客户端固定公网 IP**：
     `sudo ufw allow from <客户端IP> to any port 8765 proto tcp`
   - token 用 `python -m server.main token` 重新生成，别用示例值；
   - 不要在有其它租户的机器上跑（作业进程没有容器/chroot 隔离）。

3. 其它边界（demo 明确不做）：传输 TLS、多用户、磁盘配额、作业资源硬限制。
   详见 `docs/ARCHITECTURE.md` 第 4 节。

4. token 存在 `/etc/rgpu/server.env`（640 root:rgpu）；systemd 以 root 读环境变量后
   降权到 `rgpu` 运行服务。**不要把 token 提交进 git**。

## 6. 日志、重启、升级

```bash
sudo systemctl status  rgpu-server        # 状态
sudo systemctl restart rgpu-server        # 重启（会 terminate 正在跑的作业）
sudo systemctl stop    rgpu-server        # 停止
sudo journalctl -u rgpu-server -f         # 服务端实时日志
sudo journalctl -u rgpu-server -n 200     # 最近 200 行
sudo journalctl -u rgpu-server --since "1 hour ago"
```

作业级日志与状态：

```
/var/lib/rgpu/
  jobs/<job_id>/
    spec.json        # 客户端提交的任务描述
    inputs/          # 上传的代码/权重/数据
    outputs/         # 执行后按 outputs 模式收集的产物
    logs/job.log     # 作业 stdout+stderr 合并日志
    state.json       # 状态、时间戳、退出码、产物 sha256、上传记录
  envs/<fingerprint>/
    .ready           # 存在 = 环境可用（venv 模式还有 venv/ 子目录）
```

**升级**：把新代码 `scp` 上去后重新执行 `sudo bash deploy/install_server.sh`
（token 保留），或手动 `cp -a common server /opt/rgpu-server/ && sudo systemctl restart rgpu-server`。

**清理**：demo 不自动删除作业/环境（`keep_finished_jobs` 只做保留数量控制，
尚未接入定时任务）。手动清理时**只删已结束作业**：

```bash
# 列出作业状态
for d in /var/lib/rgpu/jobs/*/; do echo "$(basename "$d"): $(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['state'])" "$d/state.json" 2>/dev/null)"; done
# 删除已结束作业（示例：删除 30 天前的目录，按需改）
sudo find /var/lib/rgpu/jobs -maxdepth 1 -type d -mtime +30 -exec rm -rf {} +
# 删除不再使用的环境缓存（确认没有作业在跑）
sudo du -sh /var/lib/rgpu/envs/* | sort -h
```

## 7. 卸载

```bash
sudo systemctl disable --now rgpu-server
sudo rm -f /etc/systemd/system/rgpu-server.service
sudo systemctl daemon-reload
sudo rm -rf /opt/rgpu-server /etc/rgpu
# 数据目录确认不再需要后再删：
# sudo rm -rf /var/lib/rgpu
sudo userdel rgpu        # 若还想保留数据目录，先确认归属
```

## 8. 故障排查

| 现象 | 排查 |
|------|------|
| `systemctl status` 起不来 | `journalctl -u rgpu-server -n 100`；最常见：token 文件缺失/权限不对（服务端无 token 会 exit 2） |
| 客户端连不上 | 服务端 `ss -ltnp \| grep 8765`；安全组/ufw；是否只监听了 127.0.0.1 |
| `BAD_AUTH` | 客户端 `--token` 与 `/etc/rgpu/server.env` 里 `RGPU_TOKEN` 不一致 |
| `ENV_BUILD_FAILED` | `logs/job.log` 里 `[env]` 行；venv 模式多为 pip 无外网/依赖版本不存在；docker 模式看 `docker build` 报错 |
| 作业 `failed` 退出码非 0 | 看 `logs/job.log` 与 `job_finished.log_tail`；跟本地跑一次对比 |
| 作业 `timeout` | 客户端 `--timeout` 调大；服务端按 `resources.timeout_s` kill 进程树 |
| 产物为空 | 作业把文件写到了 `workdir` 之外，或 `outputs` 模式不匹配；模式相对**作业 cwd** 解释 |
| 无 GPU / 调度不到 | `python -m server.main gpus` 应输出 JSON 列表；返回 `[]` 说明 `nvidia-smi` 不可用，服务端仍以 CPU 模式工作 |
| 磁盘满 | 清理 `jobs/`、`envs/`（第 6 节） |

## 9. 未验证清单（逐条）

本机是 Windows 开发机，以下**全部未在真实 Linux/GPU 环境实测**：

1. 「未验证」`install_server.sh` 在 Ubuntu 22.04 / 24.04 上的完整执行
   （开发机无 Linux；Git Bash 在沙箱下启动即 `couldn't create signal pipe, Win32 error 5`，
   连 `bash -n` 语法检查都没能跑成，只做了逐行人工审阅）。
2. 「未验证」`apt-get install python3-venv` 之后的 `python3 -m venv` 行为；
   不同发行版的 `ensurepip` 差异。
3. 「未验证」systemd 单元（`rgpu-server.service`）实际启动、开机自启、
   `EnvironmentFile` + `$RGPU_EXTRA_ARGS` 展开、`KillMode=mixed` 停止行为。
4. 「未验证」安装脚本里 `sed` 模板替换与 `useradd --system --home-dir` 的实际结果。
5. 「未验证」`ufw` 提示逻辑（只打印命令，不自动改防火墙）。
6. 「未验证」docker 环境模式：开发机 docker daemon 未运行
   （`docker version` 报 `failed to connect to the docker API at npipe:...`），
   镜像构建、`--gpus device=...` 设备限制、挂载作业目录、容器内产物回传均未跑过。
7. 「未验证」`deploy/Dockerfile` 的 `docker build` 与 `docker run --gpus all`。
8. 「未验证」`nvidia-container-toolkit` 安装与 GPU 容器可见性。
9. 「未验证」Linux 上 `venv` 模式真实 `pip install -r requirements.txt`
   （开发机无 pip、无外网；只用 `--without-pip` 的零依赖路径自测过）。
10. 「未验证」真实 NVIDIA GPU 上的作业调度：
    `nvidia-smi` 解析逻辑用本机真实输出验证过（解析出 RTX 4050 / 581.08 / CUDA 13.0），
    但「分配 `CUDA_VISIBLE_DEVICES=0` 后训练脚本只看到 1 张卡」没有在 Linux 上实测。
11. 「未验证」跨平台进程组 kill：Windows 用 `taskkill /T`（自测超时用例通过），
    Linux 走 `os.killpg(..., SIGKILL)`（未实测）。
12. 「未验证」Python 3.10（Ubuntu 22.04 自带）上的兼容性；代码按 ≥3.9 编写，
    只在 3.12.14 上跑过测试。

**已在开发机（Windows，CPU 模式）实测的部分**：服务端 CLI（`gpus`/`token`/`serve`）、
`RGPU_SERVER_READY` 就绪行、握手鉴权、上传/断点续传、venv `--without-pip` 环境构建与
`cached` 复用、作业执行与流式日志、超时 kill、产物 glob 收集与下载校验、
`python server/selftest.py` 全流程 PASS、以及 `tests/` 里的协议与端到端用例。
