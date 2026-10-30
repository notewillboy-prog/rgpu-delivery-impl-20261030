# RGPU 通信协议 v1

> 状态：**已冻结**（2026-10-22，Lead）。客户端、服务端、测试都按本文件实现；
> 需要变更时由 Lead 改 `common/` 与本文档，并通知全部会话。

## 1. 传输层

- TCP 单连接，明文（生产环境建议 SSH 隧道，见 `deploy/README.md`）。
- 每一帧格式（大端）：

```
+----------------+------------+------------------+
| payload_len u32| msg_type u8| payload (bytes)  |
+----------------+------------+------------------+
```

`payload_len` 只算 payload；单帧 payload ≤ 16 MiB。
`msg_type`：`1` = JSON（UTF-8，必须是对象，含 `kind`），`2` = BLOB（原始二进制）。

- 每个二进制分片都**先**发一条 JSON 头（`file_chunk` / `output_chunk`），
  **紧跟**一个 BLOB 帧；两端必须严格一问一答地读取，不能并发乱序。
- 客户端每 20 s 发 `ping`，服务端回 `pong`；服务端 300 s 无消息则断开。
- 所有消息均带 `kind`；错误消息为 `{"kind":"error","code":...,"message":...}`。

## 2. 鉴权与握手

```jsonc
// C→S
{"kind":"hello","protocol":1,"role":"client","client_version":"0.1.0","token":"<token>"}
// S→C（成功）
{"kind":"hello_ok","protocol":1,"server_id":"srv-xxxx","hostname":"gpu-node-1",
 "platform":"linux","python":"3.11.9","max_frame":16777216,
 "gpus":[{"index":0,"name":"NVIDIA A100","memory_total_mb":40960,"memory_free_mb":40000,
          "driver":"550.54","cuda":"12.4"}],
 "max_parallel_jobs":1,"env_cache_dir":"/root/.rgpu-server/envs"}
// S→C（失败，随后断开）
{"kind":"error","code":"BAD_AUTH","message":"token 无效"}
```

客户端在收到 `hello_ok` 之前不得发送其他消息。`role` 固定 `"client"`。

## 3. 提交任务与上传

### 3.1 任务描述 `spec`（`common.models.JobSpec`）

```jsonc
{
  "job_name": "incremental-train-demo",
  "entry_argv": ["python", "train.py", "--epochs", "3", "--data", "data.csv"],
  "workdir": ".",
  "inputs": [
    {"path": "train.py", "size": 2314, "sha256": "…64hex…"},
    {"path": "model.json", "size": 512, "sha256": "…"},
    {"path": "data.csv", "size": 10000, "sha256": "…"}
  ],
  "outputs": ["outputs/**"],
  "env": {
    "mode": "venv",
    "python_version": "3.12",
    "requirements": ["numpy==2.1.0"],
    "fingerprint": "…64hex…",
    "env_vars": {},
    "docker_image": "",
    "reuse": true
  },
  "resources": {"gpu": true, "gpu_count": 1, "timeout_s": 600, "cpus": 0, "memory_gb": 0},
  "meta": {"mode": "incremental_training", "base_model": "model.json", "dataset": "data.csv"},
  "client_version": "0.1.0"
}
```

`inputs[].path` 一律 POSIX 相对路径，禁止绝对路径与 `..`。
`env.fingerprint` 由 `common.hashing.env_fingerprint` 计算，服务端会重算校验。

### 3.2 提交流程

```jsonc
// C→S；resume_job_id 可选：断线重连时带上原 job_id 续传
{"kind":"job_submit","spec":{...},"resume_job_id":null}
// S→C：接受任务，返回还缺哪些文件（none 时 missing=[]）
{"kind":"job_accepted","job_id":"job-20261022-173000-ab12cd",
 "missing":[{"path":"train.py","size":2314,"sha256":"…"}],
 "present":["data.csv"],
 "env":{"fingerprint":"…","reuse":true},
 "chunk_size":1048576}
```

### 3.3 上传单个文件（分片、可续传）

```jsonc
// C→S：当前分片头
{"kind":"file_chunk","path":"data.csv","offset":0,"length":1048576,"sha256":"<分片 sha256>"}
// 紧跟一个 BLOB 帧，payload 长度必须等于 length
// S→C：每片确认；offset = 服务端下一个期望偏移
{"kind":"file_ok","path":"data.csv","offset":1048576}
// C→S：最后一个分片发完后
{"kind":"file_done","path":"data.csv"}
// S→C：整文件 sha256 校验通过
{"kind":"file_ok","path":"data.csv","done":true}
```

- 分片 sha256 不匹配 → `error HASH_MISMATCH`，客户端重发该分片。
- 整文件 sha256 不匹配 → `error HASH_MISMATCH`，客户端从 0 重传该文件。
- 连接断开后重连：重新 `job_submit`（带 `resume_job_id`），`present` 里的文件不再上传，
  `missing` 里的文件从 `file_chunk` 的 `offset=0` 开始（未完成文件服务端会截断重传，
  demo 实现按整文件重传即可；`file_ok` 的 `offset` 机制保留给分片级续传扩展）。

### 3.4 启动与执行

```jsonc
// C→S
{"kind":"job_start","job_id":"job-…"}
// S→C：环境就绪、开始执行
{"kind":"job_started","job_id":"job-…",
 "env":{"fingerprint":"…","state":"cached"}}   // cached | built
// 环境构建中（可选，多次）
{"kind":"job_progress","job_id":"job-…","state":"env_building","progress":0.3,
 "message":"pip install -r requirements.txt (3/7)"}
// 执行期流式事件
{"kind":"job_event","job_id":"job-…","seq":1,"stream":"stdout","data":"epoch 1/3\n","ts":1761…}
// 结束（成功或失败都会发）
{"kind":"job_finished","job_id":"job-…","state":"succeeded","exit_code":0,
 "duration_s":12.3,
 "outputs":[{"path":"outputs/model.json","size":612,"sha256":"…","kind":"file"}],
 "log_tail":["…最后 20 行…"]}
```

`state`: `succeeded | failed | timeout | cancelled`。
环境构建失败时：先发 `job_progress`(state=env_building) 再发
`job_finished`(state=failed, exit_code=-1)，并在 `log_tail` 里给出 pip 错误。
任务超时：`state=timeout`，exit_code 为 `-1`。

### 3.5 状态查询

```jsonc
// C→S
{"kind":"job_status","job_id":"job-…"}
// S→C
{"kind":"job_status_ok","job_id":"job-…","state":"running","progress":0.4,
 "outputs":[...],"created_at":1761…,"started_at":1761…,"finished_at":null}
```

### 3.6 产物下载

```jsonc
// C→S：列产物
{"kind":"output_list","job_id":"job-…"}
// S→C
{"kind":"output_list_ok","job_id":"job-…","outputs":[...]}
// C→S：按偏移请求分片（offset 从 0 开始）
{"kind":"output_chunk","job_id":"job-…","path":"outputs/model.json","offset":0}
// S→C：分片头 + BLOB
{"kind":"output_chunk","job_id":"job-…","path":"outputs/model.json","offset":0,
 "length":612,"sha256":"<分片 sha256>","eof":true}
// 客户端校验分片 sha256、追加写入本地文件；offset+length < size 时继续请求
```

## 4. 错误码

| code | 含义 | 客户端处理 |
|------|------|-----------|
| `BAD_PROTOCOL` | 版本/帧格式错 | 终止，提示升级 |
| `BAD_REQUEST` | 字段缺失/非法 | 终止，打印 message |
| `BAD_AUTH` | token 无效 | 终止，提示检查 token |
| `NOT_FOUND` | job/文件/产物不存在 | 终止 |
| `CONFLICT` | 重复提交/状态不允许 | 终止 |
| `HASH_MISMATCH` | 分片或整文件哈希不符 | 重传（有限次） |
| `ENV_BUILD_FAILED` | 远端环境构建失败 | 打印 log_tail |
| `JOB_FAILED` | 作业非零退出 | 返回退出码 |
| `TIMEOUT` | 超时 | 终止 |
| `BUSY` | 队列满 | 稍后重试 |
| `INTERNAL` | 服务端异常 | 终止并上报 |

## 5. 服务端作业目录与约定

```
<data_dir>/
  envs/<fingerprint>/            # venv 或 docker 标记；存在 .ready 即复用
  jobs/<job_id>/
    spec.json                    # 原始 JobSpec
    inputs/…                     # 客户端上传的文件，保持相对路径
    outputs/…                    # 任务产物（entry 的 workdir 下的相对路径）
    logs/job.log                 # stdout+stderr 合并日志
    state.json                   # JobState、时间戳、退出码、产物清单
```

- 作业进程 cwd = `jobs/<job_id>/inputs`；`workdir` 相对该目录。
- 调度器分配的 GPU 通过环境变量 `CUDA_VISIBLE_DEVICES=<逗号分隔索引>` 注入。
- venv 环境：`<data_dir>/envs/<fingerprint>/venv`（`python -m venv` + `pip install -r`）。
- docker 模式：镜像标签 `rgpu-env:<fingerprint 前 16 位>`，容器内挂载作业目录。

## 6. 客户端 CLI 契约（冻结）

```text
# 服务端
python -m server.main serve --host 0.0.0.0 --port 8765 --token <token> --data-dir <dir>
                          [--max-parallel 1] [--python-bin python3] [--enable-docker] [--idle-timeout 300]
python -m server.main gpus          # 打印探测到的 GPU（JSON）
python -m server.main token         # 生成 token

# 客户端
python -m client.main submit <PROJECT_DIR>
    --entry "python train.py --epochs 3 --data data.csv"
    --server 127.0.0.1:8765 --token <token>
    [--requirements requirements.txt] [--env-mode venv|docker]
    [--include "*.py" ...] [--exclude ...] [--outputs "outputs/**"]
    [--gpu|--no-gpu] [--timeout 600] [--name demo]
    [--download-dir <dir>] [--watch|--no-watch] [--json]
python -m client.main status <JOB_ID> --server … --token …
python -m client.main gui [--config <path>]
python -m client.main config --server … --token … [--save]
```

退出码：`0` 任务成功；`1` 任务失败/超时；`2` 参数或配置错误；
`3` 连接/网络错误；`4` 服务端拒绝（鉴权、环境等）。

服务端就绪日志（测试用）必须包含一行：

```
RGPU_SERVER_READY host=<host> port=<port> data_dir=<dir>
```

## 7. 兼容性

- 客户端支持 Windows / Linux；服务端仅 Linux（本仓库在 Windows 上以 CPU 模式自测）。
- Python ≥ 3.9，全部标准库（示例任务也不依赖第三方包）。
- 协议版本号变更时递增 `PROTOCOL_VERSION` 并更新本文档。
