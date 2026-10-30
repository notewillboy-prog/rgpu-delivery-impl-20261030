# client/ — RGPU 客户端

Windows/Linux 通用的远程算力调度客户端，纯标准库（tkinter 仅 GUI 使用）。
协议契约见 [`../docs/PROTOCOL.md`](../docs/PROTOCOL.md)（v1，已冻结）。

## 模块

| 文件 | 职责 |
|------|------|
| `main.py` | CLI：`submit` / `status` / `gui` / `config`，`--json` 机器可读输出 |
| `packager.py` | 项目目录 → `FileEntry` 清单（sha256、POSIX 相对路径、忽略规则、requirements、JobSpec） |
| `connection.py` | TCP 帧连接：hello/hello_ok 握手、20s ping 心跳、socket 超时、有限次断线重连 |
| `uploader.py` | 1 MiB 分片上传：逐片 sha256、`file_ok.offset` 推进、分片/整文件有限重传 |
| `runner.py` | submit → job_accepted → 上传 → job_start → 流式 job_event → job_finished → output_list/分片下载（校验 sha256）→ 退出码 |
| `gui.py` | tkinter 单窗口一键增量训练；网络工作在子线程，`queue` 回主线程刷新 UI |
| `selftest.py` | 本地假服务端自测（threading + `common.protocol`），不依赖 `server/` |

## CLI（PROTOCOL.md 第 6 节）

```powershell
# 提交并跟踪（在项目根目录运行）
& .venv\Scripts\python.exe -m client.main submit examples/incremental_train `
    --entry "python train.py --epochs 3 --data data.csv" `
    --server 127.0.0.1:8765 --token <token> `
    --download-dir rgpu-outputs
```

常用可选项：`--requirements`、`--env-mode venv|docker`、`--include/--exclude/--outputs`、
`--gpu/--no-gpu`、`--timeout 600`、`--name`、`--watch/--no-watch`、`--json`、
`--incremental/--no-incremental`、`--python-version`、`--chunk-size`。

- 默认忽略 `.git/ .venv/ __pycache__/ node_modules/ outputs/ *.log` 等；
  `--include/--exclude` 不含 `/` 的模式匹配任意深度的文件名，含 `/` 匹配整条相对路径。
- 自动识别项目下 `requirements.txt` → 归一化依赖 → `env.fingerprint`（服务端复算校验）。
- 项目里同时有基座模型（如 `model.json`）与数据集（如 `data.csv`）时自动填增量训练 meta：
  `{"mode":"incremental_training","base_model":"model.json","dataset":"data.csv"}`。
  `--no-incremental` 可关闭，`--incremental` 可强制。
- `submit` 实时把远端日志写到终端/日志区；产物按服务端相对路径下载到 `--download-dir`
  （默认 `rgpu-outputs`），逐片与整文件 sha256 都校验。
- `--json`：stdout 只有一行 JSON，人类可读日志全部走 stderr（便于脚本/测试解析）。

`--json` 成功结果示例：

```json
{"ok": true, "job_id": "job-...", "state": "succeeded", "exit_code": 0,
 "client_exit": 0, "duration_s": 1.2, "download_dir": "D:\\...\\rgpu-outputs",
 "outputs": [{"path": "outputs/model.json", "local_path": "...", "size": 188,
              "sha256": "...", "verified": true}],
 "env": {"fingerprint": "...", "state": "cached"}, "spec": {...}}
```

失败结果包含 `"ok": false` 与 `"error": {"code": "...", "message": "..."}`；
任务非零退出时 `"exit_code"` 保留远端退出码。

## 退出码

| 码 | 含义 |
|----|------|
| 0 | 任务成功（产物已下载并通过 sha256 校验） |
| 1 | 任务失败 / 超时 / 取消（远端非零退出，`error.code = JOB_FAILED/TIMEOUT`） |
| 2 | 参数或配置错误（缺 token、打包失败、入口命令非法、本地文件已变化…） |
| 3 | 连接 / 网络错误（含重连失败） |
| 4 | 服务端拒绝（`BAD_AUTH`/`BAD_PROTOCOL`/`ENV_BUILD_FAILED`/`HASH_MISMATCH` 重试耗尽…） |

## 断线续传

- 上传中断：`connection.reconnect()` 有限次重连（默认 3 次，指数退避）→ 重新
  `job_submit` 并带 `resume_job_id` → 服务端返回 `present`（跳过）与 `missing`（重传）；
- 执行中断：重连后重新附着作业，并用 `job_status` 轮询到终态（协议内兼容路径）；
- 下载中断：重连后重新 `output_list` 并重下当前产物，分片与整文件 sha256 双重校验。

## 自测

```powershell
& .venv\Scripts\python.exe client/selftest.py
```

覆盖：packager 全规则、成功链路（含注入分片 HASH_MISMATCH 重传 + 上传中途断线续传）、
远端失败退出码 1、鉴权失败 4、连接失败 3、CLI `--json`/`status`/`config --save`。

## 已知限制

- 协议 3.3 约定「未完成文件整文件重传」，因此分片级断点续传尚未启用（`file_ok.offset` 机制已实现）。
- 执行期断线后的跟踪依赖服务端支持 `job_status`；若服务端丢失作业则按连接错误处理。
- token 以明文放在配置文件（`%APPDATA%\rgpu\client.json` 或 `--config` 指定路径），
  生产环境建议 SSH 隧道 + 限制文件权限。
- tkinter GUI 需要在有图形界面的会话中运行；无显示环境（headless Linux）请用 CLI。
