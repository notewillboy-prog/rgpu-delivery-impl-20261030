# 增量训练示例（纯标准库）

这是 RGPU 远程算力调度 demo 的示例训练任务：**代码 + 基座模型权重 + 增量数据** 一起
被客户端打包投递到远端服务器执行，远端把新的权重与指标回传给本地。

## 目录内容

| 文件 | 说明 |
|------|------|
| `train.py` | 纯标准库线性回归训练脚本；`--model` 存在则继续训练（增量） |
| `data.csv` | 两列 `x,y` 数据集（120 行） |
| `model.json` | 基座模型初始权重（`w`,`b`,`runs`,`epochs_total`） |
| `requirements.txt` | 只有注释：演示「环境指纹」，无第三方依赖 |
| `outputs/` | 运行时产物目录（已被 `.gitignore` 与客户端默认忽略规则排除） |

## 本地直接运行（自测用）

```powershell
cd examples\incremental_train
& ..\..\.venv\Scripts\python.exe train.py --epochs 2 --data data.csv --out outputs --model model.json
```

**连续运行两次**即可验证增量训练：

1. 第一次：`model.json`（runs=0）→ 训练 2 epoch → 写出 `outputs/model.json`
   （runs=1, epochs_total=2），并就地更新 `model.json`；
2. 第二次：读取上次的权重继续训练 → `outputs/metrics.json` 里
   `base_model_loaded=true`、`start_loss` 等于第一次的 `final_loss`，且 `final_loss` 更低。

可加 `--simulate-seconds 3` 放慢训练，便于观察客户端/服务端的**流式日志**。

## 用 RGPU 客户端一键投递

1. 启动服务端（Linux；本机 Windows 可用 CPU 模式自测）：

   ```powershell
   & .venv\Scripts\python.exe -m server.main serve --host 127.0.0.1 --port 8765 --token demo-token --data-dir _rgpu_data
   ```

2. 命令行提交（在项目根目录运行）：

   ```powershell
   & .venv\Scripts\python.exe -m client.main submit examples/incremental_train `
       --entry "python train.py --epochs 3 --data data.csv" `
       --server 127.0.0.1:8765 --token demo-token `
       --download-dir rgpu-outputs
   ```

   客户端会自动识别 `requirements.txt`、`model.json` + `data.csv`（增量训练 meta），
   并忽略 `.git/ .venv/ __pycache__/ outputs/ *.log` 等本地文件。

3. 或打开 GUI 一键投递：

   ```powershell
   & .venv\Scripts\python.exe -m client.main gui
   ```

   项目目录选本目录，入口命令填 `python train.py --epochs 3 --data data.csv`，
   点击「开始增量训练」，日志区实时显示 epoch 日志，结束后把
   `outputs/model.json`、`outputs/metrics.json` 下载到本地下载目录。

## 增量训练是怎么闭环的

- **上传**：客户端把整个项目目录做成清单（路径 + 大小 + sha256）上传，
  包含基座模型权重 `model.json` 和本次新增的数据 `data.csv`；
- **执行**：远端在隔离作业目录中运行入口命令，`train.py` 读到 `model.json` 则继续训练；
- **回传**：远端把 `outputs/model.json`、`outputs/metrics.json` 回传并校验 sha256；
- **下一轮增量**：把下载得到的 `outputs/model.json` 覆盖回项目里的 `model.json`
  （或让下载目录直接作为下一轮项目目录），再次点击「开始增量训练」即可。

每次运行都会打印 JSON 指标，例如：

```json
{
  "mode": "incremental_training",
  "base_model_loaded": true,
  "start_loss": 0.001234,
  "final_loss": 0.000456,
  "improvement": 0.000778,
  "epochs_total": 4,
  "runs": 2
}
```

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--epochs` | `3` | 本次训练轮数 |
| `--lr` | `0.05` | 学习率 |
| `--data` | `data.csv` | 数据集（两列 x,y） |
| `--model` | `model.json` | 基座模型；存在则继续训练，训练后就地更新 |
| `--out` | `outputs` | 产物目录，写 `model.json` 与 `metrics.json` |
| `--simulate-seconds` | `0` | 模拟耗时（按 epoch 均摊 sleep），用于演示流式日志 |

> 作业进程的工作目录是服务端的 `jobs/<job_id>/inputs`，所以脚本里的相对路径
> （`data.csv`、`model.json`、`outputs`）都以「项目目录」为根，与本地运行一致。
