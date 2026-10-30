#!/usr/bin/env bash
# =============================================================================
# RGPU 服务端一键安装脚本（目标系统：Ubuntu 22.04 / 24.04，Debian 12 大致可用）
#
# 用法（在项目根目录执行）：
#   sudo bash deploy/install_server.sh
#   sudo RGPU_PORT=9000 RGPU_DATA_DIR=/data/rgpu RGPU_MAX_PARALLEL=2 \
#        bash deploy/install_server.sh
#   sudo RGPU_ENABLE_DOCKER=1 bash deploy/install_server.sh   # 需先装 docker + nvidia-container-toolkit
#
# 做的事：
#   1. apt 安装 python3 / python3-venv / python3-pip / ca-certificates
#   2. 创建系统用户 rgpu（无登录 shell），数据目录归它所有
#   3. 拷贝 common/ 与 server/ 到 /opt/rgpu-server，建 .venv
#   4. 生成/保留 /etc/rgpu/server.env 里的 token（600 权限）
#   5. 安装并启动 systemd 服务 rgpu-server
#   6. 打印 token、服务状态、防火墙与 SSH 隧道建议
#
# 重复执行是安全的：已有 token 会保留，已有 venv 会复用，只更新代码并重启服务。
#
# 注意：本脚本在 Windows 开发机上**未实测**（只做过静态审阅），
#       详见 deploy/README.md「未验证」清单。请先在一台临时服务器上跑一遍。
# =============================================================================
set -euo pipefail

APP_DIR="${RGPU_APP_DIR:-/opt/rgpu-server}"
DATA_DIR="${RGPU_DATA_DIR:-/var/lib/rgpu}"
ENV_DIR="${RGPU_ENV_DIR:-/etc/rgpu}"
ENV_FILE="${RGPU_ENV_FILE:-$ENV_DIR/server.env}"
PORT="${RGPU_PORT:-8765}"
MAX_PARALLEL="${RGPU_MAX_PARALLEL:-1}"
SERVICE_NAME="rgpu-server"
SERVICE_USER="rgpu"
ENABLE_DOCKER="${RGPU_ENABLE_DOCKER:-0}"
PYTHON_BIN="${RGPU_PYTHON_BIN:-python3}"

log()  { printf '\033[1;32m[rgpu]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[rgpu]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[rgpu] 错误:\033[0m %s\n' "$*" >&2; exit 1; }

# --- 0. 前置检查 -------------------------------------------------------------
[ "$(id -u)" -eq 0 ] || die "请用 root 运行：sudo bash $0"
command -v apt-get >/dev/null 2>&1 || die "只支持 Debian/Ubuntu（找不到 apt-get）"
command -v systemctl >/dev/null 2>&1 || die "需要 systemd（找不到 systemctl）"

if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    case "${ID:-}:${VERSION_ID:-}" in
        ubuntu:22.04|ubuntu:24.04|debian:12) : ;;
        *) warn "目标系统为 ${PRETTY_NAME:-未知}；本仓库只在 Ubuntu 22.04/24.04 上设计过（未验证）" ;;
    esac
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$SRC_ROOT/server/main.py" ] || die "找不到 $SRC_ROOT/server/main.py，请在项目根目录下执行 deploy/install_server.sh"
[ -f "$SCRIPT_DIR/rgpu-server.service" ] || die "缺少 $SCRIPT_DIR/rgpu-server.service"
[ -f "$SRC_ROOT/common/protocol.py" ] || die "找不到 common/，请确认代码完整"

log "安装目录: $APP_DIR"
log "数据目录: $DATA_DIR（jobs/envs，建议挂在数据盘）"
log "监听端口: $PORT（最大并行 $MAX_PARALLEL）"

# --- 1. 系统依赖 --------------------------------------------------------------
log "安装系统依赖：python3 python3-venv python3-pip ca-certificates"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ca-certificates

if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
    die "需要 Python >= 3.9，当前 $("$PYTHON_BIN" --version 2>&1)"
fi
log "Python: $("$PYTHON_BIN" --version 2>&1)"

# --- 2. 服务用户与目录 --------------------------------------------------------
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    log "创建系统用户 $SERVICE_USER"
    useradd --system --create-home --home-dir "$DATA_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
fi
mkdir -p "$APP_DIR" "$DATA_DIR/envs" "$DATA_DIR/jobs" "$ENV_DIR"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$DATA_DIR"
chmod 750 "$DATA_DIR"

# --- 3. 拷贝代码 --------------------------------------------------------------
log "拷贝 common/ 与 server/ 到 $APP_DIR"
rm -rf "$APP_DIR/common" "$APP_DIR/server"
cp -a "$SRC_ROOT/common" "$APP_DIR/common"
cp -a "$SRC_ROOT/server" "$APP_DIR/server"
find "$APP_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
chown -R root:root "$APP_DIR"
chmod -R a+rX "$APP_DIR"

# --- 4. Python 环境 -----------------------------------------------------------
if [ ! -x "$APP_DIR/.venv/bin/python" ]; then
    log "创建虚拟环境 $APP_DIR/.venv"
    "$PYTHON_BIN" -m venv "$APP_DIR/.venv"
fi
# 服务端只用标准库；这里的 pip 升级失败（离线/无外网）不影响运行
if ! "$APP_DIR/.venv/bin/python" -m pip install --quiet --upgrade pip; then
    warn "pip 升级失败（离线环境可忽略：服务端只依赖标准库）"
fi

# --- 5. token 与运行参数 ------------------------------------------------------
TOKEN=""
if [ -f "$ENV_FILE" ]; then
    TOKEN="$(grep -E '^RGPU_TOKEN=' "$ENV_FILE" 2>/dev/null | head -n 1 | cut -d= -f2- || true)"
fi
if [ -z "$TOKEN" ]; then
    log "生成新 token"
    TOKEN="$(cd "$APP_DIR" && "$APP_DIR/.venv/bin/python" -m server.main token)"
fi

EXTRA_ARGS=""
if [ "$ENABLE_DOCKER" = "1" ]; then
    command -v docker >/dev/null 2>&1 || die "RGPU_ENABLE_DOCKER=1 但未安装 docker（先装 docker + nvidia-container-toolkit，或去掉该变量）"
    EXTRA_ARGS="--enable-docker"
    warn "已启用 docker 环境模式：需要 docker + nvidia-container-toolkit，且作业镜像基础层要匹配宿主 CUDA（未验证）"
fi

umask 077
cat > "$ENV_FILE" <<EOF
# 由 deploy/install_server.sh 生成；修改后 systemctl restart $SERVICE_NAME 生效
RGPU_TOKEN=$TOKEN
RGPU_EXTRA_ARGS=$EXTRA_ARGS
EOF
chown root:"$SERVICE_USER" "$ENV_FILE" 2>/dev/null || true
chmod 640 "$ENV_FILE"
log "token 已写入 $ENV_FILE（权限 640）"

# --- 6. systemd 安装 ----------------------------------------------------------
log "安装 systemd 服务 $SERVICE_NAME"
sed -e "s|/opt/rgpu-server|$APP_DIR|g" \
    -e "s|/var/lib/rgpu|$DATA_DIR|g" \
    -e "s|--port 8765|--port $PORT|g" \
    -e "s|--max-parallel 1|--max-parallel $MAX_PARALLEL|g" \
    -e "s|User=rgpu|User=$SERVICE_USER|g" \
    -e "s|Group=rgpu|Group=$SERVICE_USER|g" \
    "$SCRIPT_DIR/rgpu-server.service" > "/etc/systemd/system/$SERVICE_NAME.service"

systemctl daemon-reload
systemctl enable "$SERVICE_NAME" >/dev/null 2>&1 || true
systemctl restart "$SERVICE_NAME"

# --- 7. 等待端口就绪 ----------------------------------------------------------
READY=0
for _ in $(seq 1 30); do
    if "$APP_DIR/.venv/bin/python" -c 'import socket, sys
try:
    socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1.0).close()
except OSError:
    sys.exit(1)' "$PORT" >/dev/null 2>&1
    then
        READY=1
        break
    fi
    sleep 1
done

if [ "$READY" -ne 1 ]; then
    warn "服务未在 30s 内监听 127.0.0.1:$PORT，下面是排查信息："
    systemctl --no-pager --full status "$SERVICE_NAME" || true
    journalctl -u "$SERVICE_NAME" -n 50 --no-pager || true
    die "启动失败，请按上面的日志排查（常见原因：Python 版本 / 端口被占用 / token 文件权限）"
fi

# --- 8. 完成信息 --------------------------------------------------------------
log "服务已启动：systemctl status $SERVICE_NAME"
systemctl --no-pager --full status "$SERVICE_NAME" | head -n 12 || true

HOST_FQDN="$(hostname -f 2>/dev/null || hostname)"
cat <<EOF

================= RGPU 服务端安装完成 =================
 服务名      : $SERVICE_NAME（systemd，开机自启）
 代码目录    : $APP_DIR
 数据目录    : $DATA_DIR  （jobs/<id>/... 与 envs/<fingerprint>/...）
 环境变量    : $ENV_FILE（含 RGPU_TOKEN，权限 640）
 监听        : 0.0.0.0:$PORT
 Token       : $TOKEN
 主机名      : $HOST_FQDN

常用命令：
  sudo systemctl status  $SERVICE_NAME
  sudo systemctl restart $SERVICE_NAME
  sudo journalctl -u $SERVICE_NAME -f          # 实时日志
  sudo journalctl -u $SERVICE_NAME -n 200      # 最近 200 行

安全建议（明文 TCP，务必不要裸奔到公网）：
  1) 优先走 SSH 隧道，本机执行：
       ssh -N -L $PORT:127.0.0.1:$PORT <你的用户名>@<服务器IP>
     客户端 --server 填 127.0.0.1:$PORT
  2) 或者只放行客户端公网 IP（ufw 示例）：
       sudo ufw allow from <客户端公网IP> to any port $PORT proto tcp
  3) 云厂商安全组只开放 22 端口，$PORT 端口不要对 0.0.0.0/0 开放

客户端示例：
  python -m client.main submit <项目目录> --entry "python train.py" \\
      --server <服务器IP>:$PORT --token <上面的 Token>
=======================================================
EOF

if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -qi active; then
    warn "检测到 ufw 已启用：请按上面的 ufw 示例放行，或改用 SSH 隧道（推荐）"
fi
if command -v docker >/dev/null 2>&1; then
    log "检测到 docker（若要用 docker 环境模式，需 nvidia-container-toolkit 并重跑本脚本时加 RGPU_ENABLE_DOCKER=1）"
fi
log "完成。"
