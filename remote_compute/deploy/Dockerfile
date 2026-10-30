# =============================================================================
# RGPU 服务端容器镜像（可选部署方式之一）
#
# 目标：在带 NVIDIA GPU 的 Linux 主机上，把「RGPU 服务端」本体跑在容器里，
#       从而不污染宿主机 Python 环境（宿主机仍需 NVIDIA 驱动 + docker +
#       nvidia-container-toolkit）。
#
# 这不是 docker「环境模式」的作业镜像：环境模式的镜像是服务端 envbuilder 按
# 环境指纹动态生成的（基础层同样是 nvidia/cuda runtime + python3-pip，
# 见 deploy/README.md 第 4 节）。
#
# 构建（在项目根目录执行，构建上下文必须是仓库根）：
#   docker build -t rgpu-server:local -f deploy/Dockerfile .
#
# 运行（token 通过环境变量传入，数据目录挂卷）：
#   docker run -d --name rgpu-server --restart unless-stopped \
#     --gpus all -p 8765:8765 \
#     -e RGPU_TOKEN='<你的 token>' \
#     -v /srv/rgpu-data:/data \
#     rgpu-server:local
#
# 注意：本文件在开发机（Windows，docker daemon 未运行）上**未实测**，
#       只做了静态审阅。见 deploy/README.md「未验证」清单。
# =============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    RGPU_DATA_DIR=/data

# 服务端只用标准库；python3-venv 供「venv 环境模式」创建作业环境用
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/rgpu-server
COPY common/ /opt/rgpu-server/common/
COPY server/ /opt/rgpu-server/server/
COPY deploy/ /opt/rgpu-server/deploy/

# 与宿主机部署保持同样的目录结构：<data_dir>/jobs、<data_dir>/envs
RUN python3 -m venv /opt/rgpu-server/.venv \
 && mkdir -p /data/jobs /data/envs \
 && chmod -R a+rX /opt/rgpu-server

VOLUME ["/data"]
EXPOSE 8765

# token 必须通过 -e RGPU_TOKEN=... 提供（serve 支持该环境变量）；缺失时启动即报错退出
ENTRYPOINT ["/opt/rgpu-server/.venv/bin/python", "-m", "server.main"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8765", "--data-dir", "/data"]
