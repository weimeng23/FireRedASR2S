# syntax=docker/dockerfile:1

ARG UV_VERSION=0.11.28
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# torch 的 cu126 wheel 自带完整 CUDA 运行时与 cuDNN，无需 CUDA 基础镜像；
# 驱动 libcuda.so 由 nvidia-container-toolkit 在运行时从宿主注入。
FROM ubuntu:24.04

# 补回 nvidia/cuda 基底原本预置的变量：legacy nvidia runtime 靠它们决定设备注入
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

COPY --from=uv /uv /usr/local/bin/uv
RUN sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.cloud.tencent.com/ubuntu/|g' \
        -e 's|http://security.ubuntu.com/ubuntu/|http://mirrors.cloud.tencent.com/ubuntu/|g' \
        /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 ca-certificates libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/FireRedASR2S
# copy 链接模式配合下面的 cache mount（缓存与 venv 不同文件系统，硬链接不可用）
# UV_CACHE_DIR 必须与 cache mount 的 target 一致，否则缓存会静默落进镜像层
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_CACHE_DIR=/root/.cache/uv \
    PATH=/opt/FireRedASR2S/.venv/bin:${PATH} \
    PYTHONUNBUFFERED=1

# 依赖层：只依赖 pyproject.toml，源码改动不会使其失效
# Match the repository's existing image policy: uv.lock is excluded by .dockerignore.
# V100 (Volta) requires cu126 wheels; override only this image's project copy.
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    sed -i \
        -e 's/pytorch-cu130/pytorch-cu126/g' \
        -e 's|https://download.pytorch.org/whl/cu130|https://download.pytorch.org/whl/cu126|g' \
        pyproject.toml \
    && uv sync --python /usr/bin/python3 --no-dev --no-install-project

# 项目层：sed 静默失配时，由下面的断言拦截错误的 CUDA 变体
COPY fireredasr2s ./fireredasr2s
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --python /usr/bin/python3 --no-dev --no-editable \
    && .venv/bin/python -c "import torch, torchaudio; assert torch.version.cuda == '12.6', torch.version.cuda; assert '+cu126' in torch.__version__, torch.__version__; assert '+cu126' in torchaudio.__version__, torchaudio.__version__"

COPY configs/fireredppl_server.yaml ./configs/fireredppl_server.yaml

EXPOSE 12345
ENTRYPOINT ["/opt/FireRedASR2S/.venv/bin/fireredppl-server", "--config", "/opt/FireRedASR2S/configs/fireredppl_server.yaml"]
# 模型不打进镜像，运行时挂载：-v /path/to/FireRedASR2-AED:/opt/FireRedASR2-AED
CMD ["--model-dir", "/opt/FireRedASR2-AED"]
