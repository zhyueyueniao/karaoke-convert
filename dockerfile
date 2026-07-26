# syntax=docker/dockerfile:1
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/venv/bin:$PATH"
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# ==================== Layer 1: 系统依赖 ====================
# 改动最少，利用 apt 缓存挂载，重复构建秒过
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv ffmpeg libgl1 libglib2.0-0 \
    libnuma1 libtbb12 gcc python3-dev \
    wget ca-certificates gnupg tini fonts-noto-cjk && \
    apt-get clean

# ==================== Layer 2: Intel GPU 驱动（全部从 Intel 官方仓库安装） ====================
RUN mkdir -p /usr/share/keyrings && \
    wget -q --timeout=15 -O - https://repositories.intel.com/gpu/intel-graphics.key | \
      gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" > \
      /etc/apt/sources.list.d/intel-gpu.list && \
    echo "✅ Intel GPU 仓库已添加" && \
    apt-get update && apt-get install -y --no-install-recommends \
    intel-opencl-icd intel-media-va-driver-non-free \
    intel-level-zero-gpu level-zero && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ==================== Layer 3: Python 依赖（OpenVINO 改用 pip 安装，省去 ~600MB tgz）====================
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv /app/venv && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir \
    "demucs>=4.0,<5.0" \
    "diffq" \
    "onnxruntime-openvino>=1.18,<2.0" \
    "openvino~=2025.4" \
    "faster-whisper>=1.0,<2.0" \
    "ctranslate2>=4.0,<5.0" \
    "stable-ts>=2.18.0" \
    "gradio>=6.0,<7.0" \
    "fastapi>=0.110" "uvicorn>=0.29" "python-multipart>=0.0.9" && \
    find /app/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true && \
    find /app/venv -type f -name "*.pyc" -delete 2>/dev/null; true

# ==================== Layer 4: 应用代码 ====================
# app.py 等通过 volume 热挂载，这一层只有文件拷贝，瞬间完成
WORKDIR /app
COPY app.py api.py config.py model_loader.py processing.py model_manager.py /app/
RUN mkdir -p /app/output /app/workspace

EXPOSE 7860
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/venv/bin/python3", "/app/app.py"]
