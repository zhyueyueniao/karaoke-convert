# syntax=docker/dockerfile:1
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:$PATH"
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# ==================== Layer 1: 系统依赖 ====================
# 改动最少，利用 apt 缓存挂载，重复构建秒过
# （wget/gnupg 移到 Layer 2 同层装删——跨层删除文件不会减小镜像体积）
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv ffmpeg libgl1 libglib2.0-0 \
    libnuma1 libtbb12 gcc python3-dev \
    ca-certificates tini fonts-noto-cjk && \
    apt-get clean

# ==================== Layer 2: Intel GPU 驱动（全部从 Intel 官方仓库安装） ====================
# gcc/python3-dev 保留：diffq 为 sdist 分发，构建其 C++ 扩展需要编译工具链
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends wget gnupg && \
    mkdir -p /usr/share/keyrings && \
    wget -q --timeout=15 -O - https://repositories.intel.com/gpu/intel-graphics.key | \
      gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" > \
      /etc/apt/sources.list.d/intel-gpu.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    intel-opencl-icd intel-media-va-driver-non-free \
    intel-level-zero-gpu level-zero && \
    apt-get purge -y wget gnupg && apt-get autoremove -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ==================== Layer 3: Python 依赖 ====================
# torch/torchaudio 必须成对从 CPU 专用源安装（~250MB，不含 CUDA 与任何 nvidia-* 运行时，
# 相比默认 CUDA 版省 ~4-5GB）；后续包看到已满足依赖不会重复安装。
# 注意：torchaudio 必须同源——PyPI 上的 torchaudio wheel 是按 CUDA 版 torch 构建的，
# 与 CPU 版 torch 混装会在 import 时报 libcudart.so.x 缺失。
# 本镜像目标硬件为 Intel iGPU/CPU（无 NVIDIA），CUDA 模式不可用；
# 如需 NVIDIA 加速请另建变体镜像（换装 CUDA 版 torch + nvidia 运行时 + GPU 设备映射）。
# download.pytorch.org 国内较慢时可改用阿里镜像：
#   pip install torch -f https://mirrors.aliyun.com/pytorch-wheels/cpu/
# onnxruntime-openvino 已移除：全项目无 import onnxruntime（Demucs GPU 加速走 openvino 直连）
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv /opt/venv && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    "demucs>=4.0,<5.0" \
    "diffq" \
    "openvino~=2025.4" \
    "faster-whisper>=1.0,<2.0" \
    "ctranslate2>=4.0,<5.0" \
    "stable-ts>=2.18.0" \
    "gradio>=6.0,<7.0" \
    "fastapi>=0.110" "uvicorn>=0.29" "python-multipart>=0.0.9" \
    "qwen-asr" "sherpa-onnx" && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true && \
    find /opt/venv -type f -name "*.pyc" -delete 2>/dev/null; true

# ==================== Layer 4: 应用代码 ====================
# app.py 等通过 volume 热挂载，这一层只有文件拷贝，瞬间完成
WORKDIR /app
COPY app.py api.py config.py model_loader.py model_manager.py processing.py \
     qwen_aligner.py sensevoice_asr.py /app/
RUN mkdir -p /app/output /app/workspace

EXPOSE 7860
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/venv/bin/python3", "/app/app.py"]
