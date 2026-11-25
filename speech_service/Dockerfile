# 语音 sidecar：Qwen3-ASR（高精度识别）+ Qwen3-TTS（语音合成/克隆）独立容器
# 基于主服务镜像（复用 ffmpeg / 系统依赖），语音依赖装入独立 venv /opt/speech-venv，
# 与主服务 /opt/venv 完全隔离（依赖隔离 + compose 内存限额实现内存硬隔离）
FROM karaoke-tool-ultra255:latest

# sox 系统工具：qwen-tts / qwen-asr 依赖的 python sox 包需要系统 sox 命令
RUN apt-get update && apt-get install -y --no-install-recommends \
    sox libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# 独立 venv（不污染主服务 /opt/venv）
RUN python3 -m venv /opt/speech-venv

# torch/torchaudio CPU 轮子（download.pytorch.org CPU 官方源。
# ⚠️ NAS 实测：基础镜像 Python 较新（无 torch 2.6 的轮子），必须用 2.9.1；
#    阿里云 pytorch-wheels CPU 镜像最高只有 2.6.0 不可用；此源较慢约需 25 分钟，
#    但为已验证可行的唯一 CPU 轮子来源）
RUN /opt/speech-venv/bin/pip install --no-cache-dir torch==2.9.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 公共依赖。
# ⚠️ 版本冲突处理：qwen-tts 钉 transformers==4.57.3，qwen-asr 钉 transformers==4.57.6，
# 两个包在同一 venv 内 pip 解析必然失败 → 双双 --no-deps 安装，依赖统一取 transformers==4.57.6
# （gradio 故意不装：依赖树巨大，qwen 包仅其 demo 用到；运行时若报缺可再补）
RUN /opt/speech-venv/bin/pip install --no-cache-dir \
    "openvino>=2025.4.0" "transformers==4.57.6" "accelerate==1.12.0" \
    huggingface_hub numpy soundfile librosa modelscope \
    fastapi uvicorn python-multipart einops onnxruntime \
    qwen-omni-utils nagisa==0.2.11 soynlp==0.0.493 pytz flask sox nncf \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN /opt/speech-venv/bin/pip install --no-cache-dir --no-deps qwen-tts qwen-asr \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 代码目录：compose 会将 ./speech_service 热挂载到 /app/speech（改代码 restart 即可，无需重建镜像）
WORKDIR /app
COPY qwen_3_tts_helper.py qwen_3_asr_helper.py server.py /app/speech/
COPY static/ /app/speech/static/

EXPOSE 7862
CMD ["/opt/speech-venv/bin/python", "-u", "/app/speech/server.py"]
