# 🎤 卡拉OK智能工作站

一键 MV 转 KTV / 人声伴奏分离 / 歌词生成对齐 / REST API，支持 Intel GPU (OpenVINO) 加速（CUDA 代码骨架已预留）。

## 功能

### 🎬 一键 MV 转 KTV

上传 MV 视频 → 提取音频 → 分离人声+伴奏 → 生成逐词歌词 → 烧录卡拉OK字幕 → **多音轨 KTV 视频**。

- 输出视频含双音轨：音轨1 = 原唱（人声+伴奏），音轨2 = 纯伴奏，播放器可切换
- 歌词来源：填写歌词文本走 CTC 对齐（更精准），留空则自动听写
- 字幕样式可选（经典KTV / 清新白 / 文艺红 / 暗夜蓝 / 高级灰等）

### 🚀 MP3 一键提取伴奏歌词

上传完整歌曲 → 自动分离人声+伴奏 → 自动听写生成 LRC 歌词。

### 🎵 单独分离人声伴奏

上传音频 → 分离人声 + 伴奏（基于 Demucs 模型）。

### 🎯 歌词生成对齐

上传音频 → 自动生成逐词 LRC 歌词。

- **无歌词**：Whisper 自动听写生成歌词
- **有歌词**：CTC 强制对齐已有歌词文本（支持纯文本或带旧时间戳的 LRC）

### 📡 REST API

异步任务模式 + SSE 流式推送，支持外部调用：

| 模式 | 端点 | 说明 |
|------|------|------|
| 轮询 | `POST /submit/karaoke` | 一键处理（可选 lyrics 参数） |
| 轮询 | `POST /submit/separate` | 人声伴奏分离 |
| 轮询 | `POST /submit/align` | 歌词生成/对齐（无歌词听写，有歌词对齐） |
| 轮询 | `POST /submit/mv2ktv` | MV 转 KTV 视频 |
| 轮询 | `GET /status/{job_id}` | 查询任务状态 |
| 轮询 | `GET /result/{job_id}/{filename}` | 下载结果文件 |
| SSE | `POST /stream/karaoke` | 一键处理（实时进度推送） |
| SSE | `POST /stream/separate` | 人声分离（实时进度推送） |
| SSE | `POST /stream/align` | 歌词生成/对齐（实时进度推送） |
| SSE | `POST /stream/mv2ktv` | MV 转 KTV（实时进度推送） |

### ⚙️ 模型管理

- **Whisper**：tiny / base / small / medium / large-v3（下载、切换、删除）
- **Demucs**：mdx_extra / mdx_q / htdemucs / htdemucs_ft / mdx（下载、切换、删除）

## 技术栈

- **人声分离**：[Demucs](https://github.com/facebookresearch/demucs)（Hybrid Transformer 音源分离）
- **语音识别/对齐**：[faster-whisper](https://github.com/SYSTRAN/faster-whisper)（Whisper CTranslate2 加速版）
- **歌词对齐**：[stable-ts](https://github.com/jianfch/stable-ts)（CTC 强制对齐，支持逐词时间戳）
- **前端 UI**：[Gradio](https://gradio.app/)（>=4.12）
- **API 服务**：[FastAPI](https://fastapi.tiangolo.com/) + Uvicorn
- **视频处理**：FFmpeg
- **GPU 加速**：NVIDIA CUDA / Intel OpenVINO（自动探测优先级）

## 运行

### Docker 部署（推荐）

```bash
# 1. 构建镜像（首次或 dockerfile 有变更时）
docker compose build

# 2. 启动服务
docker compose up -d
```

默认端口：
- `7860` — Gradio Web UI（`http://<IP>:7860`）
- `7861` — REST API 文档（`http://<IP>:7861/docs`）

### 宿主机直跑

```bash
# 依赖
pip install demucs diffq onnxruntime-openvino openvino \
            faster-whisper ctranslate2 stable-ts \
            gradio fastapi uvicorn python-multipart \
            ffmpeg-python

# 启动
python app.py
```

## GPU 加速

| 模式 | Demucs | Whisper | Docker 默认 |
|------|--------|---------|------------|
| Intel GPU (Arc / UHD) | OpenVINO (int8) | CPU (int8) | ✅ 已集成 |
| NVIDIA GPU | CUDA (float16) | CUDA (float16) | ⚠️ 代码骨架已预留 |
| 纯 CPU | CPU (float32) | CPU (int8) | ✅ 已集成 |

`config.py` 的 `detect_best_devices()` 会自动探测 NVIDIA GPU，但当前 Docker 镜像只安装了 Intel + CPU 依赖。如需启用 CUDA，需在 `dockerfile` 中额外安装 NVIDIA CUDA 版依赖（`torch` + `ctranslate2` 等），并在 `docker-compose.yaml` 中配置 GPU 设备映射。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `KARAOKE_FORCE_DEVICE` | 自动探测 | 强制加速模式：cuda / openvino / cpu |
| `KARAOKE_API` | `1` | 设为 `0` 禁用 REST API |
| `API_PORT` | `7861` | API 端口（容器内部） |
| `HF_ENDPOINT` | `hf-mirror.com` | HuggingFace 镜像（国内加速） |
| `DEMUCS_MODEL` | `mdx_extra` | 默认 Demucs 模型 |
| `WHISPER_MODEL` | `medium` | 默认 Whisper 模型 |

## 模型存储

模型首次使用会自动下载，默认缓存路径：

| 模型 | 路径 |
|------|------|
| HuggingFace | `/root/.cache/huggingface` |
| Torch Hub (Demucs) | `/root/.cache/torch` |

Docker 部署时建议挂载这些目录为持久化卷，避免重启重新下载。

## 项目结构

```
karaoke-convert/
├── app.py              # Gradio UI 主入口
├── api.py              # FastAPI REST API + SSE 流式
├── config.py           # 配置、硬件探测、模型选择
├── processing.py       # 人声分离 / 歌词听写 / CTC 对齐 / 一键处理
├── video_processing.py # MV 转 KTV 视频处理 / LRC 转 ASS 字幕
├── model_loader.py     # 模型加载 + 切换
├── model_manager.py    # 模型下载 / 删除 / 状态
├── docker-compose.yaml # Docker 编排
├── dockerfile          # 镜像构建
└── README.md
```
