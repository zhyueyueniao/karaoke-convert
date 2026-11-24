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

### 🗣️ ASR 语音识别（SenseVoice）

上传音频 → 纯语音识别，输出文本 + 带时间戳分段（**不生成歌词，独立于歌词管线**）。

- **阿里 SenseVoice-Small**（sherpa-onnx int8，CPU 推理）：非自回归 NAR 架构，速度约为 Whisper 的 5~15 倍
- 支持中/粤/英/日/韩，带 ITN 标点；token 级时间戳按句末标点聚段
- API：`POST /asr/sync`（同步，短音频低延迟，推荐给外部语音输入）/ `POST /submit/asr`（轮询）/ `POST /stream/asr`（SSE，done 含 `text`/`ts_text`）
- 音频由 ffmpeg 统一转 16k 单声道，webm/opus、mp3、m4a、wav 均可直传，调用方零转码

### 🎯 歌词生成对齐

上传音频 → 自动生成逐词 LRC 歌词。

- **无歌词**：Whisper 自动听写生成歌词
- **有歌词**：强制对齐已有歌词文本（支持纯文本或带旧时间戳的 LRC），两个对齐后端可选：
  - `stable-ts`（默认）：Whisper 识别 + DTW 锚定
  - `qwen3`：**Qwen3-ForcedAligner-0.6B**（阿里 Qwen3-ASR 家族），非自回归专用对齐模型，文本↔音频直接对齐不经识别环节，原生支持中文/粤语；加载/推理失败自动回退 stable-ts

### 📡 REST API

异步任务模式 + SSE 流式推送，支持外部调用：

| 模式 | 端点 | 说明 |
|------|------|------|
| 轮询 | `POST /submit/karaoke` | 一键处理（可选 lyrics 参数） |
| 轮询 | `POST /submit/separate` | 人声伴奏分离 |
| 轮询 | `POST /submit/align` | 歌词生成/对齐（无歌词听写，有歌词对齐） |
| 轮询 | `POST /submit/mv2ktv` | MV 转 KTV 视频 |
| 轮询 | `POST /submit/asr` | ASR 语音识别（SenseVoice 纯转写） |
| 轮询 | `GET /status/{job_id}` | 查询任务状态 |
| 轮询 | `GET /result/{job_id}/{filename}` | 下载结果文件 |
| SSE | `POST /stream/karaoke` | 一键处理（实时进度推送） |
| SSE | `POST /stream/separate` | 人声分离（实时进度推送） |
| SSE | `POST /stream/align` | 歌词生成/对齐（实时进度推送） |
| SSE | `POST /stream/mv2ktv` | MV 转 KTV（实时进度推送） |
| SSE | `POST /stream/asr` | ASR 语音识别（done 含 text/ts_text） |
| 同步 | `POST /asr/sync` | ASR 短音频一次请求返回文本（低延迟语音输入，见下） |

#### 同步 ASR：`POST /asr/sync`

给「按住说话」这类短音频用：一次 HTTP 请求直接拿到文本，不建 job、不轮询、不解析 SSE，自带上传临时文件清理。

```bash
curl -s -F 'file=@speech.webm' http://<IP>:7861/asr/sync
# {"text":"全押","ts_text":"[00:00.00] 全押","elapsed_ms":842}
```

| 项 | 说明 |
|---|---|
| 入参 | multipart/form-data `file`（任意 ffmpeg 可解码音频）；可选头 `X-ASR-Token` |
| 成功 | `{"text","ts_text","elapsed_ms"}`；`elapsed_ms` 可用于核对端到端延迟 |
| 错误 | 400 空文件/无语音、401 token 无效、413 超限、429 并发已满、503 模型未就绪/加载超时、500 解码或推理失败 |
| 环境变量 | `ASR_SYNC_MAX_BYTES`（默认 10MB）、`ASR_SYNC_TOKEN`（默认空 = 不鉴权）、`ASR_SYNC_READY_TIMEOUT`（默认 15s，只约束等待模型加载）、并发固定 2 |

> 仅 `/asr/sync` 带大小/并发/token 限制并清理上传临时文件；`/submit/asr` 与 `/stream/asr` 保持原有行为。7861 为内网端口，不要暴露公网，外部调用请经鉴权代理。

### ⚙️ 模型管理

- **Whisper**：tiny / base / small / medium / large-v3（下载、切换、删除）
- **Demucs**：mdx_extra / mdx_q / htdemucs / htdemucs_ft / mdx（下载、切换、删除）
- **对齐模型（阿里 Qwen3）**：Qwen/Qwen3-ForcedAligner-0.6B（下载、删除；`ALIGN_BACKEND=qwen3` 时使用）
- **ASR 识别模型（阿里 SenseVoice）**：csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue（仅下载 int8 模型 + 词表，~230MB）

## 技术栈

- **人声分离**：[Demucs](https://github.com/facebookresearch/demucs)（Hybrid Transformer 音源分离）；Intel GPU 走 **OpenVINO hybrid**（`ov_demucs.py`：torch CPU 前后处理 + OV GPU 两分支主干，fp16 默认）
- **语音识别/对齐**：[faster-whisper](https://github.com/SYSTRAN/faster-whisper)（Whisper CTranslate2 加速版）
- **独立 ASR 语音识别**：[SenseVoice-Small](https://github.com/FunAudioLLM/SenseVoice)（阿里 FunAudioLLM）+ [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)（int8 CPU 推理）
- **歌词对齐**：[stable-ts](https://github.com/jianfch/stable-ts)（CTC 强制对齐，支持逐词时间戳）；可选 [Qwen3-ForcedAligner](https://github.com/QwenLM/Qwen3-ASR)（阿里 Qwen3-ASR 家族，`ALIGN_BACKEND=qwen3` 启用）
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
| Intel GPU (Arc / 核显) | **OpenVINO hybrid（fp16 默认）** | CPU (int8) | ✅ 已集成 |
| NVIDIA GPU | CUDA (float16) | CUDA (float16) | ❌ 镜像为 CPU-only torch 精简版，需另建 CUDA 变体 |
| 纯 CPU | CPU (float32) | CPU (int8) | ✅ 已集成 |

**Demucs OpenVINO hybrid 管线**（`DEMUCS_DEVICE=openvino` 时启用，`ov_demucs.py`）：
torch CPU 负责 STFT/归一化/mask/ISTFT（调用 demucs 官方 `_spec/_mask/_ispec`，语义逐位一致），
OpenVINO GPU 只跑两分支主干（htdemucs_bb.xml，首次运行自动从 torch 权重编译并序列化到 `/app/output/`）。
实测 Core Ultra 225H iGPU：**fp16 约 0.45s/段**（旧 OV CPU 全模型 1.5s/段的 3.3 倍，rel_err 0.43% 无 NaN），
fp32 约 0.98s/段（数值完美）。397 秒歌曲端到端（解码+分离+编码）约 49 秒。

- **精度切换**：写 `/app/output/ov_demucs_precision.txt` 内容为 `fp16`（默认）或 `fp32`，改完即时生效无需重启
- **模型支持**：仅 HTDemucs 架构（`htdemucs` / `htdemucs_ft`，跟随前端模型切换，torch 实例与 GPU 主干产物按模型各缓存一份）；`mdx_extra` / `mdx_q` / `mdx` 是经典 Demucs（无时间分支），选择后**自动回退 demucs CLI（CPU）**并在 UI 中标注
- 主干模型缺失时自愈重建（需容器内有 torch + demucs 权重缓存，均已具备）
- 旧 adowu/htdemucs-onnx 整模型方案已废弃：其 ISTFT 在图内（GPU 上占 80% 耗时）且 fp16 产生 NaN

镜像内 torch 为 **CPU-only wheel**（不含 CUDA 与 nvidia-* 运行时，省 ~4-5GB），目标硬件为 Intel iGPU/CPU。torch 在 hybrid 管线中承担 CPU 前后处理，保留完整版。`config.py` 的 `detect_best_devices()` 仍会探测 NVIDIA GPU，代码骨架保留；如需 CUDA 加速，需另建变体镜像：`dockerfile` 中把 torch 换回 CUDA 版（`pip install torch --index-url https://download.pytorch.org/whl/cu124`），并在 `docker-compose.yaml` 中配置 GPU 设备映射。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `KARAOKE_FORCE_DEVICE` | 自动探测 | 强制加速模式：cuda / openvino / cpu |
| `KARAOKE_API` | `1` | 设为 `0` 禁用 REST API |
| `API_PORT` | `7861` | API 端口（容器内部） |
| `HF_ENDPOINT` | `hf-mirror.com` | HuggingFace 镜像（国内加速） |
| `DEMUCS_MODEL` | `mdx_extra` | 默认 Demucs 模型 |
| `WHISPER_MODEL` | `medium` | 默认 Whisper 模型 |
| `ALIGN_BACKEND` | `stable-ts` | 歌词对齐后端：`stable-ts` / `qwen3`（阿里 Qwen3-ForcedAligner，失败自动回退）；**前端 ⚙️ 模型管理 可运行时切换**（选择持久化到 `output/.align_backend_pref`，优先于环境变量） |
| `ALIGNER_MODEL` | `Qwen/Qwen3-ForcedAligner-0.6B` | 对齐模型 HF/ModelScope 仓库名或本地目录 |
| `SENSEVOICE_MODEL` | `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` | ASR 识别模型 HF 仓库名或本地目录 |

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
├── ov_demucs.py        # Demucs OpenVINO GPU hybrid 推理（DEMUCS_DEVICE=openvino 时替代 demucs CLI）
├── video_processing.py # MV 转 KTV 视频处理 / LRC 转 ASS 字幕
├── qwen_aligner.py     # Qwen3-ForcedAligner 对齐后端（ALIGN_BACKEND=qwen3）
├── sensevoice_asr.py   # SenseVoice 纯 ASR 语音识别（独立于歌词管线）
├── model_loader.py     # 模型加载 + 切换
├── model_manager.py    # 模型下载 / 删除 / 状态
├── docker-compose.yaml # Docker 编排
├── dockerfile          # 镜像构建
└── README.md
```
