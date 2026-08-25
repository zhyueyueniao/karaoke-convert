"""全局配置、硬件探测、常量"""
import os
import json
import shutil
import subprocess

# 输出目录（人声、伴奏、歌词等所有产出文件）
OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 预导入 OpenVINO（Intel GPU 加速运行时）
try:
    import openvino as ov
    _HAS_OPENVINO = True
except ImportError:
    ov = None  # type: ignore
    _HAS_OPENVINO = False


def detect_best_devices():
    """[硬件探测] 启动时检测可用 GPU，返回 (demucs_mode, whisper_device, whisper_precision)
    
    优先级：NVIDIA CUDA > Intel OpenVINO > 纯 CPU
    可通过环境变量 KARAOKE_FORCE_DEVICE 强制指定加速模式
    """
    force = os.environ.get("KARAOKE_FORCE_DEVICE", "").lower().strip()
    if force in ("cuda", "openvino", "cpu"):
        print(f"⚙️  用户强制指定加速模式: {force}")
        if force == "cuda":
            return "cuda", "cuda", "float16"
        elif force == "openvino":
            return "openvino", "cpu", "int8"
        else:
            return "", "cpu", "int8"

    has_nvidia = False
    has_intel_gpu = False

    print("🔍 正在探测硬件加速...")

    # Intel GPU
    dri_nodes = [p for p in ["/dev/dri/renderD128", "/dev/dri/card0"] if os.path.exists(p)]
    if dri_nodes:
        print(f"   发现 DRI 节点: {dri_nodes}")
        if _HAS_OPENVINO:
            try:
                core = ov.Core()
                available = core.available_devices
                print(f"   OpenVINO available_devices: {available}")
                if "GPU" in available:
                    has_intel_gpu = True
            except Exception as e:
                print(f"   OpenVINO 探测失败: {e}")
    else:
        print("   未发现 /dev/dri 渲染节点")

    # NVIDIA GPU
    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, timeout=5, check=True)
            has_nvidia = True
            print("   nvidia-smi 探测到 NVIDIA GPU")
        except Exception as e:
            print(f"   nvidia-smi 探测失败: {e}")
    if not has_nvidia:
        try:
            import torch
            if torch.cuda.is_available():
                has_nvidia = True
                print(f"   torch.cuda 探测到 NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    # 优先级：NVIDIA > Intel GPU > CPU
    if has_nvidia:
        print("🖥️  加速模式: NVIDIA CUDA（Demucs + Whisper）")
        return "cuda", "cuda", "float16"
    elif has_intel_gpu:
        print("🖥️  加速模式: Intel OpenVINO (Demucs) + CPU (Whisper)")
        return "openvino", "cpu", "int8"
    else:
        print("🖥️  加速模式: CPU（Demucs + Whisper 均走 CPU）")
        return "", "cpu", "int8"


DEMUCS_DEVICE, WHISPER_DEVICE, WHISPER_COMPUTE = detect_best_devices()

# 运行时切换持久化：前端切换优先，环境变量仅作为初始默认值
def _load_model_pref(key: str, default: str) -> str:
    """读取模型选择，优先级：持久化文件 > 环境变量 > 默认值"""
    pref_file = os.path.join(OUTPUT_DIR, f".{key.lower()}_pref")
    if os.path.exists(pref_file):
        return open(pref_file).read().strip()
    return os.environ.get(key, default)

def _save_model_pref(key: str, value: str):
    """持久化模型选择到文件"""
    pref_file = os.path.join(OUTPUT_DIR, f".{key.lower()}_pref")
    with open(pref_file, "w") as f:
        f.write(value)

DEMUCS_MODEL = _load_model_pref("DEMUCS_MODEL", "mdx_extra")
WHISPER_MODEL = _load_model_pref("WHISPER_MODEL", "medium")

AVAILABLE_WHISPER = ["tiny", "base", "small", "medium", "large-v3"]
AVAILABLE_DEMUCS  = ["mdx_extra", "mdx_q", "htdemucs", "htdemucs_ft", "mdx"]

HF_CACHE = "/root/.cache/huggingface"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
TORCH_CHECKPOINTS = "/root/.cache/torch/hub/checkpoints"  # 所有 Demucs 模型 torch hub 缓存目录
TORCH_MARKERS = "/root/.cache/torch/.demucs_markers"  # 所有 Demucs 模型 torch hub 缓存标记文件

WHISPER_SIZES = {"tiny": "75MB", "base": "141MB", "small": "464MB", "medium": "1.4GB", "large-v3": "~3.1GB"}
DEMUCS_SIZES  = {"mdx_extra": "639MB", "mdx_q": "199MB", "htdemucs": "80MB","htdemucs_ft": "321MB", "mdx": "695MB"}


# ============================================================
#  AI MV 生成（根据歌词意境自动生成视频）
# ============================================================

# OpenVINO 模型缓存根目录（LTX-Video / SD 转换后的 IR + GPU kernel 缓存，挂载到 ./model_cache/openvino）
OPENVIDEO_MODELS_DIR = "/root/.cache/openvino"

# ---- 三层可插拔视频生成后端降级链 ----
#  P0 外部视频生成 API（云端，质量最高） → P1 LTX-Video 真文生视频（本地 iGPU）→ P2 SD文生图+FFmpeg运动（本地兜底）
AVAILABLE_T2V_BACKENDS = ["api", "ltx-video", "sd-motion"]
T2V_BACKEND_PRIORITY = ["api", "ltx-video", "sd-motion"]
# 本地 OpenVINO GenAI 推理设备（iGPU 用 GPU，无 Intel GPU 时回退 CPU）
T2V_DEVICE = os.environ.get("T2V_DEVICE", "GPU")

# ---- 外部视频生成 API（写视频的后端，与写 prompt 的 LLM 是两套独立配置）----
# provider 支持: openai / kling / jimeng ；留空表示不启用 P0，直接进入本地降级链
EXT_VIDEO_API_PROVIDER = os.environ.get("EXT_VIDEO_API_PROVIDER", "")
EXT_VIDEO_API_KEY      = os.environ.get("EXT_VIDEO_API_KEY", "")
EXT_VIDEO_API_BASE_URL = os.environ.get("EXT_VIDEO_API_BASE_URL", "")
# P0 仅在同时配置了 provider + key 时可用
EXT_VIDEO_API_ENABLED = bool(EXT_VIDEO_API_PROVIDER and EXT_VIDEO_API_KEY)

# ---- 写 prompt 的 LLM API（可选；不配置则走规则关键词匹配场景库）----
PROMPT_LLM_PROVIDER = os.environ.get("PROMPT_LLM_PROVIDER", "")
PROMPT_LLM_API_KEY  = os.environ.get("PROMPT_LLM_API_KEY", "")
PROMPT_LLM_BASE_URL = os.environ.get("PROMPT_LLM_BASE_URL", "")
PROMPT_LLM_MODEL    = os.environ.get("PROMPT_LLM_MODEL", "")
PROMPT_LLM_ENABLED  = bool(PROMPT_LLM_PROVIDER and PROMPT_LLM_API_KEY)

# ---- 风格预设前缀库（统一所有片段视觉风格，保证连贯性）----
STYLE_PRESETS = {
    "cinematic":  "cinematic, dreamy atmosphere, soft pastel colors, film grain, consistent art style, highly detailed",
    "anime":      "anime style, vibrant colors, clean lines, cel shaded, consistent art style, highly detailed",
    "watercolor": "watercolor painting, soft brush strokes, paper texture, gentle pastel tones, consistent art style",
    "cyberpunk":  "cyberpunk, neon lights, futuristic city, high contrast, dark moody atmosphere, consistent art style",
    "vintage":    "vintage 1980s film, warm faded colors, grainy, nostalgic mood, consistent art style",
    "ink":        "Chinese ink wash painting, minimalist, black and white, elegant brushwork, consistent art style",
}
DEFAULT_STYLE = "cinematic"

# ---- 场景关键词库（无 LLM 时规则匹配：命中关键词 → 画面元素描述）----
SCENE_KEYWORDS = {
    "海":   "ocean waves crashing on shore, vast blue horizon, seagulls, golden sunset reflection on water",
    "海浪": "rolling ocean waves, sea foam, sandy beach, soft breeze",
    "山":   "majestic green mountains, misty peaks, pine forest, drifting clouds",
    "雪山": "snow-capped mountains, cold blue tone, clear sky, icy wilderness",
    "雨":   "rain falling on city streets, wet pavement reflections, neon glow, melancholic mood",
    "雪":   "falling snow, quiet winter landscape, bare trees, soft white silence",
    "夜":   "starry night sky, city lights twinkling, moonlight, peaceful darkness",
    "星":   "starry galaxy, milky way, glowing cosmos, vast silent space",
    "花":   "blooming flowers field, petals in gentle wind, warm sunlight, soft colors",
    "森林": "dense green forest, sunbeams through leaves, mossy ground, tranquil",
    "城市": "bustling modern city, skyscrapers, moving traffic, energetic atmosphere",
    "火":   "warm campfire flames, sparks rising, cozy night, orange glow",
    "风":   "wind blowing through tall grass, leaves swirling, open field, freedom",
    "爱":   "two silhouettes embracing, warm golden light, tender moment, romantic",
    "孤独": "lone figure walking empty street, long shadows, blue hour, solitude",
    "梦":   "surreal dreamscape, floating clouds, soft pastel gradient, ethereal light",
    "哭":   "teardrop on cheek, dim room, soft rain on window, sorrowful mood",
    "笑":   "bright smiling face, sunny meadow, laughter, warm joyful light",
    "离别": "train leaving station, waving goodbye, dusk, melancholy",
    "春":   "spring cherry blossoms, fresh green sprouts, gentle sunlight, renewal",
    "秋":   "autumn maple leaves falling, golden wheat field, crisp air, nostalgia",
    "夏":   "summer beach, bright sun, blue ocean, vivid colors, warmth",
    "冬":   "winter wonderland, frozen lake, bare branches, cold serenity",
}

# 未命中任何关键词时的兜底画面描述
SCENE_DEFAULT = "abstract atmospheric landscape, soft flowing light, gentle gradient, emotional mood"

# ---- LTX-Video 模型（需从 HF 源模型转换 IR；官方无现成 OpenVINO IR）----
LTXVIDEO_REPO = "Lightricks/LTX-Video"
LTXVIDEO_OV_DIR = os.path.join(OPENVIDEO_MODELS_DIR, "ltx-video-ov")

# ---- SD / SDXL 文生图模型（官方已提供 OpenVINO 转换 IR，开箱即用）----
SD15_REPO = "OpenVINO/stable-diffusion-v1-5-fp16-ov"
SD15_OV_DIR = os.path.join(OPENVIDEO_MODELS_DIR, "sd15-ov")
SDXL_REPO = "OpenVINO/stable-diffusion-xl-base-1.0-fp16-ov"
SDXL_OV_DIR = os.path.join(OPENVIDEO_MODELS_DIR, "sdxl-ov")

AVAILABLE_SD_MODELS = ["sd15", "sdxl"]
# 供 UI 模型管理 Tab 选择的生成模型（LTX-Video + SD/SDXL）
AVAILABLE_AIMV_MODELS = ["ltx-video", "sd15", "sdxl"]
DEFAULT_AIMV_MODEL = "sd15"

# 各生成模型在 HF 上的 repo（用于 model_manager 下载/转换）
AIMV_MODEL_REPOS = {
    "ltx-video": LTXVIDEO_REPO,
    "sd15": SD15_REPO,
    "sdxl": SDXL_REPO,
}
AIMV_MODEL_OV_DIRS = {
    "ltx-video": LTXVIDEO_OV_DIR,
    "sd15": SD15_OV_DIR,
    "sdxl": SDXL_OV_DIR,
}
# 模型大小预估值（用于 UI 展示）
AIMV_MODEL_SIZES = {
    "ltx-video": "~6GB（含转换）",
    "sd15": "1.6GB",
    "sdxl": "6.9GB",
}

# ---- 视频生成参数默认值 ----
AIMV_RESOLUTION = (512, 320)      # LTX-Video 低分辨率片段；SD+运动也沿用
AIMV_SEG_MIN = 8.0                # 单段最短时长（秒）
AIMV_SEG_MAX = 15.0               # 单段最长时长（秒）
AIMV_XFADE = 0.6                  # 段间交叉淡化时长（秒）
AIMV_LTX_CLIP_SEC = 3.0           # LTX-Video 单次生成片段秒数（后用 FFmpeg 延长到目标时长）
AIMV_TIMEOUT_PER_CLIP = 1800      # 单段生成超时（秒），LTX-Video 最慢


# ---- QSV 硬件编码探测（启动时一次，缓存结果；不可用时 fallback libx264）----
def detect_qsv() -> bool:
    """[硬件探测] 检测 FFmpeg 是否带 h264_qsv 编码器"""
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=20)
        return "h264_qsv" in out.stdout
    except Exception:
        return False

HAS_QSV = detect_qsv()
print(f"🎬 QSV 硬件编码: {'✅ 启用 (h264_qsv)' if HAS_QSV else '⚠️ 不可用，回退 libx264'}")
