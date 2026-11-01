"""全局配置、硬件探测、常量"""
import os
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
# 语音识别/对齐的目标语言（stable_whisper 的 align() 强制要求显式传 language）
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "zh")

# ---- 歌词对齐后端 ----
# stable-ts（默认）: whisper 识别 + DTW 锚定，锚点质量受歌声识别精度制约
# qwen3: Qwen3-ForcedAligner-0.6B（非自回归专用对齐模型），文本↔音频直接对齐，
#        不经识别环节，原生支持中文/粤语；加载失败自动回退 stable-ts
# 运行时切换持久化：前端下拉优先（写 output/.align_backend_pref），环境变量仅作初始默认
ALIGN_BACKEND = _load_model_pref("ALIGN_BACKEND", "stable-ts").strip().lower()
# Aligner 模型：HF/ModelScope 仓库名或本地目录（from_pretrained 自动下载到 HF_CACHE）
ALIGNER_MODEL = os.environ.get("ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")

# ---- ASR 语音识别（SenseVoice，独立于歌词生成/对齐管线） ----
# sherpa-onnx SenseVoice int8（阿里 FunAudioLLM SenseVoice-Small 的 onnx 移植），
# 非 NAR 自回归架构，CPU int8 极快；仅下载 model.int8.onnx + tokens.txt（~230MB）
SENSEVOICE_MODEL = os.environ.get(
    "SENSEVOICE_MODEL", "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")

AVAILABLE_WHISPER = ["tiny", "base", "small", "medium", "large-v3"]
AVAILABLE_DEMUCS  = ["mdx_extra", "mdx_q", "htdemucs", "htdemucs_ft", "mdx"]

HF_CACHE = "/root/.cache/huggingface"
# 统一 HuggingFace 缓存目录：huggingface_hub 默认按 HF_HOME/HF_HUB_CACHE 环境变量决定
# 缓存位置（hub>=1.0 时默认是 $HF_HOME/hub，而非 HF_HOME 本身），而 model_manager 下载时
# 显式使用 HF_CACHE。二者不一致会导致“模型管理 tab 显示已下载，但处理接口却报模型不存在
# /一直 等待 Whisper 下载”。这里强制 HF_HOME 与 HF_HUB_CACHE 都指向 HF_CACHE，
# 使模型加载与下载（含兜底下载）走同一个目录。
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = HF_CACHE
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
TORCH_CHECKPOINTS = "/root/.cache/torch/hub/checkpoints"  # 所有 Demucs 模型 torch hub 缓存目录
TORCH_MARKERS = "/root/.cache/torch/.demucs_markers"  # 所有 Demucs 模型 torch hub 缓存标记文件

WHISPER_SIZES = {"tiny": "75MB", "base": "141MB", "small": "464MB", "medium": "1.4GB", "large-v3": "~3.1GB"}
DEMUCS_SIZES  = {"mdx_extra": "639MB", "mdx_q": "199MB", "htdemucs": "80MB","htdemucs_ft": "321MB", "mdx": "695MB"}
