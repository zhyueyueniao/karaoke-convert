"""Qwen3 语音 sidecar：ASR（高精度识别）+ TTS（语音合成/音色克隆）独立容器

独立于主服务的 FastAPI 服务，模型惰性后台加载：
  - ASR / TTS(CustomVoice) 启动即预载（两模型 fp16 常驻约 8-9GB）
  - TTS Base（音色克隆用）惰性加载：首次克隆请求时才加载（约 3.8GB），平时不占内存

端点：
  GET  /health           健康检查（设备/三模型就绪状态）
  GET  /voices           TTS 内置音色列表（9 音色，含中文描述）
  GET  /models/status    三模型的下载/转换/就绪状态与磁盘占用
  POST /models/download  SSE 流式：modelscope 下载权重 → OpenVINO IR 转换（kind=tts|tts-base|asr, force）
  POST /models/delete    删除已下载模型并卸载内存实例（JSON: {"kind": "tts"|"tts-base"|"asr"}）
  POST /tts/synthesize   JSON {"text","speaker","language","instruct"} → audio/wav（长文本自动按句切分）
  POST /tts/clone        multipart 上传参考音频 + text(+ref_text) → audio/wav（音色克隆，Base 模型）
  POST /asr/transcribe   音频上传 → {"text","language","elapsed_ms"}（ffmpeg 转 16k 后 Qwen3-ASR 识别）
  GET  /                 内置网页（合成/克隆/识别，麦克风录音输入）
  POST /voices/custom    保存参考音频为自定义克隆音色（name, ref_text 可选）
  DELETE /voices/custom/{vid}          删除自定义音色
  GET  /voices/custom/{vid}/ref.wav    试听自定义音色的参考音频

克隆说明：仅 Base 模型支持（generate_voice_clone）；ICL 模式需 ref_audio+ref_text（效果最好），
x-vector 模式仅需 ref_audio（x_vector_only=true）。参考音频建议 ≥5s 干净人声。
克隆音色可保存为自定义音色（POST /voices/custom 或 /tts/clone 的 save_as 参数），
持久化在 /models/voices/，出现在 GET /voices 列表，可直接当 speaker 用于 /tts/synthesize。

自动下载：首次调用合成/克隆/识别时若模型未下载，自动触发后台「modelscope 下载→OV 转换→加载」
全链路（SPEECH_AUTO_DOWNLOAD=0 关闭）；请求等待就绪（默认最长 30-40 分钟），超时返回进度提示，
后台继续不中断；同 kind 互斥，手动 /models/download 与自动下载不会重复下载。

页面：GET / 提供内置网页（合成/克隆/识别，支持麦克风录音输入）。

鉴权：设置 SPEECH_TOKEN 后除 /health 外均需请求头 X-Speech-Token。
设备：SPEECH_DEVICE=CPU|GPU|NPU，留空自动探测（GPU 优先，失败回退 CPU）。
"""
import io
import os
import re
import sys
import queue
import json
import time
import shutil
import threading
import subprocess

import numpy as np
import soundfile as sf
import openvino as ov
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse, Response, FileResponse
import uvicorn

# ======================== 配置 ========================
MODELS_ROOT = os.environ.get("SPEECH_MODELS_DIR", "/models")
TTS_MODEL_ID = os.environ.get("SPEECH_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
TTS_BASE_MODEL_ID = os.environ.get("SPEECH_TTS_BASE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B")
ASR_MODEL_ID = os.environ.get("SPEECH_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
TTS_DEFAULT_SPEAKER = os.environ.get("TTS_DEFAULT_SPEAKER", "Serena")
SPEECH_TOKEN = os.environ.get("SPEECH_TOKEN", "").strip()
SPEECH_DEVICE = os.environ.get("SPEECH_DEVICE", "").strip().upper()  # 空 = 自动探测
TTS_LOAD_TIMEOUT = float(os.environ.get("TTS_LOAD_TIMEOUT", "2700"))  # 等待模型就绪秒数（自动下载串行转换时首请求会等待较久）
TTS_BASE_LOAD_TIMEOUT = float(os.environ.get("TTS_BASE_LOAD_TIMEOUT", "3000"))  # Base 惰性加载（含首次下载）
TTS_BASE_IDLE_UNLOAD = float(os.environ.get("TTS_BASE_IDLE_UNLOAD", "30"))  # Base 空闲卸载阈值（分钟，0=常驻）
# TTS int8 权重压缩（SPEECH_TTS_INT8=0 关闭）：加载前对 fp16 IR 做 nncf int8 压缩（原地覆盖+标记）。
# 逐 token 自回归生成受内存带宽限制，int8 预计提速 1.5-2 倍；只压 talker 语言模型（耗时大头），
# 首次加载多花 1-3 分钟做一次性压缩，之后直接用压缩产物
TTS_INT8 = os.environ.get("SPEECH_TTS_INT8", "1") == "1"
ASR_LOAD_TIMEOUT = float(os.environ.get("ASR_LOAD_TIMEOUT", "2700"))
# 首次调用自动下载：模型未下载时，首个请求触发后台「下载→转换→加载」全链路，
# 请求本身等待就绪；超时则返回进度提示，后台继续不中断，后续请求 succeeds。
# 设 SPEECH_AUTO_DOWNLOAD=0 关闭（恢复为报错提示手动下载）。
AUTO_DOWNLOAD = os.environ.get("SPEECH_AUTO_DOWNLOAD", "1") == "1"
UPLOAD_MAX_BYTES = int(os.environ.get("SPEECH_UPLOAD_MAX_BYTES", str(20 * 1024 * 1024)))

TTS_DIR = os.path.join(MODELS_ROOT, "qwen3-tts")
ASR_DIR = os.path.join(MODELS_ROOT, "qwen3-asr")
TTS_CKPT = os.path.join(TTS_DIR, "ckpt")   # modelscope 下载的原始权重
TTS_OV = os.path.join(TTS_DIR, "ov")       # OpenVINO IR（推理用）
ASR_CKPT = os.path.join(ASR_DIR, "ckpt")
ASR_OV = os.path.join(ASR_DIR, "ov")
# 克隆用 Base 模型（独立目录，与 CustomVoice 互不影响）
TTS_BASE_DIR = os.path.join(MODELS_ROOT, "qwen3-tts-base")
TTS_BASE_CKPT = os.path.join(TTS_BASE_DIR, "ckpt")
TTS_BASE_OV = os.path.join(TTS_BASE_DIR, "ov")

# kind → (模型ID, ckpt 目录, ov 目录)
MODEL_KINDS = {
    "tts": (TTS_MODEL_ID, TTS_CKPT, TTS_OV),
    "tts-base": (TTS_BASE_MODEL_ID, TTS_BASE_CKPT, TTS_BASE_OV),
    "asr": (ASR_MODEL_ID, ASR_CKPT, ASR_OV),
}

# 自定义（克隆）音色持久化目录：每个音色一个子目录（meta.json + ref.wav 24k 单声道）
VOICES_DIR = os.path.join(MODELS_ROOT, "voices")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# 自动下载状态（kind → 进度描述 / "done" / "error: ..."）与同 kind 互斥锁（防并发重复下载）
_dl_state: dict[str, str] = {"tts": "", "tts-base": "", "asr": ""}
_dl_locks: dict[str, threading.Lock] = {k: threading.Lock() for k in MODEL_KINDS}
# 各模型内存实例的 error 读取器（_wait_or_detail 用）
_MODEL_ERR = {"tts": lambda: _tts_error, "tts-base": lambda: _tts_base_error, "asr": lambda: _asr_error}

# 转换全局串行锁：两个 fp16 转换进程并行会超 16G 内存限制被 cgroup OOM 杀死，
# 产物被截断（实测：xml 写成 0 字节 / decoder 文件缺失）。串行转换后峰值减半。
_CONV_LOCK = threading.Lock()
# 加载链全局锁：转换期间其它模型若已常驻内存（约 4GB/个），转换峰值 + 常驻仍会超限
# （实测 TTS speech tokenizer decoder 阶段再次被杀）。锁住「转换+加载」全程串行：
# TTS 转换时内存中无任何常驻模型，完成后释放再轮到 ASR。
_LOAD_CHAIN = threading.Lock()

# 各 kind 转换完成后必须存在且非 0 字节的核心产物（相对 ov 目录；损坏缓存靠它兜底重转）
_EXPECTED_IR = {
    "tts": ["openvino_talker_language_model.xml", "openvino_talker_language_model.bin",
            "openvino_talker_code_predictor_model.xml", "openvino_talker_code_predictor_model.bin",
            "openvino_talker_code_predictor_embedding_model.xml", "openvino_talker_code_predictor_embedding_model.bin",
            "openvino_talker_embedding_model.xml", "openvino_talker_embedding_model.bin",
            "openvino_talker_text_embedding_model.xml", "openvino_talker_text_embedding_model.bin",
            "openvino_talker_text_projection_model.xml", "openvino_talker_text_projection_model.bin",
            "speech_tokenizer/openvino_speech_tokenizer_encoder_model.xml",
            "speech_tokenizer/openvino_speech_tokenizer_encoder_model.bin",
            "speech_tokenizer/openvino_speech_tokenizer_decoder_model.xml",
            "speech_tokenizer/openvino_speech_tokenizer_decoder_model.bin"],
    "tts-base": ["openvino_talker_language_model.xml", "openvino_talker_language_model.bin",
                 "openvino_talker_code_predictor_model.xml", "openvino_talker_code_predictor_model.bin",
                 "openvino_talker_code_predictor_embedding_model.xml", "openvino_talker_code_predictor_embedding_model.bin",
                 "openvino_talker_embedding_model.xml", "openvino_talker_embedding_model.bin",
                 "openvino_talker_text_embedding_model.xml", "openvino_talker_text_embedding_model.bin",
                 "openvino_talker_text_projection_model.xml", "openvino_talker_text_projection_model.bin",
                 "speech_tokenizer/openvino_speech_tokenizer_encoder_model.xml",
                 "speech_tokenizer/openvino_speech_tokenizer_encoder_model.bin",
                 "speech_tokenizer/openvino_speech_tokenizer_decoder_model.xml",
                 "speech_tokenizer/openvino_speech_tokenizer_decoder_model.bin"],
    "asr": ["thinker/openvino_thinker_language_model.xml", "thinker/openvino_thinker_language_model.bin",
            "thinker/openvino_thinker_audio_model.xml", "thinker/openvino_thinker_audio_model.bin",
            "thinker/openvino_thinker_audio_encoder_model.xml", "thinker/openvino_thinker_audio_encoder_model.bin",
            "thinker/openvino_thinker_embedding_model.xml", "thinker/openvino_thinker_embedding_model.bin"],
}

# Qwen3-TTS-12Hz-1.7B-CustomVoice 官方 9 内置音色（无克隆功能）
VOICES = [
    {"id": "Vivian",   "desc": "明亮女声（中文）"},
    {"id": "Serena",   "desc": "温柔女声（中文）"},
    {"id": "Uncle_Fu", "desc": "浑厚男声（中文）"},
    {"id": "Dylan",    "desc": "北京口音男声（中文）"},
    {"id": "Eric",     "desc": "成都口音男声（中文）"},
    {"id": "Ryan",     "desc": "严谨男声（英文）"},
    {"id": "Aiden",    "desc": "热情男声（英文）"},
    {"id": "Ono_Anna", "desc": "温柔女声（日语）"},
    {"id": "Sohee",    "desc": "活泼女声（韩语）"},
]


def _pick_device() -> str:
    """OpenVINO 设备选择：环境变量优先，否则自动探测 GPU，失败回退 CPU"""
    if SPEECH_DEVICE in ("CPU", "GPU", "NPU"):
        return SPEECH_DEVICE
    try:
        return "GPU" if "GPU" in ov.Core().available_devices else "CPU"
    except Exception as e:
        print(f"⚠️ OpenVINO 设备探测失败，回退 CPU: {e}", flush=True)
        return "CPU"


DEVICE = _pick_device()

# ======================== 模型持有与惰性加载 ========================
_tts_model = None
_tts_error: str | None = None
_tts_ready = threading.Event()

_asr_model = None
_asr_error: str | None = None
_asr_ready = threading.Event()

# TTS Base（克隆用）：惰性加载，首次 /tts/clone 请求时触发；空闲超阈值自动卸载释放内存
_tts_base_model = None
_tts_base_error: str | None = None
_tts_base_ready = threading.Event()
_tts_base_loading = False
_tts_base_last_used = 0.0  # 最近一次克隆合成完成的时间戳（0=从未使用）

# 推理并发限流：TTS 是逐 token 生成的重负载（限 1），ASR 相对轻（限 2）
_tts_slots = threading.BoundedSemaphore(1)
_asr_slots = threading.BoundedSemaphore(2)


def _load_tts():
    """[后台线程] 加载 TTS OpenVINO IR；产物缺失/损坏且开启自动下载时先下载+转换"""
    global _tts_model, _tts_error
    with _LOAD_CHAIN:
        try:
            _ensure_ir("tts")
            _compress_ir_int8("tts")
            print(f"🔄 [TTS] 加载 OpenVINO IR ({DEVICE})...", flush=True)
            from qwen_3_tts_helper import OVQwen3TTSModel
            _tts_model = OVQwen3TTSModel.from_pretrained(TTS_OV, device=DEVICE)
            _tts_ready.set()
            print(f"✅ [TTS] 模型加载完成，音色: {sorted(_tts_model.get_supported_speakers())}", flush=True)
        except Exception as e:
            _tts_error = f"TTS 模型加载失败: {e}"
            print(f"❌ [TTS] {_tts_error}", flush=True)


def _load_asr():
    """[后台线程] 加载 ASR OpenVINO IR；产物缺失/损坏且开启自动下载时先下载+转换"""
    global _asr_model, _asr_error
    with _LOAD_CHAIN:
        try:
            _ensure_ir("asr")
            print(f"🔄 [ASR] 加载 OpenVINO IR ({DEVICE})...", flush=True)
            from qwen_3_asr_helper import OVQwen3ASRModel
            _asr_model = OVQwen3ASRModel.from_pretrained(ASR_OV, device=DEVICE)
            _asr_ready.set()
            print("✅ [ASR] 模型加载完成", flush=True)
        except Exception as e:
            _asr_error = f"ASR 模型加载失败: {e}"
            print(f"❌ [ASR] {_asr_error}", flush=True)


def _load_tts_base():
    """[惰性加载] 克隆用 Base 模型；产物缺失/损坏且开启自动下载时先下载+转换"""
    global _tts_base_model, _tts_base_error, _tts_base_loading
    _tts_base_loading = True
    with _LOAD_CHAIN:
        try:
            _ensure_ir("tts-base")
            _compress_ir_int8("tts-base")
            print(f"🔄 [TTS-Base] 加载 OpenVINO IR ({DEVICE})...", flush=True)
            from qwen_3_tts_helper import OVQwen3TTSModel
            _tts_base_model = OVQwen3TTSModel.from_pretrained(TTS_BASE_OV, device=DEVICE)
            _tts_base_ready.set()
            print("✅ [TTS-Base] 模型加载完成（支持音色克隆）", flush=True)
        except Exception as e:
            _tts_base_error = f"TTS Base 模型加载失败: {e}"
            print(f"❌ [TTS-Base] {_tts_base_error}", flush=True)
        finally:
            _tts_base_loading = False


def _ensure_tts_base():
    """[惰性触发] Base 模型未加载时启动后台加载线程（幂等）"""
    global _tts_base_last_used
    if _tts_base_model is None and not _tts_base_loading:
        _tts_base_last_used = time.time()  # 视为「刚被需要」，避免空闲卸载器刚加载完就把它卸了
        threading.Thread(target=_load_tts_base, daemon=True).start()


def _tts_base_idle_watcher():
    """[后台线程] Base 模型空闲自动卸载：闲置超过 TTS_BASE_IDLE_UNLOAD 分钟释放约 3.8GB 内存

    - 仅在无请求正在合成时卸载（拿不到 _tts_slots 说明有合成进行中，跳过本轮）
    - 下次克隆/使用克隆音色时 _ensure_tts_base 会自动重新加载（需 1-2 分钟）
    - TTS_BASE_IDLE_UNLOAD=0 时本线程不启动（常驻策略）
    """
    global _tts_base_last_used
    if TTS_BASE_IDLE_UNLOAD <= 0:
        return
    while True:
        time.sleep(60)
        if _tts_base_model is None or not _tts_base_ready.is_set():
            continue
        idle_min = (time.time() - _tts_base_last_used) / 60
        if _tts_base_last_used > 0 and idle_min >= TTS_BASE_IDLE_UNLOAD:
            if _tts_slots.acquire(blocking=False):
                try:
                    if _tts_base_model is not None and _tts_base_ready.is_set():
                        _unload("tts-base")
                        print(f"💤 [TTS-Base] 空闲 {idle_min:.0f} 分钟，已卸载释放内存"
                              f"（下次克隆自动重新加载）", flush=True)
                finally:
                    _tts_slots.release()


def _unload(kind: str):
    """卸载内存模型实例（删除模型/强制重下前调用）"""
    global _tts_model, _asr_model, _tts_error, _asr_error, _tts_base_model, _tts_base_error, _tts_base_loading
    if kind == "tts":
        _tts_model = None
        _tts_error = None
        _tts_ready.clear()
        print("🗑 [TTS] 内存实例已卸载", flush=True)
    elif kind == "tts-base":
        _tts_base_model = None
        _tts_base_error = None
        _tts_base_ready.clear()
        _tts_base_loading = False
        print("🗑 [TTS-Base] 内存实例已卸载", flush=True)
    else:
        _asr_model = None
        _asr_error = None
        _asr_ready.clear()
        print("🗑 [ASR] 内存实例已卸载", flush=True)


app = FastAPI(title="Qwen3 Speech Sidecar (ASR+TTS)")


def _check_token(x_speech_token: str | None = Header(default=None, alias="X-Speech-Token")):
    """可选鉴权：仅当设置 SPEECH_TOKEN 环境变量时生效"""
    if SPEECH_TOKEN and x_speech_token != SPEECH_TOKEN:
        raise HTTPException(401, "speech token 无效")


# ======================== 工具 ========================

def _sse(event: str, data: dict) -> str:
    """构造一条 SSE 事件字符串"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _dir_size(path: str) -> int:
    """递归统计目录磁盘占用（字节）"""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _split_sentences(text: str, max_len: int = 100) -> list[str]:
    """长文本按句切分（句末标点优先，超长再按逗号二段），供逐批合成后拼接"""
    parts = re.split(r"(?<=[。！？；!?;\n])", text)
    segs: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        while len(p) > max_len:
            cut = p.rfind("，", 0, max_len)
            if cut < max_len // 2:  # 逗号太靠前就硬切，避免死循环
                cut = max_len
            segs.append(p[:cut])
            p = p[cut:]
        segs.append(p)
    return segs


def _run_streaming(cmd: list[str]):
    """运行子进程并逐行 yield stdout（stderr 合并），结束时校验退出码"""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    tail: list[str] = []
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        tail.append(line)
        tail = tail[-5:]
        yield line
    proc.wait(timeout=7200)
    if proc.returncode != 0:
        raise RuntimeError("子进程失败(exit=%d): %s" % (proc.returncode, "\n".join(tail)))


def _to_wav16k(src: str, dst: str):
    """ffmpeg 转单声道 16k wav（任意 ffmpeg 可解码格式）"""
    subprocess.run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", dst],
                   capture_output=True, timeout=120, check=True)


def _model_stat(ckpt: str, ov_dir: str, ready_ev: threading.Event, err: str | None) -> dict:
    return {
        "ckpt": os.path.isdir(ckpt),      # 原始权重已下载
        "ov": os.path.isdir(ov_dir),      # IR 已转换
        "ready": ready_ev.is_set(),       # 内存实例已就绪可推理
        "error": err,
        "size_bytes": _dir_size(ckpt) + _dir_size(ov_dir),
    }


def _dl_active(kind: str) -> bool:
    """该 kind 是否正在下载/转换（或等待中）"""
    s = _dl_state.get(kind, "")
    return bool(s) and not s.startswith(("done", "error"))


def _wait_or_detail(kind: str, ev: threading.Event, timeout: float) -> str | None:
    """等待模型就绪；就绪返回 None，超时返回给客户端的提示（含下载/转换进度或错误）"""
    if ev.wait(timeout=timeout):
        return None
    if _dl_active(kind):
        return f"{kind} 模型正在自动下载/转换（{_dl_state[kind]}），后台不会中断，请稍后重试"
    return _MODEL_ERR[kind]() or f"{kind} 模型未就绪（加载中或未下载）"


def _validate_ir(kind: str) -> None:
    """校验 ov 目录核心产物齐全且非 0 字节；不完整抛 RuntimeError（防半成品被当已转换）"""
    _, _, ov_dir = MODEL_KINDS[kind]
    missing = []
    for rel in _EXPECTED_IR[kind]:
        p = os.path.join(ov_dir, rel)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            missing.append(rel)
    if missing:
        raise RuntimeError(f"IR 产物不完整（{len(missing)} 个文件缺失/为空）: {', '.join(missing[:4])}")


def _ensure_ir(kind: str) -> None:
    """确保 IR 产物存在且完整；缺失/损坏时自动下载+转换（AUTO_DOWNLOAD=0 时报错提示手动）"""
    _, ckpt, ov_dir = MODEL_KINDS[kind]
    if os.path.isdir(ov_dir):
        try:
            _validate_ir(kind)
            return  # 产物完整，直接使用
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ [{kind}] IR 校验失败（{e}），清除损坏产物后重新转换...", flush=True)
            shutil.rmtree(ov_dir, ignore_errors=True)
    if not AUTO_DOWNLOAD:
        raise RuntimeError(_MODEL_ERR[kind]() or f"{kind} 模型未下载/未转换，请先 POST /models/download?kind={kind}")
    print(f"⬇️ [{kind}] 自动下载+转换（首次约 10-20 分钟）...", flush=True)
    _download_and_convert(kind)


def _download_and_convert(kind: str, on_line=None) -> None:
    """下载权重并转换 IR（同 kind 串行互斥；on_line 回调透传进度，缺省打印日志）

    成功后置 _dl_state="done" 且 IR 目录存在；失败置 "error: ..." 并抛异常。
    并发的后来者会等待首个下载者完成（而非重复下载）。
    """
    model_id, ckpt, ov_dir = MODEL_KINDS[kind]
    lock = _dl_locks[kind]
    if not lock.acquire(blocking=False):
        # 已有同 kind 下载在跑：轮询等待其结束
        while _dl_active(kind) and not os.path.isdir(ov_dir):
            time.sleep(5)
        if os.path.isdir(ov_dir):
            return
        raise RuntimeError(f"{kind} 自动下载未成功（{_dl_state[kind]}），请稍后重试或用 /models/download 手动触发")

    def emit(msg: str):
        _dl_state[kind] = msg
        if on_line:
            on_line(msg)
        else:
            print(f"⬇️ [{kind}] {msg}", flush=True)

    try:
        if not os.path.isdir(ckpt):
            emit(f"📥 开始下载 {model_id} (modelscope)...")
            script = _DL_SCRIPT.format(model_id=model_id, local_dir=ckpt)
            for line in _run_streaming([sys.executable, "-u", "-c", script]):
                emit(f"📥 {line}")
        if not os.path.isdir(ckpt):
            raise RuntimeError(f"下载完成但未找到权重目录 {ckpt}")
        emit("🧠 转换 OpenVINO IR（需数分钟，全局串行防 OOM）...")
        with _CONV_LOCK:
            for line in _run_streaming([sys.executable, "-u", "-c", _conv_script(kind)]):
                emit(f"🧠 {line}")
            if not os.path.isdir(ov_dir):
                raise RuntimeError(f"转换完成但未找到 IR 目录 {ov_dir}")
            _validate_ir(kind)  # 锁内校验：不完整即失败并清理，防半成品
        _dl_state[kind] = "done"
        emit(f"✅ {model_id} 下载+转换完成")
    except Exception as e:  # noqa: BLE001
        _dl_state[kind] = f"error: {e}"
        # 清理损坏的 ov 半成品（ckpt 保留，重试无需重新下载 8GB）
        shutil.rmtree(ov_dir, ignore_errors=True)
        raise
    finally:
        lock.release()


# int8 权重压缩目标（相对 ov 目录、不带扩展名）：仅 talker 语言模型（自回归逐 token 生成的耗时大头）
_INT8_TARGETS = {
    "tts": ["openvino_talker_language_model"],
    "tts-base": ["openvino_talker_language_model"],
}


def _compress_ir_int8(kind: str) -> None:
    """加载前对 fp16 IR 做 nncf int8 权重压缩（原地覆盖，.int8 标记防重复；失败降级 fp16 不阻塞加载）"""
    targets = _INT8_TARGETS.get(kind)
    if not targets or not TTS_INT8:
        return
    _, _, ov_dir = MODEL_KINDS[kind]
    if not os.path.isdir(ov_dir):
        return
    if os.path.isfile(os.path.join(ov_dir, ".int8")):
        return  # 已压缩过
    try:
        import nncf
        core = ov.Core()
        for name in targets:
            xml = os.path.join(ov_dir, name + ".xml")
            if not os.path.isfile(xml):
                continue
            print(f"🗜️ [{kind}] int8 权重压缩 {name}（一次性，约 1-3 分钟）...", flush=True)
            t0 = time.time()
            m = core.read_model(xml)
            ov.save_model(nncf.compress_weights(m, mode=nncf.CompressWeightsMode.INT8_ASYM, group_size=-1), xml)
            print(f"✅ [{kind}] {name} int8 压缩完成（{time.time() - t0:.0f}s）", flush=True)
        with open(os.path.join(ov_dir, ".int8"), "w", encoding="utf-8") as f:
            f.write("ok")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ [{kind}] int8 压缩失败（{e}），继续使用 fp16", flush=True)


# ======================== 自定义（克隆）音色管理 ========================

def _custom_voice_dir(vid: str) -> str:
    return os.path.join(VOICES_DIR, vid)


def _list_custom_voices() -> list[dict]:
    """枚举已保存的克隆音色（meta.json + ref.wav）"""
    out: list[dict] = []
    if not os.path.isdir(VOICES_DIR):
        return out
    for vid in sorted(os.listdir(VOICES_DIR)):
        meta_p = os.path.join(VOICES_DIR, vid, "meta.json")
        if not os.path.isfile(meta_p):
            continue
        try:
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
            out.append({"id": vid, "name": meta.get("name", vid),
                        "desc": meta.get("desc", "克隆音色"),
                        "custom": True,
                        "has_ref_text": bool(meta.get("ref_text")),
                        "created_at": meta.get("created_at", "")})
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ 跳过损坏的自定义音色 {vid}: {e}", flush=True)
    return out


def _load_custom_voice(vid: str) -> tuple[dict, str] | None:
    """读取自定义音色 → (meta, ref.wav 路径)；不存在返回 None"""
    d = _custom_voice_dir(vid)
    meta_p = os.path.join(d, "meta.json")
    ref_p = os.path.join(d, "ref.wav")
    if not (os.path.isfile(meta_p) and os.path.isfile(ref_p)):
        return None
    try:
        with open(meta_p, encoding="utf-8") as f:
            return json.load(f), ref_p
    except Exception:
        return None


def _new_voice_id(name: str) -> str:
    """名称 → 唯一目录 id（保留中文/字母数字/连字符，重名自动加序号）"""
    base = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", (name or "").strip()).strip("-").lower() or "voice"
    base = base[:40]
    vid, n = base, 2
    while os.path.exists(_custom_voice_dir(vid)):
        vid = f"{base}-{n}"
        n += 1
    return vid


def _save_ref_as_voice(src_path: str, name: str, ref_text: str | None) -> dict:
    """把参考音频转 24k 单声道 wav 并保存为自定义音色；ref_text 为空时尝试 ASR 自动转写"""
    vid = _new_voice_id(name)
    d = _custom_voice_dir(vid)
    os.makedirs(d, exist_ok=True)
    ref_wav = os.path.join(d, "ref.wav")
    subprocess.run(["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "24000", ref_wav],
                   capture_output=True, timeout=120, check=True)
    auto_note = None
    if not ref_text:
        ref_text = _auto_ref_text(ref_wav)
        auto_note = "已用 Qwen3-ASR 自动转写" if ref_text else "自动转写失败（ASR 未就绪），合成时将用声纹模式"
    meta = {"id": vid, "name": name.strip() or vid, "desc": "克隆音色",
            "ref_text": ref_text or "", "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return {"meta": meta, "ref_text_auto_note": auto_note}


def _auto_ref_text(ref_wav: str) -> str | None:
    """用已就绪的 Qwen3-ASR 自动转写参考音频（供 ICL 克隆模式）；未就绪/失败返回 None"""
    try:
        if not _asr_ready.is_set():
            return None
        if not _asr_slots.acquire(blocking=False):
            return None
        try:
            results = _asr_model.transcribe(ref_wav, language=None)
        finally:
            _asr_slots.release()
        return (results[0].text.strip() or None) if results else None
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 参考音频自动转写失败: {e}", flush=True)
        return None


def _clone_wav_response(text: str, ref_wav: str, ref_text: str | None,
                        language: str, x_vector_only: bool, started: float) -> Response:
    """[共享] 克隆路径合成：确保 Base 模型 → 按句分批 generate_voice_clone → wav Response

    无 ref_text 时自动降级为声纹模式（x_vector_only）。
    """
    global _tts_base_last_used
    _ensure_tts_base()
    detail = _wait_or_detail("tts-base", _tts_base_ready, TTS_BASE_LOAD_TIMEOUT)
    if detail:
        raise HTTPException(503, detail)
    if not _tts_slots.acquire(blocking=False):
        raise HTTPException(429, "TTS 合成并发已满，请稍后重试")
    try:
        segments = _split_sentences(text)
        xvec = bool(x_vector_only) or not ref_text  # 无转写文本则只能用声纹模式
        wavs: list[np.ndarray] = []
        fs = None
        for i in range(0, len(segments), 4):
            batch = segments[i:i + 4]
            w, fs = _tts_base_model.generate_voice_clone(
                batch, language=language if language and language != "Auto" else None,
                ref_audio=ref_wav, ref_text=ref_text,
                x_vector_only_mode=xvec, non_streaming_mode=True)
            wavs.extend(np.asarray(x).reshape(-1) for x in w)
        if not wavs or fs is None:
            raise HTTPException(500, "合成结果为空")
        audio = np.concatenate(wavs)
        buf = io.BytesIO()
        sf.write(buf, audio, fs, format="WAV")
        _tts_base_last_used = time.time()  # 供空闲卸载器计时
        elapsed = int((time.monotonic() - started) * 1000)
        print(f"🎙️ 克隆合成完成: {len(segments)} 段 {len(audio) / fs:.1f}s 耗时 {elapsed}ms "
              f"(x_vector_only={xvec})", flush=True)
        return Response(content=buf.getvalue(), media_type="audio/wav",
                        headers={"X-Elapsed-Ms": str(elapsed), "X-Segments": str(len(segments)),
                                 "X-Xvector-Only": "true" if xvec else "false"})
    finally:
        _tts_slots.release()


# 子进程脚本模板：modelscope 下载权重（带心跳，避免网络空闲时 SSE 假死）
_DL_SCRIPT = """\
import time, threading
start = time.time()
stop = threading.Event()
def heartbeat():
    while not stop.wait(5):
        e = int(time.time() - start)
        print('[%02d:%02d] downloading...' % (e//60, e%60), flush=True)
hb = threading.Thread(target=heartbeat, daemon=True); hb.start()
try:
    print('connecting...', flush=True)
    from modelscope import snapshot_download
    snapshot_download('{model_id}', local_dir=r'{local_dir}')
    print('DONE', flush=True)
finally:
    stop.set(); hb.join(timeout=1)
"""

# 子进程脚本模板：权重 → OpenVINO IR 转换（隔离在子进程，转换期 fp16 torch 模型内存不占服务进程）
_CONV_TPL = """\
import sys
sys.path.insert(0, r'{helper_dir}')
from {helper_mod} import {convert_fn}
{convert_fn}(r'{ckpt}', r'{out}')
print('CONVERT_DONE', flush=True)
"""


def _conv_script(kind: str) -> str:
    helper_dir = os.path.dirname(os.path.abspath(__file__))
    model_id, ckpt, out = MODEL_KINDS[kind]
    convert_fn = "convert_qwen3_tts_model" if kind in ("tts", "tts-base") else "convert_qwen3_asr_model"
    helper_mod = "qwen_3_tts_helper" if kind in ("tts", "tts-base") else "qwen_3_asr_helper"
    return _CONV_TPL.format(helper_dir=helper_dir, helper_mod=helper_mod,
                            convert_fn=convert_fn, ckpt=ckpt, out=out)


# ======================== 端点 ========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "tts": _tts_ready.is_set(),
        "tts_base": _tts_base_ready.is_set(),
        "asr": _asr_ready.is_set(),
        "tts_downloading": _dl_active("tts"),
        "tts_base_downloading": _dl_active("tts-base"),
        "asr_downloading": _dl_active("asr"),
        "tts_model_id": TTS_MODEL_ID,
        "tts_base_model_id": TTS_BASE_MODEL_ID,
        "asr_model_id": ASR_MODEL_ID,
    }


@app.get("/voices", dependencies=[Depends(_check_token)])
def voices():
    """音色列表 = 内置 9 音色 + 自定义克隆音色（custom=true）；模型已加载时用模型侧校验"""
    items = [dict(v) for v in VOICES]
    if _tts_model is not None:
        supported = set(_tts_model.get_supported_speakers())
        for v in items:
            v["available"] = v["id"] in supported
    items.extend(_list_custom_voices())
    return {"voices": items, "default": TTS_DEFAULT_SPEAKER}


@app.post("/voices/custom", dependencies=[Depends(_check_token)])
def voices_custom_create(file: UploadFile = File(...),
                         name: str = Form(...),
                         ref_text: str | None = Form(None)):
    """保存参考音频为自定义（克隆）音色，出现在 /voices 列表可直接当 speaker 使用

    ref_text 留空时自动用 Qwen3-ASR 转写参考音频（ASR 未就绪则降级为声纹模式）。
    """
    name = (name or "").strip()
    if not name:
        raise HTTPException(400, "name 不能为空")
    ref_text = (ref_text or "").strip() or None
    data = file.file.read(UPLOAD_MAX_BYTES + 1)
    if not data:
        raise HTTPException(400, "上传文件为空")
    if len(data) > UPLOAD_MAX_BYTES:
        raise HTTPException(413, f"参考音频超过 {UPLOAD_MAX_BYTES // 1024 // 1024}MB 上限")
    import tempfile
    suffix = os.path.splitext(file.filename or "ref")[1][:16] or ".wav"
    fd, src = None, None
    try:
        fd, src = tempfile.mkstemp(prefix="voice-ref-", suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            fd = None
            f.write(data)
        res = _save_ref_as_voice(src, name, ref_text)
        print(f"💾 自定义音色已保存: {res['meta']['id']} ({res['meta']['name']})", flush=True)
        return {"status": "ok", **res["meta"], "note": res["ref_text_auto_note"]}
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"参考音频解码失败: {e}")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"保存音色失败: {e}")
    finally:
        if fd is not None:
            os.close(fd)
        if src and os.path.exists(src):
            try:
                os.remove(src)
            except OSError:
                pass


@app.delete("/voices/custom/{vid}", dependencies=[Depends(_check_token)])
def voices_custom_delete(vid: str):
    """删除自定义克隆音色"""
    d = _custom_voice_dir(vid)
    if not os.path.isdir(d):
        raise HTTPException(404, f"音色不存在: {vid}")
    shutil.rmtree(d, ignore_errors=True)
    print(f"🗑 自定义音色已删除: {vid}", flush=True)
    return {"status": "ok", "id": vid}


@app.get("/voices/custom/{vid}/ref.wav", dependencies=[Depends(_check_token)])
def voices_custom_ref(vid: str):
    """试听自定义音色的参考音频"""
    ref_p = os.path.join(_custom_voice_dir(vid), "ref.wav")
    if not os.path.isfile(ref_p):
        raise HTTPException(404, f"音色不存在: {vid}")
    return FileResponse(ref_p, media_type="audio/wav")


@app.get("/", include_in_schema=False)
def index():
    """内置网页（合成/克隆/识别，支持麦克风录音输入）"""
    page = os.path.join(STATIC_DIR, "index.html")
    if not os.path.isfile(page):
        return Response("static/index.html 缺失", status_code=404, media_type="text/plain")
    return FileResponse(page, media_type="text/html")


@app.get("/models/status", dependencies=[Depends(_check_token)])
def models_status():
    return {
        "device": DEVICE,
        "tts_model_id": TTS_MODEL_ID,
        "tts_base_model_id": TTS_BASE_MODEL_ID,
        "asr_model_id": ASR_MODEL_ID,
        "tts": dict(_model_stat(TTS_CKPT, TTS_OV, _tts_ready, _tts_error),
                    downloading=_dl_active("tts"), progress=_dl_state["tts"]),
        "tts-base": dict(_model_stat(TTS_BASE_CKPT, TTS_BASE_OV, _tts_base_ready, _tts_base_error),
                         downloading=_dl_active("tts-base"), progress=_dl_state["tts-base"]),
        "asr": dict(_model_stat(ASR_CKPT, ASR_OV, _asr_ready, _asr_error),
                    downloading=_dl_active("asr"), progress=_dl_state["asr"]),
    }


@app.post("/models/download", dependencies=[Depends(_check_token)])
def models_download(kind: str = "tts", force: bool = False):
    """SSE 流式下载+转换：event: progress/done/error，data: {"progress": "..."}"""
    kind = (kind or "").lower().strip()
    if kind not in MODEL_KINDS:
        raise HTTPException(400, "kind 必须是 tts / tts-base / asr")
    model_id, ckpt, ov_dir = MODEL_KINDS[kind]

    def gen():
        try:
            if force:
                yield _sse("progress", {"progress": f"🗑 清除旧缓存 {ckpt} / {ov_dir}..."})
                _unload(kind)
                shutil.rmtree(ckpt, ignore_errors=True)
                shutil.rmtree(ov_dir, ignore_errors=True)
                _dl_state[kind] = ""
            # 复用自动下载的同一函数（共享同 kind 互斥锁，手动/自动并发不会重复下载）
            q: queue.Queue = queue.Queue()
            def _run():
                try:
                    _download_and_convert(kind, on_line=lambda m: q.put(m))
                    q.put(None)  # 结束哨兵
                except Exception as e:  # noqa: BLE001
                    q.put(e)
            threading.Thread(target=_run, daemon=True).start()
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield _sse("progress", {"progress": item})
            # 下载转换完成后后台预载（tts-base 惰性模型也直接预载，克隆首次请求少等一次加载）
            _unload(kind)
            loader = {"tts": _load_tts, "tts-base": _load_tts_base, "asr": _load_asr}[kind]
            threading.Thread(target=loader, daemon=True).start()
            yield _sse("done", {"progress": f"✅ {model_id} 就绪"})
        except Exception as e:  # noqa: BLE001
            yield _sse("error", {"progress": f"❌ {e}"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                                      "X-Accel-Buffering": "no"})


@app.post("/models/delete", dependencies=[Depends(_check_token)])
def models_delete(body: dict):
    """删除已下载模型（权重+IR）并卸载内存实例"""
    kind = (body.get("kind") or "").lower().strip()
    if kind not in MODEL_KINDS:
        raise HTTPException(400, "kind 必须是 tts / tts-base / asr")
    _unload(kind)
    _, ckpt, ov_dir = MODEL_KINDS[kind]
    for d in (ckpt, ov_dir):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    return {"status": "ok", "kind": kind, "ckpt": os.path.isdir(ckpt), "ov": os.path.isdir(ov_dir)}


@app.post("/tts/synthesize", dependencies=[Depends(_check_token)])
def tts_synthesize(body: dict):
    """语音合成：JSON {"text","speaker","language","instruct"} → audio/wav

    speaker 可为内置音色 id，也可为自定义克隆音色 id（走 Base 模型克隆路径，
    有转写文本用 ICL，否则声纹模式）。长文本自动按句切分、分批（每批 4 段）合成拼接。
    """
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text 不能为空")
    speaker = body.get("speaker") or TTS_DEFAULT_SPEAKER
    language = body.get("language") or "Auto"
    instruct = (body.get("instruct") or "").strip() or None

    # 自定义克隆音色：路由到 Base 模型克隆路径
    custom = _load_custom_voice(speaker)
    if custom is not None:
        meta, ref_wav = custom
        ref_text = meta.get("ref_text") or None
        return _clone_wav_response(text, ref_wav, ref_text, language,
                                   x_vector_only=not ref_text, started=time.monotonic())

    detail = _wait_or_detail("tts", _tts_ready, TTS_LOAD_TIMEOUT)
    if detail:
        raise HTTPException(503, detail)

    if not _tts_slots.acquire(blocking=False):
        raise HTTPException(429, "TTS 合成并发已满，请稍后重试")
    started = time.monotonic()
    try:
        segments = _split_sentences(text)
        wavs: list[np.ndarray] = []
        fs = None
        for i in range(0, len(segments), 4):
            batch = segments[i:i + 4]
            w, fs = _tts_model.generate_custom_voice(
                batch, speaker=speaker, language=language, instruct=instruct,
                non_streaming_mode=True)
            wavs.extend(np.asarray(w).reshape(-1) for w in w)
        if not wavs or fs is None:
            raise HTTPException(500, "合成结果为空")
        audio = np.concatenate(wavs)
        buf = io.BytesIO()
        sf.write(buf, audio, fs, format="WAV")
        elapsed = int((time.monotonic() - started) * 1000)
        print(f"🔊 TTS 完成: {len(segments)} 段 {len(audio)/fs:.1f}s 耗时 {elapsed}ms", flush=True)
        return Response(content=buf.getvalue(), media_type="audio/wav",
                        headers={"X-Elapsed-Ms": str(elapsed), "X-Segments": str(len(segments))})
    finally:
        _tts_slots.release()


@app.post("/tts/clone", dependencies=[Depends(_check_token)])
def tts_clone(file: UploadFile = File(...),
              text: str = Form(...),
              ref_text: str | None = Form(None),
              language: str = Form("Auto"),
              x_vector_only: bool = Form(False),
              save_as: str | None = Form(None)):
    """音色克隆合成：上传参考音频 + 目标文本 → audio/wav（Base 模型，惰性加载）

    multipart/form-data 字段：
      file          参考音频（任意 ffmpeg 可解码格式，建议 ≥5s 干净人声，统一转 24k 单声道）
      text          要合成的目标文本（长文本自动按句切分）
      ref_text      参考音频的文字转写（留空且 ASR 就绪时自动转写）
      language      目标语言，默认 Auto（自动）
      x_vector_only true = 仅用声纹克隆（无需 ref_text）
      save_as       可选：同时把该参考音频保存为自定义音色（出现在 /voices，可直接当 speaker 用）
    注：首次调用会触发 Base 模型惰性加载（数分钟），之后常驻。
    """
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "text 不能为空")
    ref_text = (ref_text or "").strip() or None
    save_as = (save_as or "").strip() or None

    started = time.monotonic()
    fd, src, ref_wav = None, None, None
    try:
        data = file.file.read(UPLOAD_MAX_BYTES + 1)
        if not data:
            raise HTTPException(400, "上传文件为空")
        if len(data) > UPLOAD_MAX_BYTES:
            raise HTTPException(413, f"参考音频超过 {UPLOAD_MAX_BYTES // 1024 // 1024}MB 上限")

        import tempfile
        suffix = os.path.splitext(file.filename or "ref")[1][:16] or ".wav"
        fd, src = tempfile.mkstemp(prefix="clone-ref-", suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            fd = None
            f.write(data)
        # 参考音频统一转 24k 单声道 wav（speaker encoder 采样率 24k）
        ref_wav = src + ".24k.wav"
        subprocess.run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "24000", ref_wav],
                       capture_output=True, timeout=120, check=True)
        os.remove(src)
        src = None

        # 可选：保存为自定义音色（ref_text 留空时自动 ASR 转写，转写结果也用于本次合成）
        save_note = None
        if save_as:
            res = _save_ref_as_voice(ref_wav, save_as, ref_text)
            ref_text = res["meta"].get("ref_text") or None
            save_note = res["meta"]["id"]
            print(f"💾 克隆请求已保存自定义音色: {save_note}", flush=True)

        resp = _clone_wav_response(text, ref_wav, ref_text, language,
                                   x_vector_only=x_vector_only, started=started)
        if save_note:
            resp.headers["X-Saved-Voice"] = save_note
        return resp
    except HTTPException:
        raise
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"参考音频解码失败: {e}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"克隆合成失败: {e}")
    finally:
        if fd is not None:
            os.close(fd)
        for p in (src, ref_wav):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


@app.post("/asr/transcribe", dependencies=[Depends(_check_token)])
def asr_transcribe(file: UploadFile = File(...)):
    """Qwen3-ASR 识别：任意 ffmpeg 可解码格式 → {"text","language","elapsed_ms"}

    注：OV 版 transcribe 不返回时间戳（return_time_stamps 不支持），仅纯文本。
    """
    started = time.monotonic()
    suffix = os.path.splitext(file.filename or "audio")[1][:16] or ".webm"
    fd, src, wav_path = None, None, None
    try:
        data = file.file.read(UPLOAD_MAX_BYTES + 1)
        if not data:
            raise HTTPException(400, "上传文件为空")
        if len(data) > UPLOAD_MAX_BYTES:
            raise HTTPException(413, f"音频超过 {UPLOAD_MAX_BYTES // 1024 // 1024}MB 上限")

        import tempfile
        fd, src = tempfile.mkstemp(prefix="asr-", suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            fd = None
            f.write(data)
        wav_path = src + ".16k.wav"
        _to_wav16k(src, wav_path)
        os.remove(src)
        src = None  # 原始上传文件已可删除，16k wav 转写完成后再清

        detail = _wait_or_detail("asr", _asr_ready, ASR_LOAD_TIMEOUT)
        if detail:
            raise HTTPException(503, detail)
        if not _asr_slots.acquire(blocking=False):
            raise HTTPException(429, "ASR 并发已满，请稍后重试")
        try:
            results = _asr_model.transcribe(wav_path, language=None)  # 自动检测语言
        finally:
            _asr_slots.release()
        text = results[0].text.strip() if results else ""
        if not text:
            raise HTTPException(400, "未识别到语音，请重试")
        os.remove(wav_path)  # 转写完成即清理，防临时文件累积
        elapsed = int((time.monotonic() - started) * 1000)
        print(f"🗣️ ASR 完成: {len(text)} 字 耗时 {elapsed}ms", flush=True)
        return {"text": text, "language": results[0].language, "elapsed_ms": elapsed}
    except HTTPException:
        raise
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"音频解码失败: {e}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"ASR 识别失败: {e}")
    finally:
        if fd is not None:
            os.close(fd)
        for p in (src, wav_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


if __name__ == "__main__":
    # 启动即后台预载两模型（fp16 常驻约 8-9GB；克隆用 Base 模型按需惰性加载）
    threading.Thread(target=_load_tts, daemon=True).start()
    threading.Thread(target=_load_asr, daemon=True).start()
    # Base 模型空闲自动卸载器（TTS_BASE_IDLE_UNLOAD>0 时生效，默认 30 分钟）
    threading.Thread(target=_tts_base_idle_watcher, daemon=True).start()
    print(f"🚀 speech sidecar 启动 http://0.0.0.0:7862/health (device={DEVICE})", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=7862)
