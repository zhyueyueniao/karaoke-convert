"""Qwen3-ForcedAligner 对齐后端：模型惰性后台加载 + 词级时间戳对齐

独立于 stable-ts 的第二个对齐后端（config.ALIGN_BACKEND = "qwen3" 启用）。
设计动机：stable-ts align() 的对齐锚点依赖 whisper 的歌声识别质量——识别错则
DTW 锚点漂移，中文歌声场景尤甚。Qwen3-ForcedAligner-0.6B 是专用非自回归（NAR）
对齐模型，文本↔音频直接对齐，不经识别环节，原生支持中文/粤语。

官方用法（QwenLM/Qwen3-ASR README）:
    from qwen_asr import Qwen3ForcedAligner
    model = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B", ...)
    results = model.align(audio=..., text=..., language="Chinese")
    results[0] -> 词对象列表，每个含 .text / .start_time / .end_time

约束与取舍：
- 官方标注单次对齐语音上限 5 分钟。≤5min 整段一次对齐；超长则音频 ffmpeg
  二分 + 歌词行对半切分，递归对齐后按偏移拼接（切点处逐词时间可能有 1~2s 误差，
  KTV 曲库超 5 分钟的歌占比小，Pilot 阶段接受）。
- 输入音频统一转 16kHz 单声道 wav 临时文件，规避 qwen_asr 内部加载器
  对部分容器格式（mp4/mkv）的不兼容。
- 语言映射：config.WHISPER_LANGUAGE(zh/yue/en) → Aligner 语言全名。
"""
import os
import subprocess
import tempfile
import threading
import time

import config

_aligner = None                    # Qwen3ForcedAligner 实例（后台加载完成后赋值）
_ready = threading.Event()         # 模型就绪事件
_error: str | None = None          # 加载失败信息
_load_started = False
_load_start_time = 0.0

MAX_SEGMENT_SEC = 300.0            # 官方单次对齐语音长度上限（秒）

_ALIGN_LANGUAGE = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
}


def _pick_device_dtype():
    """优先 CUDA(bfloat16)，否则 CPU(float32)。bfloat16 在部分 CPU 上不可用，稳妥用 float32。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0", torch.bfloat16
        return "cpu", torch.float32
    except Exception:
        return "cpu", None


def _load():
    """[后台线程] 下载并加载 Aligner，完成后置 _ready。失败重试 3 次（仿 model_loader）。"""
    global _aligner, _error, _load_start_time
    _load_start_time = time.time()
    last_err = None
    for attempt in range(1, 4):
        try:
            import torch
            from qwen_asr import Qwen3ForcedAligner
            device, dtype = _pick_device_dtype()
            print(f"🔄 [后台] 正在加载 Qwen3-ForcedAligner ({config.ALIGNER_MODEL}, {device})... (尝试 {attempt}/3)", flush=True)
            kwargs = {"device_map": device}
            if dtype is not None:
                kwargs["dtype"] = dtype
            _aligner = Qwen3ForcedAligner.from_pretrained(config.ALIGNER_MODEL, **kwargs)
            _ready.set()
            print("✅ [后台] Qwen3-ForcedAligner 加载完成", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"⚠️ [后台] Aligner 加载失败（尝试 {attempt}/3）: {e}", flush=True)
            if attempt < 3:
                time.sleep(5)
    _error = str(last_err)
    print(f"❌ [后台] Qwen3-ForcedAligner 加载失败: {_error}", flush=True)


def ensure_ready() -> tuple[bool, str | None]:
    """未加载时触发后台加载，返回 (ok, 错误信息)"""
    global _load_started
    if not _load_started:
        _load_started = True
        threading.Thread(target=_load, daemon=True).start()
    if _ready.is_set():
        return True, None
    if _error:
        return False, f"❌ Qwen3-Aligner 加载失败: {_error}"
    elapsed = int(time.time() - _load_start_time) if _load_start_time else 0
    return False, f"⏳ Qwen3-Aligner 下载/加载中... 已等待 {elapsed // 60:02d}:{elapsed % 60:02d}"


def wait_ready():
    """[生成器] 等待模型就绪，每次 yield (ready, 状态消息)"""
    ensure_ready()
    while not _ready.is_set():
        if _error:
            yield False, f"❌ Qwen3-Aligner 加载失败: {_error}"
            return
        elapsed = int(time.time() - _load_start_time)
        yield False, f"⌛ 等待 Qwen3-Aligner 下载... 已等待 {elapsed // 60:02d}:{elapsed % 60:02d}"
        time.sleep(3)
    yield True, None


def _probe_duration(audio_path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True, timeout=30)
    return float(out.stdout.strip())


def _to_wav_segment(audio_path: str, start: float, end: float) -> str:
    """切出 [start,end) 段并转 16kHz 单声道 wav 临时文件（-to 用相对起点时长防精度漂移）"""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
         "-i", audio_path, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", path],
        check=True, capture_output=True, timeout=300)
    return path


def _align_once(audio_path: str, text: str, offset: float) -> list[tuple[str, float, float]]:
    """单次对齐，返回 [(词文本, 绝对起始秒, 绝对结束秒)]"""
    language = _ALIGN_LANGUAGE.get(getattr(config, "WHISPER_LANGUAGE", "zh"), "Chinese")
    results = _aligner.align(audio=audio_path, text=text, language=language)
    words: list[tuple[str, float, float]] = []
    for w in results[0]:
        t = (w.text or "").strip()
        if not t:
            continue
        words.append((t, float(w.start_time) + offset, float(w.end_time) + offset))
    return words


def align_words(audio_path: str, clean_text: str) -> list[tuple[str, float, float]]:
    """对齐入口（阻塞调用，建议配 _heartbeat_blocking 使用）。

    - 音频 ≤5min：整段一次前向对齐全文（最快路径）
    - 音频 >5min：ffmpeg 二分音频 + 歌词行对半切分，递归对齐后拼接
    返回 [(词文本, 绝对起始秒, 绝对结束秒)]；失败向上抛异常，由调用方回退 stable-ts。
    """
    lines = [l for l in clean_text.split("\n") if l.strip()]
    if not lines:
        raise ValueError("歌词内容为空")
    duration = _probe_duration(audio_path)
    if duration <= MAX_SEGMENT_SEC or len(lines) == 1:
        wav = _to_wav_segment(audio_path, 0.0, duration)
        try:
            return _align_once(wav, clean_text, 0.0)
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass
    # 超长：二分（行对半 + 音频中点切分）
    mid = len(lines) // 2
    cut = duration / 2.0
    left_wav = _to_wav_segment(audio_path, 0.0, cut)
    right_wav = _to_wav_segment(audio_path, cut, duration)
    try:
        left = _align_once(left_wav, "\n".join(lines[:mid]), 0.0)
        right = _align_once(right_wav, "\n".join(lines[mid:]), cut)
    finally:
        for p in (left_wav, right_wav):
            try:
                os.remove(p)
            except OSError:
                pass
    return left + right
