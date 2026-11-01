"""SenseVoice-Small ASR 语音识别后端（sherpa-onnx int8 CPU）：惰性后台加载 + 带时间戳转写

独立的纯 ASR 功能（音频 → 文本/带时间戳分段），不参与歌词生成/对齐管线——
SenseVoice 无强制对齐能力，歌词管线仍走 Whisper 听写 + stable-ts/Qwen3 对齐。

模型：sherpa-onnx SenseVoice int8（csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue，
阿里 FunAudioLLM SenseVoice-Small 的 onnx 移植，~230MB）。非自回归 NAR 架构，
CPU int8 推理速度约为 faster-whisper medium 的 5~15 倍，支持中/粤/英/日/韩。

sherpa-onnx 官方用法:
    import sherpa_onnx
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model="model.int8.onnx", tokens="tokens.txt", use_itn=True)
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_streams([stream])
    stream.result.text / .timestamps / .tokens

约束与取舍：
- 输入音频统一 ffmpeg 转 16kHz 单声道 wav（sherpa-onnx 只吃波形，不吃容器格式）。
- 时间戳为 CTC token 级，按句末标点聚段后展示，适合阅读/检索，非 KTV 逐词精度。
- tokens/timestamps 依赖较新的 sherpa-onnx 版本，缺失时优雅降级为纯文本。
"""
import os
import subprocess
import tempfile
import threading
import time

import config

_recognizer = None                 # sherpa_onnx.OfflineRecognizer 实例（加载完成后赋值）
_ready = threading.Event()         # 模型就绪事件
_error: str | None = None          # 加载失败信息
_load_started = False
_load_start_time = 0.0
_heal_attempts = 0                 # 粘性错误自愈重试次数（防模型损坏时无限循环）

_SENTENCE_ENDS = "。！？；?!;，,、"   # 分段标点（句末优先换行）


def _resolve_model_dir(model_name: str) -> str | None:
    """定位已下载的 sherpa-onnx SenseVoice 模型目录（含 model.int8.onnx + tokens.txt）。

    支持两种形态：
    - HF 仓库名（如 csukuangfj/sherpa-onnx-sense-voice-...）→ 在 HF_CACHE 下找 snapshots/<rev>
    - 本地目录 → 直接使用
    未找到完整模型则返回 None（由加载器报错，提示先在模型管理下载）。
    """
    if os.path.isdir(model_name):
        return model_name if os.path.exists(os.path.join(model_name, "model.int8.onnx")) else None
    base = os.path.join(config.HF_CACHE, "models--" + model_name.replace("/", "--"))
    snapshots = os.path.join(base, "snapshots")
    try:
        if not os.path.isdir(snapshots):
            return None
        revs = sorted((d for d in os.listdir(snapshots) if os.path.isdir(os.path.join(snapshots, d))),
                      reverse=True)
    except OSError:
        return None
    for rev in revs:
        d = os.path.join(snapshots, rev)
        files = set(os.listdir(d))
        if "model.int8.onnx" in files and "tokens.txt" in files:
            return d
    return None


def _pick_num_threads() -> int:
    """按 CPU 核数选推理线程（P+E 核机器上 4~6 线程通常最优）"""
    try:
        return max(2, min(6, (os.cpu_count() or 4) // 2))
    except Exception:
        return 4


def _load():
    """[后台线程] 加载 SenseVoice int8 模型，完成后置 _ready。失败重试 3 次（仿 model_loader）。"""
    global _recognizer, _error, _load_start_time
    _load_start_time = time.time()
    last_err = None
    for attempt in range(1, 4):
        try:
            import sherpa_onnx
            model_dir = _resolve_model_dir(config.SENSEVOICE_MODEL)
            if not model_dir:
                raise FileNotFoundError(
                    f"未找到 SenseVoice 模型（{config.SENSEVOICE_MODEL}），请先在「模型管理」下载")
            print(f"🔄 [后台] 正在加载 SenseVoice ({model_dir}, cpu int8)... (尝试 {attempt}/3)", flush=True)
            _recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=os.path.join(model_dir, "model.int8.onnx"),
                tokens=os.path.join(model_dir, "tokens.txt"),
                num_threads=_pick_num_threads(),
                use_itn=True,
            )
            _ready.set()
            global _heal_attempts
            _heal_attempts = 0
            print("✅ [后台] SenseVoice 加载完成", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"⚠️ [后台] SenseVoice 加载失败（尝试 {attempt}/3）: {e}", flush=True)
            if attempt < 3:
                time.sleep(5)
    _error = str(last_err)
    print(f"❌ [后台] SenseVoice 加载失败: {_error}", flush=True)


def ensure_ready() -> tuple[bool, str | None]:
    """未加载时触发后台加载，返回 (ok, 错误信息)

    自愈：曾加载失败（典型场景：点识别时模型还没下载完，3 次重试后 _error 粘住）后，
    每次调用先重新探测模型文件——已出现则清错误重启加载，否则下载完成后永远报"未找到模型"。
    自愈最多 3 次，防止模型文件损坏导致无限重试。
    """
    global _load_started, _error, _heal_attempts
    if _error is not None and _resolve_model_dir(config.SENSEVOICE_MODEL):
        if _heal_attempts < 3:
            _heal_attempts += 1
            print(f"🔄 [自愈] 检测到模型文件已就位，清除历史失败状态并重新加载（第 {_heal_attempts}/3 次）", flush=True)
            _error = None
            _load_started = False
            _ready.clear()
    if not _load_started:
        _load_started = True
        threading.Thread(target=_load, daemon=True).start()
    if _ready.is_set():
        return True, None
    if _error:
        return False, f"❌ SenseVoice 加载失败: {_error}"
    elapsed = int(time.time() - _load_start_time) if _load_start_time else 0
    return False, f"⏳ SenseVoice 加载中... 已等待 {elapsed // 60:02d}:{elapsed % 60:02d}"


def wait_ready():
    """[生成器] 等待模型就绪，每次 yield (ready, 状态消息)"""
    ensure_ready()
    while not _ready.is_set():
        if _error:
            if _resolve_model_dir(config.SENSEVOICE_MODEL):
                ensure_ready()  # 模型文件已出现：清错误重启加载，继续等
                time.sleep(3)
                continue
            yield False, f"❌ SenseVoice 加载失败: {_error}"
            return
        elapsed = int(time.time() - _load_start_time)
        yield False, f"⌛ 等待 SenseVoice 加载... 已等待 {elapsed // 60:02d}:{elapsed % 60:02d}"
        time.sleep(3)
    yield True, None


def reset():
    """[模型管理] 删除缓存后重置运行时状态，下次识别将重新加载"""
    global _recognizer, _error, _load_started, _load_start_time, _heal_attempts
    _recognizer = None
    _error = None
    _load_started = False
    _load_start_time = 0.0
    _heal_attempts = 0
    _ready.clear()


def _to_wav16k(audio_path: str) -> str:
    """任意容器格式 → 16kHz 单声道 wav 临时文件"""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", audio_path,
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", path],
        check=True, capture_output=True, timeout=600)
    return path


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m):02d}:{s:05.2f}"


def _segment_text(tokens: list[str], timestamps: list[float]) -> str:
    """token 级时间戳 → 按句末标点聚段的带时间文本：[MM:SS.cc] 句子

    过滤 <|zh|>/<|EMO_UNKNOWN|>/<|withitn|> 等特殊标记（result.text 官方已剥，
    但 tokens 序列可能仍含这些标签）。
    """
    lines: list[str] = []
    buf: list[str] = []
    line_start: float | None = None
    for tok, ts in zip(tokens, timestamps):
        t = (tok or "").strip()
        if not t or (t.startswith("<|") and t.endswith("|>")):
            continue
        if line_start is None:
            line_start = float(ts)
        buf.append(t)
        if any(ch in _SENTENCE_ENDS for ch in t):
            lines.append(f"[{_fmt_ts(line_start)}] {''.join(buf)}")
            buf, line_start = [], None
    if buf and line_start is not None:
        lines.append(f"[{_fmt_ts(line_start)}] {''.join(buf)}")
    return "\n".join(lines)


def transcribe(audio_path: str) -> tuple[str, str]:
    """[核心] SenseVoice 听写。返回 (纯文本, 带时间戳分段文本)。

    阻塞调用，建议配 processing._heartbeat_blocking 使用。
    失败向上抛异常，由调用方处理。
    """
    if _recognizer is None:
        raise RuntimeError("SenseVoice 模型未就绪")
    import numpy as np
    wav = _to_wav16k(audio_path)
    try:
        import wave
        with wave.open(wav, "rb") as wf:
            sample_rate = wf.getframerate()
            # pcm_s16le 字节流必须按 int16 读再归一化到 [-1,1]；
            # 直接按 float32 解读是错的（长度减半、数值全错）
            samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            samples = samples.astype(np.float32) / 32768.0
        stream = _recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        _recognizer.decode_streams([stream])
        result = stream.result
        text = (result.text or "").strip()
        if not text:
            raise RuntimeError("识别结果为空（音频无语音或格式异常）")
        tokens = list(getattr(result, "tokens", None) or [])
        timestamps = list(getattr(result, "timestamps", None) or [])
        ts_text = _segment_text(tokens, timestamps) if tokens and len(tokens) == len(timestamps) else text
        return text, ts_text
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass
