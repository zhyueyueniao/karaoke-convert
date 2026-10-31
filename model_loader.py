"""Whisper 模型后台加载、运行时切换、系统状态 HTML 生成"""
import os
import threading
import time
import gradio as gr
import config
import stable_whisper


# ---- 全局状态 ----
model = None                     # Whisper 模型实例（后台加载完成后赋值）
model_ready = threading.Event()  # 模型就绪事件
model_error: str | None = None   # 模型加载错误信息
_model_load_started = False       # 是否已触发后台加载
_model_load_start_time: float = 0.0  # 加载开始时间戳


def _resolve_cached_whisper_dir(model_name: str) -> str | None:
    """在 config.HF_CACHE（模型管理 tab 的下载位置）下定位已下载的 faster-whisper 模型目录。

    返回 snapshots/<rev> 本地路径，供 load_faster_whisper 直接加载——完全绕开
    huggingface_hub 的缓存解析（其默认缓存目录 HF_HUB_CACHE 在 hub>=1.0 时为
    $HF_HOME/hub，与 HF_CACHE 不一致时会找不到已下模型、触发重复下载，表现为
    “模型管理 tab 显示已下载，但处理时一直 等待 Whisper 下载”）。
    未找到含权重文件的完整模型目录则返回 None（由加载器走首次下载）。
    """
    base = os.path.join(config.HF_CACHE, "models--Systran--faster-whisper-" + model_name.replace(".", "--"))
    snapshots = os.path.join(base, "snapshots")
    try:
        if not os.path.isdir(snapshots):
            return None
        revs = [d for d in os.listdir(snapshots) if os.path.isdir(os.path.join(snapshots, d))]
    except OSError:
        return None
    # 取最新的有效 revision：须同时含 config.json 与权重文件（model.bin / model.ct2），
    # 避免残缺缓存（只有 refs 而无完整 blobs）被当成可用模型。
    for rev in sorted(revs, reverse=True):
        d = os.path.join(snapshots, rev)
        try:
            files = set(os.listdir(d))
        except OSError:
            continue
        if "config.json" in files and any(f in files for f in ("model.bin", "model.ct2")):
            return d
    return None


def _load_model():
    """[后台线程] 下载并加载 Whisper 模型，完成后设置 model_ready 事件。

    对 hf-mirror 等不稳定镜像的瞬时断开（'Server disconnected without sending a response.'）
    做有限重试（3 次，间隔 5s 退避），避免一次性网络失败直接降级为 NoneType 崩溃。
    """
    global model, model_error, _model_load_start_time
    _model_load_start_time = time.time()
    last_err = None
    for attempt in range(1, 4):
        try:
            print(f"🔄 [后台] 正在加载 Whisper {config.WHISPER_MODEL} ({config.WHISPER_DEVICE})... (尝试 {attempt}/3)")
            kwargs = {"device": config.WHISPER_DEVICE}
            if config.WHISPER_COMPUTE:
                kwargs["compute_type"] = config.WHISPER_COMPUTE
            # 优先直接加载 model_manager 已下载到 config.HF_CACHE 的本地模型目录，
            # 完全绕开 huggingface_hub 的缓存解析（默认缓存目录与 HF_CACHE 不一致
            # 时会重复下载，表现为“已下载却一直 等待 Whisper 下载”）。
            local_model = _resolve_cached_whisper_dir(config.WHISPER_MODEL)
            if local_model:
                print(f"✅ [后台] 命中本地缓存，直接加载: {local_model}")
                model = stable_whisper.load_faster_whisper(local_model, **kwargs)
            else:
                print(f"⬇️ [后台] 未命中本地缓存，按模型名解析/下载 {config.WHISPER_MODEL}")
                # 按函数签名条件化传参，避免旧版本无 cache_directory 形参时报 unexpected kwarg。
                import inspect
                try:
                    if "cache_directory" in inspect.signature(stable_whisper.load_faster_whisper).parameters:
                        kwargs["cache_directory"] = config.HF_CACHE
                except (ValueError, TypeError):
                    pass
                model = stable_whisper.load_faster_whisper(config.WHISPER_MODEL, **kwargs)
            model_ready.set()
            print("✅ [后台] 模型加载完成")
            return
        except Exception as e:
            last_err = e
            print(f"⚠️ [后台] 模型加载失败（尝试 {attempt}/3）: {e}")
            if attempt < 3:
                time.sleep(5)
    model_error = str(last_err)
    print(f"❌ [后台] 模型加载失败: {last_err}")


def ensure_model_ready():
    """[检查模型状态] 未加载时触发后台加载，返回 (ok, 错误信息)"""
    global _model_load_started
    if not _model_load_started:
        _model_load_started = True
        threading.Thread(target=_load_model, daemon=True).start()
    if model_ready.is_set():
        return True, None
    if model_error:
        return False, f"❌ 模型加载失败: {model_error}"
    if _model_load_start_time:
        elapsed = int(time.time() - _model_load_start_time)
        return False, f"⏳ Whisper 下载中... 已等待 {elapsed // 60:02d}:{elapsed % 60:02d}"
    return False, "⏳ Whisper 正在下载/加载中..."


def wait_model_ready():
    """[生成器] 等待 Whisper 模型就绪，每次 yield (ready, 状态消息)
    
    用于 transcribe/align 流程前半段，阻塞式等待模型加载完成
    """
    ensure_model_ready()
    while not model_ready.is_set():
        if model_error:
            yield False, f"❌ 模型加载失败: {model_error}"
            return
        elapsed = int(time.time() - _model_load_start_time)
        yield False, f"⌛ 等待 Whisper 下载... 已等待 {elapsed // 60:02d}:{elapsed % 60:02d}"
        time.sleep(3)
    yield True, None


def switch_whisper_model(model_name: str) -> str:
    """[运行时切换] 仅允许切换到已下载的模型，未下载弹出警告提示"""
    from model_manager import is_whisper_cached
    global model, model_ready, model_error, _model_load_start_time
    if model_name not in config.AVAILABLE_WHISPER:
        gr.Warning(f"无效模型: {model_name}")
        return config.WHISPER_MODEL
    if model_name == config.WHISPER_MODEL and model_ready.is_set():
        gr.Info(f"已在使用 {model_name}，无需切换")
        return model_name
    if not is_whisper_cached(model_name):
        gr.Warning(f"{model_name} 未下载，请在下方模型列表点击 📥下载")
        return config.WHISPER_MODEL
    config.WHISPER_MODEL = model_name
    config._save_model_pref("WHISPER_MODEL", model_name)
    model = None
    model_ready.clear()
    model_error = None
    if hasattr(stable_whisper, "_MODELS"):
        stable_whisper._MODELS.clear()
    threading.Thread(target=_load_model, daemon=True).start()
    _model_load_start_time = time.time()
    gr.Info(f"正在切换到 {model_name}，后台加载中...")
    return model_name


def get_status_html():
    """[系统状态栏] 生成首页 HTML 状态面板，包含加速模式/模型版本/GPU 信息"""
    if config.DEMUCS_DEVICE == "cuda":
        mode = "NVIDIA CUDA（全加速）"
        whisper_mode = "CUDA"
    elif config.DEMUCS_DEVICE == "openvino":
        mode = "混合加速"
        whisper_mode = "CPU (ctranslate2 4.x 无 OpenVINO 后端)"
    else:
        mode = "纯 CPU"
        whisper_mode = "CPU"

    dri_nodes = [p for p in ["/dev/dri/renderD128", "/dev/dri/card0"] if os.path.exists(p)]
    dri_info = ", ".join(dri_nodes) if dri_nodes else "无"
    ov_info = "未安装"
    if config._HAS_OPENVINO:
        try:
            core = config.ov.Core()
            ov_info = ", ".join(core.available_devices)
        except Exception as e:
            ov_info = f"探测失败: {e}"

    # 歌词对齐后端（仅"有歌词文本"时使用；无歌词听写恒为 Whisper）
    if config.ALIGN_BACKEND == "qwen3":
        align_backend_info = (f"Qwen3-ForcedAligner {config.ALIGNER_MODEL}"
                              f"（文本↔音频直接对齐，失败自动回退 stable-ts）")
    else:
        align_backend_info = "stable-ts（whisper 识别 + DTW）"

    return f"""
<div style="background:#1e1e2e; border-radius:12px; padding:16px 24px; margin-bottom:16px;
            font-family:monospace; line-height:1.8;">
<b style="color:#cdd6f4; font-size:16px;">📊 系统状态</b><br>
<span style="color:#a6adc8;">整体模式</span>
  <b style="color:#89b4fa;">{mode}</b><br>
<span style="color:#a6adc8;">人声分离 (Demucs)</span>
  <b style="color:#89b4fa;">{config.DEMUCS_DEVICE or "CPU"}</b><br>
<span style="color:#a6adc8;">听写/对齐 (Whisper)</span>
  <span style="color:#cdd6f4;">{whisper_mode}</span><br>
<span style="color:#a6adc8;">Whisper 模型</span>
  <span style="color:#cdd6f4;">{config.WHISPER_MODEL} (faster-whisper + stable-ts)</span><br>
<span style="color:#a6adc8;">歌词对齐（有歌词时）</span>
  <span style="color:#cdd6f4;">{align_backend_info}</span><br>
<span style="color:#a6adc8;">人声分离模型</span>
  <span style="color:#cdd6f4;">Demucs {config.DEMUCS_MODEL}</span><br>
<span style="color:#a6adc8;">DRI 节点</span>
  <span style="color:#cdd6f4;">{dri_info}</span><br>
<span style="color:#a6adc8;">OpenVINO 设备</span>
  <span style="color:#cdd6f4;">{ov_info}</span><br>
</div>
"""
