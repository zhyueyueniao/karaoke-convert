"""音频处理：人声分离、歌词听写、歌词对齐、一键处理"""
import os
import re
import shutil
import subprocess
import threading
import time
import gradio as gr
import config


# ---- 工具函数 ----

def _cleanup_empty_dirs(root: str):
    """[清理] 递归删除空目录（Demucs 分离后残留的空文件夹）"""
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if not dirnames and not filenames and dirpath != root:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


def _fmt_elapsed(start: float) -> str:
    """[格式化] 将耗时秒数转为 MM:SS 格式"""
    elapsed = int(time.time() - start)
    return f"{elapsed // 60:02d}:{elapsed % 60:02d}"


def _heartbeat_blocking(worker, start, interval=1.5, pct_start=15, pct_step=4, pct_cap=85):
    """[工具] 后台线程执行阻塞式 worker()，期间持续产出心跳进度。

    stable-ts 的 transcribe/align 是阻塞调用且不提供进度回调；ctranslate2
    推理会释放 GIL，因此把重活丢到后台线程，主生成器每 interval 秒 sleep 一次
    并 yield 递增的心跳百分比，让 SSE 客户端在长任务期间也能看到“还在跑”的进度。

    生成器 yield:
      (pct, "MM:SS")  —— 任务进行中的心跳（pct 从 pct_start 递增至 pct_cap）
      (None, result) —— 任务完成（result 为 worker() 返回值；worker 抛异常则向上抛出）
    """
    holder: dict = {}

    def _run():
        try:
            holder["result"] = worker()
        except BaseException as e:  # noqa: BLE001 - 透传原异常给调用方
            holder["error"] = e

    t = threading.Thread(target=_run, daemon=True, name="whisper-progress")
    t.start()
    beats = 0
    while t.is_alive():
        time.sleep(interval)
        beats += 1
        pct = min(pct_start + beats * pct_step, pct_cap)
        el = int(time.time() - start)
        yield pct, f"{el // 60:02d}:{el % 60:02d}"
    if "error" in holder:
        raise holder["error"]
    yield None, holder.get("result")


def _find_demucs_outputs(base_name: str) -> tuple[str | None, str | None]:
    """[查找文件] 搜索 Demucs 分离后的人声和伴奏文件，返回 (vocals_path, accompaniment_path)"""
    separated_dir = os.path.join(config.OUTPUT_DIR, config.DEMUCS_MODEL, base_name)
    vocal, accomp = None, None
    for ext in (".mp3", ".wav"):
        vf = os.path.join(separated_dir, f"vocals{ext}")
        af = os.path.join(separated_dir, f"no_vocals{ext}")
        if vocal is None and os.path.exists(vf):
            vocal = vf
        if accomp is None and os.path.exists(af):
            accomp = af
    return vocal, accomp


# ---- 人声分离 ----

def _run_demucs(audio_file):
    """[核心] 调用 Demucs 分离人声+伴奏，Popen 实时解析 tqdm 进度条
    
    生成器 yield: (vocal_path, accomp_path, 状态文字)
    - 进行中: (None, None, "进度...")
    - 完成: (path, path, "✅ 完成")
    """
    if audio_file is None:
        yield None, None, "❌ 请上传音频文件"
        return
    yield None, None, f"🎵 开始人声分离（模型: {config.DEMUCS_MODEL}）..."
    start = time.time()
    try:
        filename = os.path.basename(audio_file)
        base_name = os.path.splitext(filename)[0]
        vocal_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_vocals.mp3")
        accomp_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_accomp.mp3")

        cmd = ["python3", "-m", "demucs", "--two-stems", "vocals",
               "-n", config.DEMUCS_MODEL, "--mp3", "-o", config.OUTPUT_DIR]
        if config.DEMUCS_DEVICE and config.DEMUCS_DEVICE != "openvino":
            cmd.extend(["-d", config.DEMUCS_DEVICE])
        cmd.append(audio_file)

        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                text=True, bufsize=1)
        last_pct, stage, stderr_buf = -1, 1, []
        for line in proc.stderr:
            stderr_buf.append(line)
            # 模型下载进度
            if any(kw in line.lower() for kw in ["download", "fetch", "checkpoint", "transfer"]):
                m = re.search(r"(\d+)%", line)
                if m:
                    eta_m = re.search(r"(\d{1,2}:\d{2})", line)
                    eta_suffix = f" 预计 {eta_m.group(1)}" if eta_m else ""
                    yield None, None, f"⏱️ {_fmt_elapsed(start)} | 📥 下载模型中... {m.group(1)}%{eta_suffix}"
                else:
                    yield None, None, f"⏱️ {_fmt_elapsed(start)} | 📥 下载模型文件中..."
                continue
            # 分离进度 (tqdm 非 TTY 模式逐行输出)
            if "%|" in line:
                m = re.search(r"(\d+)%", line)
                if m:
                    pct = int(m.group(1))
                    if last_pct > 80 and pct < 10:
                        stage += 1  # 进度回退 -> 下一个子模型开始
                    if pct != last_pct:
                        last_pct = pct
                        yield None, None, f"⏱️ {_fmt_elapsed(start)} | 🔊 分离中 ({stage}/N)... {pct}%"

        proc.wait(timeout=900)
        if proc.returncode != 0:
            tail = "".join(stderr_buf[-15:])
            print(f"❌ Demucs 失败 (exit={proc.returncode})\n{tail}", flush=True)
            if any(kw in "".join(stderr_buf).lower() for kw in ["downloading", "checkpoint", "not found"]):
                yield None, None, f"⏱️ {_fmt_elapsed(start)} | ⏳ 模型文件下载中，请稍后重试"
            else:
                yield None, None, f"⏱️ {_fmt_elapsed(start)} | ❌ 分离失败: {tail.strip()[-200:]}"
            return

        sep_vocal, sep_accomp = _find_demucs_outputs(base_name)
        if sep_vocal is None and sep_accomp is None:
            yield None, None, f"⏱️ {_fmt_elapsed(start)} | ❌ 未找到分离后的文件"
            return

        if sep_vocal:
            shutil.move(sep_vocal, vocal_path)
        else:
            vocal_path = None
        if sep_accomp:
            shutil.move(sep_accomp, accomp_path)
        else:
            accomp_path = None
        _cleanup_empty_dirs(os.path.join(config.OUTPUT_DIR, config.DEMUCS_MODEL))
        yield vocal_path, accomp_path, f"⏱️ {_fmt_elapsed(start)} | ✅ 人声+伴奏分离完成"
    except subprocess.TimeoutExpired:
        yield None, None, f"⏱️ {_fmt_elapsed(start)} | ❌ 分离超时（超过15分钟）"
    except Exception as e:
        yield None, None, f"⏱️ {_fmt_elapsed(start)} | ❌ 分离异常: {str(e)}"


def separate_vocals(audio_file):
    """[UI 包装] 「单独分离人声」Tab 的入口，带按钮禁用/恢复"""
    if audio_file is None:
        yield None, None, "❌ 请上传音频文件", gr.update(interactive=True)
        return
    for v, a, s in _run_demucs(audio_file):
        done = v is not None or a is not None
        yield v, a, s, gr.update(interactive=done)


# ---- 歌词听写 ----

def _run_transcribe(audio_file):
    """[核心] 调用 faster-whisper 听写生成歌词（逐词 LRC + 行级 LRC + SRT）
    
    生成器 yield: (逐词 LRC 文本, 行级 LRC 文本, 状态文字)
    - Whisper 内部是阻塞调用，无法获取实时百分比进度
    """
    import model_loader
    if audio_file is None:
        yield None, None, "❌ 请上传音频文件"
        return
    start = time.time()
    yield None, None, f"📝 开始歌词听写（模型: {config.WHISPER_MODEL}）..."
    # 等待模型就绪（首次需下载 ~1.5GB）
    model_ok = False
    for ready, msg in model_loader.wait_model_ready():
        if not ready:
            yield None, None, f"⏱️ {_fmt_elapsed(start)} | {msg}"
        else:
            model_ok = True
            break
    if not model_ok:
        # 模型加载失败：必须在此中止，否则会落到 model_loader.model.transcribe() 上 None 模型，
        # 抛出 'NoneType' object has no attribute 'transcribe' 掩盖真实原因。
        yield None, None, f"⏱️ {_fmt_elapsed(start)} | ❌ Whisper 模型加载失败，终止听写（详见 karaoke-convert 服务端日志）"
        return
    try:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        lrc_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_transcribe.lrc")
        lrc_line_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_transcribe_line.lrc")
        srt_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_transcribe.srt")

        yield None, None, f"⏱️ {_fmt_elapsed(start)} | 📝 正在加载音频... 10%"
        yield None, None, f"⏱️ {_fmt_elapsed(start)} | 📝 正在进行语音识别... 15%"
        # transcribe 是阻塞调用且无进度回调：后台线程执行，主流程每 1.5s 心跳一次
        result = None
        for hb_pct, hb_el in _heartbeat_blocking(
                lambda: model_loader.model.transcribe(audio_file, word_timestamps=True, vad_filter=True),
                start):
            if hb_pct is not None:
                yield None, None, f"⏱️ {hb_el} | 🔊 语音识别中... {hb_pct}%"
            else:
                result = hb_el
        if result is None:
            raise RuntimeError("Whisper 听写未返回结果")

        yield None, None, f"⏱️ {_fmt_elapsed(start)} | 📝 正在生成 LRC 歌词... 90%"
        try:
            lrc_word_lines, lrc_line_lines, srt_lines = [], [], []
            for i, seg in enumerate(result.segments, 1):
                text = seg.text.strip()
                if not text:
                    continue
                st, et = seg.start, seg.end
                ts_head = f"[{int(st)//60:02d}:{int(st)%60:02d}.{int((st%1)*100):02d}]"
                # 行级 LRC（卡拉OK 整行切换）
                lrc_line_lines.append(f"{ts_head}{text}")
                # 逐词 LRC（卡拉OK 逐词变色）
                words = getattr(seg, "words", None)
                if words and len(words) > 1:
                    parts = [ts_head]
                    for w in words:
                        wt = w.start
                        parts.append(f"<{int(wt)//60:02d}:{int(wt)%60:02d}.{int((wt%1)*100):02d}>{w.word.strip()}")
                    lrc_word_lines.append("".join(parts))
                else:
                    lrc_word_lines.append(f"{ts_head}{text}")
                # SRT 字幕
                srt_lines.append(str(i))
                srt_lines.append(f"{int(st//3600):02d}:{int((st%3600)//60):02d}:{int(st%60):02d},{int((st%1)*1000):03d}"
                                 f" --> "
                                 f"{int(et//3600):02d}:{int((et%3600)//60):02d}:{int(et%60):02d},{int((et%1)*1000):03d}")
                srt_lines.append(text)
                srt_lines.append("")
            lrc_content = "\n".join(lrc_word_lines) if lrc_word_lines else ""
            lrc_line_content = "\n".join(lrc_line_lines) if lrc_line_lines else ""
            srt_content = "\n".join(srt_lines)
        except AttributeError as e:
            print(f"⚠️ segments 访问失败: {e}", flush=True)
            lrc_content = result.to_lrc() or ""
            lrc_line_content = lrc_content
            srt_content = ""
        with open(lrc_path, "w", encoding="utf-8") as f:
            f.write(lrc_content)
        with open(lrc_line_path, "w", encoding="utf-8") as f:
            f.write(lrc_line_content)
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        yield (lrc_content, lrc_line_content,
               f"⏱️ {_fmt_elapsed(start)} | ✅ 听写完成 100%: {lrc_path} / {lrc_line_path}")
    except Exception as e:
        yield None, None, f"⏱️ {_fmt_elapsed(start)} | ❌ 听写异常: {str(e)}"


# ---- 歌词生成/对齐（统一入口） ----

def align_lyrics(audio_file, lyrics_text):
    """[歌词生成/对齐] 统一入口：
    - lyrics_text 为空 → 听写生成歌词（Whisper 语音识别）
    - lyrics_text 有值 → CTC 强制对齐已有歌词，生成精准时间戳 LRC
    
    支持输入纯文本或带旧时间戳的 LRC（自动清除旧时间戳后重新对齐）
    生成器 yield: (逐词 LRC, 行级 LRC, 状态文字, 按钮状态更新)
    """
    import model_loader
    if audio_file is None:
        yield None, None, "❌ 请上传音频文件", gr.update(interactive=True)
        return
    # 无歌词 → 听写模式
    if not lyrics_text or not lyrics_text.strip():
        for lw, ll, s in _run_transcribe(audio_file):
            done = lw is not None or ll is not None
            yield lw, ll, s, gr.update(interactive=done)
        return
    # 有歌词 → 对齐模式
    start = time.time()
    model_ok = False
    for ready, msg in model_loader.wait_model_ready():
        yield None, None, f"⏱️ {_fmt_elapsed(start)} | {msg}", gr.update(interactive=not ready)
        if ready:
            model_ok = True
            break
    if not model_ok:
        # 模型加载失败（wait_model_ready 仅在失败时以 ready=False 结束）：
        # 必须在此中止，否则会落到 model_loader.model.align() 上 None 模型，
        # 抛出 'NoneType' object has no attribute 'align' 掩盖真实原因。
        yield None, None, f"⏱️ {_fmt_elapsed(start)} | ❌ Whisper 模型加载失败，终止对齐（详见 karaoke-convert 服务端日志）", gr.update(interactive=True)
        return
    try:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        lrc_word_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_aligned_word.lrc")
        lrc_line_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_aligned.lrc")
        clean_lines = [l.strip() for l in lyrics_text.strip().split("\n") if l.strip()]
        clean_text = "\n".join(clean_lines)
        if not clean_text:
            yield None, None, f"⏱️ {_fmt_elapsed(start)} | ❌ 歌词内容为空", gr.update(interactive=True)
            return
        print(f"🎯 对齐中（{len(clean_lines)} 句）: {audio_file}")
        yield None, None, f"⏱️ {_fmt_elapsed(start)} | 🎯 对齐 {len(clean_lines)} 句... 10%", gr.update(interactive=False)
        # stable-ts / faster-whisper 不同版本对 align() 的签名不同：部分版本不接受
        # word_timestamps，直接传会抛 “got unexpected keyword argument(s): word_timestamps”。
        # 这里按运行时签名条件化传参取最大兼容；word 时间戳本就是对齐的默认产物，
        # 缺失该形参也不影响逐词结果。
        import inspect
        align_fn = model_loader.model.align
        try:
            align_params = inspect.signature(align_fn).parameters
        except (ValueError, TypeError):
            align_params = {}
        align_kwargs = {"language": config.WHISPER_LANGUAGE}
        if "word_timestamps" in align_params:
            align_kwargs["word_timestamps"] = True
        # align 是阻塞调用且无进度回调：后台线程执行，主流程每 1.5s 心跳一次，
        # 长任务期间 SSE 客户端也能持续看到“🔤 正在对齐… NN%”的实时进度。
        result = None
        for hb_pct, hb_el in _heartbeat_blocking(lambda: align_fn(audio_file, clean_text, **align_kwargs), start):
            if hb_pct is not None:
                yield None, None, f"⏱️ {hb_el} | 🔤 正在对齐（{len(clean_lines)} 句）... {hb_pct}%", gr.update(interactive=False)
            else:
                result = hb_el
        if result is None:
            raise RuntimeError("Whisper 对齐未返回结果")
        # 逐词 LRC
        yield None, None, f"⏱️ {_fmt_elapsed(start)} | ✍️ 正在生成 LRC... 90%", gr.update(interactive=False)
        lrc_word_lines = []
        for seg in result.segments:
            text = seg.text.strip()
            if not text: continue
            st = seg.start
            ts = f"[{int(st)//60:02d}:{int(st)%60:02d}.{int((st%1)*100):02d}]"
            words = getattr(seg, "words", None)
            if words and len(words) > 1:
                parts = [ts]
                for w in words:
                    wt = w.start
                    parts.append(f"<{int(wt)//60:02d}:{int(wt)%60:02d}.{int((wt%1)*100):02d}>{w.word.strip()}")
                lrc_word_lines.append("".join(parts))
            else:
                lrc_word_lines.append(f"{ts}{text}")
        lrc_word = "\n".join(lrc_word_lines)
        with open(lrc_word_path, "w", encoding="utf-8") as f:
            f.write(lrc_word)
        # 行级 LRC（优先用模型自带 to_lrc()，无该方法则手动拼装，兼容 faster-whisper 原生 AlignmentResult）
        try:
            lrc_line = result.to_lrc()
        except AttributeError:
            lrc_line_lines = []
            for seg in result.segments:
                text = seg.text.strip()
                if not text:
                    continue
                st = seg.start
                ts = f"[{int(st)//60:02d}:{int(st)%60:02d}.{int((st%1)*100):02d}]"
                lrc_line_lines.append(f"{ts}{text}")
            lrc_line = "\n".join(lrc_line_lines)
        yield lrc_word, lrc_line, f"⏱️ {_fmt_elapsed(start)} | ✅ 对齐完成 100%: {lrc_line_path}", gr.update(interactive=True)
    except Exception as e:
        yield None, None, f"⏱️ {_fmt_elapsed(start)} | ❌ 对齐异常: {str(e)}", gr.update(interactive=True)


# ---- 一键处理 ----

def one_click_karaoke(audio_file):
    """[一键处理] 分离人声+伴奏 → 听写歌词 → 输出逐词 LRC + 行级 LRC"""
    if audio_file is None:
        yield None, None, None, None, "❌ 请上传音频文件", gr.update(interactive=True)
        return
    start = time.time()
    try:
        vocal_path = accomp_path = None
        # 第1步：分离人声+伴奏
        for vocal, accomp, status in _run_demucs(audio_file):
            yield vocal, accomp, None, None, status, gr.update(interactive=False)
            if vocal is not None: vocal_path = vocal
            if accomp is not None: accomp_path = accomp
        if vocal_path is None or not os.path.exists(vocal_path):
            yield None, None, None, None, f"⏱️ {_fmt_elapsed(start)} | ❌ 分离失败", gr.update(interactive=True)
            return
        # 第2步：听写歌词（逐词 + 行级）
        lrc_word_out = lrc_line_out = None
        for lrc_word, lrc_line, status in _run_transcribe(vocal_path):
            yield vocal_path, accomp_path, lrc_word, lrc_line, status, gr.update(interactive=False)
            if lrc_word is not None: lrc_word_out = lrc_word
            if lrc_line is not None: lrc_line_out = lrc_line
        final = "✅ 全部完成！" if lrc_word_out else "⚠️ 分离完成但听写未成功"
        yield (vocal_path, accomp_path, lrc_word_out, lrc_line_out,
               f"⏱️ {_fmt_elapsed(start)} | {final}", gr.update(interactive=True))
    except Exception as e:
        yield None, None, None, None, f"⏱️ {_fmt_elapsed(start)} | ❌ 一键处理异常: {str(e)}", gr.update(interactive=True)
