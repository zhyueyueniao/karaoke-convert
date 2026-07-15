"""MV转KTV视频：音频提取、LRC转ASS字幕(可配置样式)、多音轨视频合成"""
import os
import re
import shutil
import subprocess
import time
import gradio as gr
import config


# ======================== 字幕样式预设 ========================
# ASS颜色格式: &H{AA}{BB}{GG}{RR}  (AA=透明度00不透明80半透明, BGR反序)
# PrimaryColour = 已唱颜色, SecondaryColour = 未唱颜色

ASS_STYLE_PRESETS = {
    "经典KTV": {
        "label": "经典KTV（白字蓝边·底部居中）",
        "fontname": "Noto Sans CJK SC",
        "fontsize": 58,
        "primary_color": "&H00FFFFFF",   # 白色(已唱)
        "secondary_color": "&H00999999",  # 灰色(未唱)
        "outline_color": "&H000000FF",    # 蓝色描边
        "back_color": "&H80000000",       # 半透明黑阴影
        "bold": True, "italic": False,
        "outline": 3, "shadow": 1,
        "alignment": 2,  # 底部居中
        "margin_l": 60, "margin_r": 60, "margin_v": 50,
    },
    "暖橙风": {
        "label": "暖橙风（橙字白边·底部居中）",
        "fontname": "Noto Sans CJK SC",
        "fontsize": 55,
        "primary_color": "&H0000A5FF",   # 橙色(已唱)
        "secondary_color": "&H00505050",  # 深灰(未唱)
        "outline_color": "&H00FFFFFF",    # 白色描边
        "back_color": "&H80000000",
        "bold": True, "italic": False,
        "outline": 2, "shadow": 1,
        "alignment": 2,
        "margin_l": 60, "margin_r": 60, "margin_v": 50,
    },
    "霓虹绿": {
        "label": "霓虹绿（亮绿字黑边·底部居中）",
        "fontname": "Noto Sans CJK SC",
        "fontsize": 55,
        "primary_color": "&H0000FF00",   # 亮绿(已唱)
        "secondary_color": "&H00004000",  # 暗绿(未唱)
        "outline_color": "&H00000000",    # 黑色描边
        "back_color": "&H00000000",
        "bold": True, "italic": False,
        "outline": 2, "shadow": 0,
        "alignment": 2,
        "margin_l": 60, "margin_r": 60, "margin_v": 50,
    },
    "简约白": {
        "label": "简约白（白字黑边·底部居中·细体）",
        "fontname": "Noto Sans CJK SC",
        "fontsize": 50,
        "primary_color": "&H00FFFFFF",   # 白色(已唱)
        "secondary_color": "&H00808080",  # 灰色(未唱)
        "outline_color": "&H00000000",    # 黑色描边
        "back_color": "&H00000000",
        "bold": False, "italic": False,
        "outline": 2, "shadow": 0,
        "alignment": 2,
        "margin_l": 60, "margin_r": 60, "margin_v": 50,
    },
    "顶部悬浮": {
        "label": "顶部悬浮（白字黑边·顶部居中·不挡画面）",
        "fontname": "Noto Sans CJK SC",
        "fontsize": 52,
        "primary_color": "&H00FFFFFF",
        "secondary_color": "&H00999999",
        "outline_color": "&H00000000",
        "back_color": "&H80000000",
        "bold": True, "italic": False,
        "outline": 3, "shadow": 1,
        "alignment": 8,  # 顶部居中
        "margin_l": 60, "margin_r": 60, "margin_v": 40,
    },
}

# 下拉框选项：[(显示名, 内部key), ...]  显示名带括号说明，内部仍用 key 查找预设
PRESET_CHOICES = [(cfg["label"], key) for key, cfg in ASS_STYLE_PRESETS.items()]
# 兼容旧引用
PRESET_NAMES = list(ASS_STYLE_PRESETS.keys())


# ======================== 工具函数 ========================

def _fmt_elapsed(start: float) -> str:
    elapsed = int(time.time() - start)
    return f"{elapsed // 60:02d}:{elapsed % 60:02d}"


def _get_video_duration(video_path: str) -> float:
    """用 ffprobe 获取视频时长（秒）"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _format_ass_time(seconds: float) -> str:
    """秒数转 ASS 时间格式 H:MM:SS.cc (cc=厘秒=1/100秒)"""
    total_cs = int(seconds * 100)
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


# ======================== LRC 解析 ========================

def _parse_time_tag(tag: str) -> float:
    """解析 [mm:ss.xx] 或 <mm:ss.xx> 时间标签，返回秒数"""
    m = re.match(r"(\d+):(\d+)(?:\.(\d+))?", tag)
    if not m:
        return 0.0
    minutes = int(m.group(1))
    seconds = int(m.group(2))
    frac_str = m.group(3) or "0"
    # 补齐到2位
    frac_str = (frac_str + "00")[:2]
    return minutes * 60 + seconds + int(frac_str) / 100


def _parse_word_lrc(lrc_text: str):
    """解析逐词 LRC，返回 [(line_start, line_end, [(word, word_dur_cs), ...]), ...]
    
    逐词LRC格式: [mm:ss.xx]<mm:ss.xx>词1<mm:ss.xx>词2<mm:ss.xx>词3
    每个词的持续时间 = 下一个词开始时间 - 当前词开始时间 (厘秒)
    """
    lines_data = []
    all_timestamps = []  # 用于估算行结束时间

    for raw_line in lrc_text.strip().split("\n"):
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        # 提取行起始时间 [mm:ss.xx]
        line_time_match = re.match(r"\[(\d+:\d+(?:\.\d+)?)\]", raw_line)
        if not line_time_match:
            continue
        line_start = _parse_time_tag(line_time_match.group(1))
        content = raw_line[line_time_match.end():]

        # 提取逐词时间戳 <mm:ss.xx>
        # 格式: <mm:ss.xx>词1<mm:ss.xx>词2...
        word_pattern = r"<(\d+:\d+(?:\.\d+)?)>([^<]*)"
        word_matches = list(re.finditer(word_pattern, content))

        if word_matches:
            words = []
            for i, wm in enumerate(word_matches):
                word_start = _parse_time_tag(wm.group(1))
                word_text = wm.group(2).strip()
                if not word_text:
                    continue
                # 词持续时间 = 下一个词开始时间 - 当前词开始时间
                if i + 1 < len(word_matches):
                    next_start = _parse_time_tag(word_matches[i + 1].group(1))
                else:
                    # 行内最后一个词：估算持续1秒
                    next_start = word_start + 1.0
                word_dur_cs = max(1, int((next_start - word_start) * 100))
                words.append((word_text, word_dur_cs))
            if words:
                line_end = _parse_time_tag(word_matches[-1].group(1)) + words[-1][1] / 100
                lines_data.append((line_start, line_end, words))
                all_timestamps.append(line_start)
                all_timestamps.append(line_end)
        else:
            # 无逐词标签，整行作为一个块
            text = content.strip()
            if text:
                lines_data.append((line_start, line_start + 3.0, [(text, 300)]))
                all_timestamps.append(line_start)

    # 后处理：修正最后一个词的持续时间（如果太长，用下一行起始时间限制）
    for i, (ls, le, words) in enumerate(lines_data):
        if i + 1 < len(lines_data):
            next_line_start = lines_data[i + 1][0]
            if le > next_line_start:
                # 缩短最后一个词的持续时间
                last_word, last_dur = words[-1]
                new_total_cs = max(1, int((next_line_start - ls) * 100))
                used_cs = sum(d for _, d in words[:-1])
                words[-1] = (last_word, max(1, new_total_cs - used_cs))
                lines_data[i] = (ls, next_line_start, words)

    return lines_data


# ======================== LRC 转 ASS ========================

def _build_ass_style(style_config: dict) -> str:
    """生成 ASS 样式行"""
    s = style_config
    # ASS Style 格式:
    # Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,
    # Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,
    # BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
    bold = -1 if s["bold"] else 0
    italic = -1 if s["italic"] else 0
    return (f"Karaoke,{s['fontname']},{s['fontsize']},"
            f"{s['primary_color']},{s['secondary_color']},{s['outline_color']},{s['back_color']},"
            f"{bold},{italic},0,0,100,100,0,0,"
            f"1,{s['outline']},{s['shadow']},{s['alignment']},"
            f"{s['margin_l']},{s['margin_r']},{s['margin_v']},1")


def lrc_to_ass(lrc_text: str, style_preset: str = "经典KTV") -> str:
    """将逐词 LRC 转为 ASS 字幕文本（含卡拉OK \kf 渐变效果）
    
    Args:
        lrc_text: 逐词或行级 LRC 文本
        style_preset: ASS_STYLE_PRESETS 中的预设名称(内部key)或带说明的显示名(label)
    Returns:
        ASS 格式字幕文本
    """
    # 兼容传入显示名(label)：自动反查回内部 key
    if style_preset not in ASS_STYLE_PRESETS:
        for key, cfg in ASS_STYLE_PRESETS.items():
            if cfg.get("label") == style_preset:
                style_preset = key
                break
    style_config = ASS_STYLE_PRESETS.get(style_preset, ASS_STYLE_PRESETS["经典KTV"])
    lines_data = _parse_word_lrc(lrc_text)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: {_build_ass_style(style_config)}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, Effect, Text
"""

    events = []
    for line_start, line_end, words in lines_data:
        start_str = _format_ass_time(line_start)
        end_str = _format_ass_time(max(line_end, line_start + 0.3))
        # 构建 \kf 序列
        kf_parts = []
        for word_text, word_dur_cs in words:
            kf_parts.append(f"{{\\kf{word_dur_cs}}}{word_text}")
        text = "".join(kf_parts)
        events.append(f"Dialogue: 0,{start_str},{end_str},Karaoke,,0,0,0,{text}")

    return header + "\n".join(events) + "\n"


# ======================== 音频提取 ========================

def _run_extract_audio(video_file):
    """从视频提取音频，生成器 yield (audio_path, status)"""
    if video_file is None:
        yield None, "❌ 请上传视频文件"
        return
    yield None, "🎬 正在提取音频..."
    start = time.time()
    try:
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        audio_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_mv_audio.wav")
        cmd = ["ffmpeg", "-y", "-i", video_file, "-vn",
               "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", audio_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            yield None, f"⏱️ {_fmt_elapsed(start)} | ❌ 音频提取失败: {proc.stderr[-200:]}"
            return
        yield audio_path, f"⏱️ {_fmt_elapsed(start)} | ✅ 音频提取完成"
    except subprocess.TimeoutExpired:
        yield None, f"⏱️ {_fmt_elapsed(start)} | ❌ 音频提取超时"
    except Exception as e:
        yield None, f"⏱️ {_fmt_elapsed(start)} | ❌ 音频提取异常: {str(e)}"


# ======================== 视频合成 ========================

def _run_render_ktv(video_file, ass_path, original_audio, accomp_audio, output_path):
    """ffmpeg 合成多音轨 KTV 视频，生成器 yield (status)
    
    音轨1: 原始音频(人声+伴奏=原唱)
    音轨2: 纯伴奏(伴唱)
    视频烧录 ASS 字幕
    """
    start = time.time()
    try:
        total_duration = _get_video_duration(video_file)
        if total_duration <= 0:
            total_duration = 1.0

        cmd = [
            "ffmpeg", "-y",
            "-i", video_file,          # 0: 视频+原始音频
            "-i", accomp_audio,         # 1: 纯伴奏
            "-map", "0:v",              # 视频流
            "-map", "0:a",              # 音轨1: 原始音频(原唱)
            "-map", "1:a",              # 音轨2: 伴奏
            "-vf", f"subtitles='{ass_path}'",
            "-c:v", "libx264", "-crf", "20", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k",
            "-metadata:s:a:0", "title=原唱",
            "-metadata:s:a:0", "language=chi",
            "-metadata:s:a:1", "title=伴奏",
            "-metadata:s:a:1", "language=chi",
            "-progress", "pipe:1", "-nostats",
            output_path
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1)
        last_pct = -1
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_ms="):
                try:
                    out_ms = int(line.split("=")[1])
                    pct = int(out_ms / (total_duration * 1000) * 100)
                    pct = min(99, max(0, pct))
                    if pct != last_pct and pct % 2 == 0:
                        last_pct = pct
                        yield f"⏱️ {_fmt_elapsed(start)} | 🎬 渲染中... {pct}%"
                except (ValueError, ZeroDivisionError):
                    pass
            elif line.startswith("progress=end"):
                break

        proc.wait(timeout=3600)
        if proc.returncode != 0:
            stderr_tail = proc.stderr.read()[-500:] if proc.stderr else ""
            yield f"⏱️ {_fmt_elapsed(start)} | ❌ 渲染失败: {stderr_tail}"
            return

        if os.path.exists(output_path):
            yield f"⏱️ {_fmt_elapsed(start)} | ✅ KTV视频渲染完成: {output_path}"
        else:
            yield f"⏱️ {_fmt_elapsed(start)} | ❌ 输出文件未生成"
    except subprocess.TimeoutExpired:
        yield f"⏱️ {_fmt_elapsed(start)} | ❌ 渲染超时（超过1小时）"
    except Exception as e:
        yield f"⏱️ {_fmt_elapsed(start)} | ❌ 渲染异常: {str(e)}"


# ======================== 一键 MV 转 KTV ========================

def mv_to_ktv(video_file, lyrics_text, style_preset="经典KTV"):
    """[一键处理] MV视频 → 提取音频 → 分离人声伴奏 → 听写/对齐歌词 → 生成ASS → 合成KTV视频
    
    生成器 yield: (vocal_path, accomp_path, lrc_text, ass_text, video_path, status, btn_update)
    """
    from processing import _run_demucs, _run_transcribe, align_lyrics

    if video_file is None:
        yield None, None, None, None, None, "❌ 请上传MV视频文件", gr.update(interactive=True)
        return

    start = time.time()
    try:
        base_name = os.path.splitext(os.path.basename(video_file))[0]

        # ---- 第1步：提取音频 ----
        audio_path = None
        for path, status in _run_extract_audio(video_file):
            yield None, None, None, None, None, status, gr.update(interactive=False)
            if path:
                audio_path = path
        if audio_path is None or not os.path.exists(audio_path):
            yield None, None, None, None, None, \
                f"⏱️ {_fmt_elapsed(start)} | ❌ 音频提取失败，终止", gr.update(interactive=True)
            return

        # ---- 第2步：分离人声 + 伴奏 ----
        vocal_path = accomp_path = None
        for vocal, accomp, status in _run_demucs(audio_path):
            yield vocal, accomp, None, None, None, status, gr.update(interactive=False)
            if vocal: vocal_path = vocal
            if accomp: accomp_path = accomp
        if accomp_path is None or not os.path.exists(accomp_path):
            yield None, None, None, None, None, \
                f"⏱️ {_fmt_elapsed(start)} | ❌ 人声分离失败，终止", gr.update(interactive=True)
            return

        # ---- 第3步：歌词生成（有歌词→对齐，无歌词→听写）----
        lrc_word = None
        if lyrics_text and lyrics_text.strip():
            yield vocal_path, accomp_path, None, None, None, \
                f"⏱️ {_fmt_elapsed(start)} | 🎯 使用已有歌词进行对齐...", gr.update(interactive=False)
            for lw, ll, status, *_ in align_lyrics(vocal_path, lyrics_text):
                yield vocal_path, accomp_path, lw, None, None, status, gr.update(interactive=False)
                if lw: lrc_word = lw
        else:
            yield vocal_path, accomp_path, None, None, None, \
                f"⏱️ {_fmt_elapsed(start)} | 📝 无歌词文本，开始听写...", gr.update(interactive=False)
            for lw, ll, status in _run_transcribe(vocal_path):
                yield vocal_path, accomp_path, lw, None, None, status, gr.update(interactive=False)
                if lw: lrc_word = lw

        if not lrc_word:
            yield vocal_path, accomp_path, None, None, None, \
                f"⏱️ {_fmt_elapsed(start)} | ❌ 歌词生成失败，终止", gr.update(interactive=True)
            return

        # ---- 第4步：LRC 转 ASS ----
        yield vocal_path, accomp_path, lrc_word, None, None, \
            f"⏱️ {_fmt_elapsed(start)} | 📝 生成卡拉OK字幕(样式: {style_preset})...", gr.update(interactive=False)
        ass_text = lrc_to_ass(lrc_word, style_preset)
        ass_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_ktv.ass")
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_text)

        # ---- 第5步：合成多音轨 KTV 视频 ----
        output_video = os.path.join(config.OUTPUT_DIR, f"{base_name}_ktv.mp4")
        # 原始音频 = 提取的 MV 音频
        original_audio = audio_path
        for status in _run_render_ktv(video_file, ass_path, original_audio, accomp_path, output_video):
            yield vocal_path, accomp_path, lrc_word, ass_text, None, status, gr.update(interactive=False)

        if os.path.exists(output_video):
            yield vocal_path, accomp_path, lrc_word, ass_text, output_video, \
                f"⏱️ {_fmt_elapsed(start)} | ✅ KTV视频生成完成！", gr.update(interactive=True)
        else:
            yield vocal_path, accomp_path, lrc_word, ass_text, None, \
                f"⏱️ {_fmt_elapsed(start)} | ❌ 视频生成失败", gr.update(interactive=True)
    except Exception as e:
        yield None, None, None, None, None, \
            f"⏱️ {_fmt_elapsed(start)} | ❌ 一键处理异常: {str(e)}", gr.update(interactive=True)
