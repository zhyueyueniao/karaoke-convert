"""AI MV 生成核心模块

根据歌词意境自动生成视频：
  1. 歌词分段（复用 LRC 时间戳，按时长合并为 8-15s 段）
  2. 意境 → Prompt（规则关键词匹配场景库；可选外部 LLM 升级）
  3. 三层可插拔视频生成后端（自动降级链）：
       P0 外部视频生成 API（云端，质量最高）
       P1 LTX-Video 真文生视频（本地 Intel iGPU, OpenVINO GenAI）
       P2 SD/SDXL 文生图 + FFmpeg 运动合成（本地兜底，100% 出片）
  4. 片段时长匹配 + xfade 交叉淡化拼接 + 全局调色 + QSV 硬编

所有产物写入 config.OUTPUT_DIR。主编排器 generate_aimv 为生成器，逐段 yield 进度字符串。
"""
import os
import re
import json
import math
import time
import shutil
import threading
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import requests

from config import (
    OUTPUT_DIR, HAS_QSV, T2V_DEVICE,
    AVAILABLE_T2V_BACKENDS, T2V_BACKEND_PRIORITY,
    EXT_VIDEO_API_PROVIDER, EXT_VIDEO_API_KEY, EXT_VIDEO_API_BASE_URL, EXT_VIDEO_API_ENABLED,
    PROMPT_LLM_PROVIDER, PROMPT_LLM_API_KEY, PROMPT_LLM_BASE_URL, PROMPT_LLM_MODEL, PROMPT_LLM_ENABLED,
    STYLE_PRESETS, DEFAULT_STYLE, SCENE_KEYWORDS, SCENE_DEFAULT,
    LTXVIDEO_OV_DIR, SD15_OV_DIR, SDXL_OV_DIR, AVAILABLE_SD_MODELS, DEFAULT_AIMV_MODEL,
    AIMV_RESOLUTION, AIMV_SEG_MIN, AIMV_SEG_MAX, AIMV_XFADE, AIMV_LTX_CLIP_SEC, AIMV_TIMEOUT_PER_CLIP,
)
# 复用现有 LRC 逐词解析
from video_processing import _parse_word_lrc

FPS = 25


# ============================================================
#  数据结构
# ============================================================
@dataclass
class LyricSegment:
    """歌词分段数据"""
    index: int
    start_time: float          # 段起始时间戳（秒）
    end_time: float            # 段结束时间戳（秒）
    duration: float            # end - start
    text: str                  # 该段歌词文本
    prompt: str = ""           # 生成的画面描述 prompt
    clip_path: Optional[str] = None   # 生成的视频片段路径


# ============================================================
#  歌词分段
# ============================================================
def _group_plain_text(lyrics: str, song_duration: Optional[float]) -> list[LyricSegment]:
    """纯文本（无时间戳）按标点切句，沿歌曲时长均匀分布"""
    sentences = re.split(r"[。！？!?；;\n]+", lyrics)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [lyrics.strip() or "music"]
    total = song_duration or (len(sentences) * 11.0)
    n = len(sentences)
    seg_len = max(AIMV_SEG_MIN, total / n)
    segments = []
    t = 0.0
    for i, s in enumerate(sentences):
        end = total if i == n - 1 else min(total, t + seg_len)
        segments.append(LyricSegment(i, t, end, end - t, s))
        t = end
    return segments


def group_lyric_segments(lyrics: str, song_duration: Optional[float] = None) -> list[LyricSegment]:
    """把 LRC/纯文本歌词合并为 8-15s 的段，并为每段生成 prompt

    Args:
        lyrics: LRC 文本或纯歌词文本
        song_duration: 歌曲总时长（秒），用于修正最后一段结束时间
    """
    lines_data = _parse_word_lrc(lyrics) if lyrics and "[" in lyrics else []
    segments: list[LyricSegment] = []
    idx = 0

    if lines_data:
        cur_start = None
        cur_end = None
        cur_texts: list[str] = []
        for (ls, le, words) in lines_data:
            text = "".join(w for w, _ in words).strip()
            if not text:
                continue
            if cur_start is None:
                cur_start, cur_end = ls, le
                cur_texts = [text]
            elif (le - cur_start) <= AIMV_SEG_MAX:
                cur_end = le
                cur_texts.append(text)
            else:
                segments.append(LyricSegment(idx, cur_start, cur_end,
                                             round(cur_end - cur_start, 3), " ".join(cur_texts)))
                idx += 1
                cur_start, cur_end = ls, le
                cur_texts = [text]
        if cur_start is not None:
            segments.append(LyricSegment(idx, cur_start, cur_end,
                                         round(cur_end - cur_start, 3), " ".join(cur_texts)))
            idx += 1
    else:
        segments = _group_plain_text(lyrics, song_duration)

    # 修正最后一段结束时间为歌曲总时长（无则保持）
    if song_duration and segments:
        if segments[-1].end_time < song_duration:
            segments[-1].end_time = song_duration
            segments[-1].duration = round(song_duration - segments[-1].start_time, 3)

    # 为每段生成 prompt
    for seg in segments:
        seg.prompt = generate_prompt(seg.text)
    return segments


# ============================================================
#  Prompt 生成（规则关键词匹配 + 可选 LLM 升级）
# ============================================================
def _rule_based_prompt(text: str) -> str:
    """规则关键词匹配场景库，拼出画面描述"""
    matched = []
    for kw, desc in SCENE_KEYWORDS.items():
        if kw in text:
            matched.append(desc)
    if not matched:
        return SCENE_DEFAULT
    # 去重并限长，避免 prompt 过长
    seen = set()
    uniq = []
    for m in matched:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return ", ".join(uniq[:3])


def generate_prompt(text: str, style: str = DEFAULT_STYLE, use_llm: bool = None) -> str:
    """生成单段画面的完整 prompt：风格前缀 + 意境描述

    Args:
        text: 歌词文本
        style: STYLE_PRESETS 中的风格键
        use_llm: 是否用外部 LLM 升级写 prompt；None 时按 PROMPT_LLM_ENABLED 自动决定
    """
    style_prefix = STYLE_PRESETS.get(style, STYLE_PRESETS[DEFAULT_STYLE])
    scene = _rule_based_prompt(text)

    if use_llm is None:
        use_llm = PROMPT_LLM_ENABLED
    if use_llm:
        rich = _call_prompt_llm(text)
        if rich:
            return f"{style_prefix}, {rich}"

    return f"{style_prefix}, {scene}, lyrics theme: {text[:60]}"


def _call_prompt_llm(text: str) -> Optional[str]:
    """调用外部 LLM 把歌词意境扩展为英文画面描述，失败返回 None"""
    if not (PROMPT_LLM_PROVIDER and PROMPT_LLM_API_KEY):
        return None
    base = PROMPT_LLM_BASE_URL.rstrip("/") if PROMPT_LLM_BASE_URL else "https://api.openai.com/v1"
    model = PROMPT_LLM_MODEL or "gpt-4o-mini"
    sys = ("You are a cinematic visual director. Given a short lyric line, output a single concise "
           "English visual prompt (under 40 words) describing a concrete, atmospheric video scene that "
           "matches the mood. No explanation, no quotes.")
    try:
        resp = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {PROMPT_LLM_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": model, "temperature": 0.8,
                  "messages": [{"role": "system", "content": sys},
                               {"role": "user", "content": text[:200]}]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip().strip("\"'")
    except Exception as e:
        print(f"⚠️ LLM 写 prompt 失败，回退规则匹配: {e}", flush=True)
        return None


# ============================================================
#  FFmpeg 工具
# ============================================================
def _run_ffmpeg(args: list[str], timeout: int = 1800) -> tuple[bool, str]:
    """运行 ffmpeg，返回 (是否成功, 错误尾部文本)"""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "")[-500:]
            return False, err
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "FFmpeg 超时"
    except Exception as e:
        return False, str(e)


def match_duration(clip_path: str, target_duration: float, out_path: str,
                   resolution: tuple[int, int] = AIMV_RESOLUTION) -> bool:
    """把片段裁剪/循环到目标时长（通过 -t + -stream_loop 循环）"""
    w, h = resolution
    # 先探测实际时长
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", clip_path],
            capture_output=True, text=True, timeout=30)
        cur = float(probe.stdout.strip() or "0")
    except Exception:
        cur = 0.0
    if abs(cur - target_duration) < 0.05 or cur <= 0:
        # 时长已一致或无法探测：直接复制（重新编码统一编码参数）
        ok, _ = _run_ffmpeg(
            ["-i", clip_path, "-t", f"{target_duration:.3f}",
             "-vf", f"scale={w}:{h}", "-r", str(FPS), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", out_path])
        return ok
    if cur < target_duration:
        # 太短：循环到目标时长
        ok, _ = _run_ffmpeg(
            ["-stream_loop", "-1", "-i", clip_path, "-t", f"{target_duration:.3f}",
             "-vf", f"scale={w}:{h}", "-r", str(FPS), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", out_path])
        return ok
    # 太长：裁剪
    ok, _ = _run_ffmpeg(
        ["-i", clip_path, "-t", f"{target_duration:.3f}",
         "-vf", f"scale={w}:{h}", "-r", str(FPS), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", out_path])
    return ok


# ============================================================
#  可插拔视频生成后端
# ============================================================
class VideoBackend:
    """视频生成后端接口：为单段生成视频片段，失败抛异常以触发降级"""
    name = "base"

    def generate_clip(self, segment: LyricSegment, output_path: str,
                      style_prefix: str, resolution: tuple[int, int]) -> str:
        raise NotImplementedError


class APIBackend(VideoBackend):
    """P0 外部视频生成 API（云端）。provider 支持 openai 兼容视频接口；未配置则直接降级。"""
    name = "api"

    def generate_clip(self, segment, output_path, style_prefix, resolution):
        if not EXT_VIDEO_API_ENABLED:
            raise RuntimeError("外部视频 API 未配置（需 EXT_VIDEO_API_PROVIDER + EXT_VIDEO_API_KEY）")
        # OpenAI 兼容视频生成端点（部分 provider 支持 /videos/generations）
        base = EXT_VIDEO_API_BASE_URL.rstrip("/") if EXT_VIDEO_API_BASE_URL else "https://api.openai.com/v1"
        prompt = f"{style_prefix}, {segment.prompt}"
        w, h = resolution
        try:
            r = requests.post(
                f"{base}/videos/generations",
                headers={"Authorization": f"Bearer {EXT_VIDEO_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": EXT_VIDEO_API_PROVIDER, "prompt": prompt,
                      "duration": min(segment.duration, 10),
                      "width": w, "height": h, "n": 1},
                timeout=300,
            )
            r.raise_for_status()
            data = r.json()
            # 取视频 URL（兼容 {data:[{url}]} 或 {url}）
            url = ""
            if isinstance(data.get("data"), list) and data["data"]:
                url = data["data"][0].get("url", "")
            elif data.get("url"):
                url = data["url"]
            if not url:
                raise RuntimeError(f"API 未返回视频 URL: {data}")
            # 轮询/直接下载
            vr = requests.get(url, timeout=600)
            vr.raise_for_status()
            tmp = output_path + ".mp4"
            with open(tmp, "wb") as f:
                f.write(vr.content)
            if not match_duration(tmp, segment.duration, output_path, resolution):
                shutil.move(tmp, output_path)
            return output_path
        except Exception as e:
            raise RuntimeError(f"外部视频 API 调用失败: {e}")


class LTXVideoBackend(VideoBackend):
    """P1 LTX-Video 真文生视频（OpenVINO GenAI Text2VideoPipeline，本地 iGPU）"""
    name = "ltx-video"
    _pipe = None
    _lock = threading.Lock()

    @classmethod
    def _load(cls):
        if cls._pipe is not None:
            return cls._pipe
        with cls._lock:
            if cls._pipe is not None:
                return cls._pipe
            if not os.path.isdir(LTXVIDEO_OV_DIR) or not any(
                    f.endswith(".xml") for _, _, fs in os.walk(LTXVIDEO_OV_DIR) for f in fs):
                raise RuntimeError(f"LTX-Video IR 未就绪: {LTXVIDEO_OV_DIR}，请先在模型管理下载")
            from openvino_genai import Text2VideoPipeline
            cls._pipe = Text2VideoPipeline(LTXVIDEO_OV_DIR, T2V_DEVICE)
        return cls._pipe

    def generate_clip(self, segment, output_path, style_prefix, resolution):
        pipe = self._load()
        w, h = resolution
        prompt = f"{style_prefix}, {segment.prompt}"
        # 生成约 AIMV_LTX_CLIP_SEC 秒片段
        n_frames = int(AIMV_LTX_CLIP_SEC * FPS)
        try:
            result = pipe.generate(prompt, width=w, height=h, num_frames=n_frames,
                                   num_inference_steps=30)
        except TypeError:
            # 不同版本参数名差异
            result = pipe.generate(prompt, height=h, width=w, num_frames=n_frames,
                                   num_inference_steps=30)
        # 兼容返回类型为帧列表或视频对象
        frames = getattr(result, "frames", None)
        if frames is None and isinstance(result, (list, tuple)):
            frames = result
        if frames is None:
            raise RuntimeError("LTX-Video 生成结果无法解析")
        # 写出临时视频（PNG 序列 → FFmpeg 编码）
        tmp_dir = output_path + ".frames"
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            import numpy as np
            for i, fr in enumerate(frames):
                arr = np.array(fr) if not isinstance(fr, np.ndarray) else fr
                from PIL import Image
                Image.fromarray(arr).save(os.path.join(tmp_dir, f"{i:05d}.png"))
            raw = output_path + ".raw.mp4"
            ok, err = _run_ffmpeg(
                ["-framerate", str(FPS), "-i", os.path.join(tmp_dir, "%05d.png"),
                 "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", raw])
            if not ok:
                raise RuntimeError(f"LTX 帧编码失败: {err}")
            # 循环/裁剪到目标时长
            if not match_duration(raw, segment.duration, output_path, resolution):
                raise RuntimeError("LTX 时长匹配失败")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if os.path.exists(output_path + ".raw.mp4"):
                os.remove(output_path + ".raw.mp4")
        return output_path


class SDMotionBackend(VideoBackend):
    """P2 SD/SDXL 文生图 + FFmpeg zoompan 运动合成（本地兜底）"""
    name = "sd-motion"
    _pipes: dict[str, object] = {}
    _lock = threading.Lock()

    @classmethod
    def _load(cls, model: str):
        if model in cls._pipes:
            return cls._pipes[model]
        with cls._lock:
            if model in cls._pipes:
                return cls._pipes[model]
            if model == "sdxl":
                ov_dir = SDXL_OV_DIR
            else:
                ov_dir = SD15_OV_DIR
            if not os.path.isdir(ov_dir) or not any(
                    f.endswith(".xml") for _, _, fs in os.walk(ov_dir) for f in fs):
                raise RuntimeError(f"SD 模型 IR 未就绪: {ov_dir}，请先在模型管理下载")
            from openvino_genai import Text2ImagePipeline
            cls._pipes[model] = Text2ImagePipeline(ov_dir, T2V_DEVICE)
        return cls._pipes[model]

    def __init__(self, model: str = None):
        self.model = model or DEFAULT_AIMV_MODEL if DEFAULT_AIMV_MODEL in AVAILABLE_SD_MODELS else "sd15"

    def generate_clip(self, segment, output_path, style_prefix, resolution):
        pipe = self._load(self.model)
        w, h = resolution
        prompt = f"{style_prefix}, {segment.prompt}"
        try:
            result = pipe.generate(prompt, width=w, height=h, num_inference_steps=20)
        except TypeError:
            result = pipe.generate(prompt, height=h, width=w, num_inference_steps=20)
        # 取首张图（ImageBatch）
        img = result[0] if isinstance(result, (list, tuple)) else result
        keyframe = output_path + ".png"
        img.save(keyframe)
        # FFmpeg zoompan 运动合成 + 淡入淡出，时长精确匹配
        fps = FPS
        nframes = max(1, int(round(segment.duration * fps)))
        # 缩放确保偶数尺寸
        w2, h2 = (w // 2 * 2), (h // 2 * 2)
        vf = (f"scale={w2}:{h2},"
              f"zoompan=z='min(zoom+0.0018,1.12)':d={nframes}:s={w2}x{h2}:fps={fps}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',"
              f"fade=t=in:st=0:d=0.5,fade=t=out:st={max(0, segment.duration-0.5):.2f}:d=0.5,"
              f"format=yuv420p")
        ok, err = _run_ffmpeg(
            ["-loop", "1", "-i", keyframe, "-t", f"{segment.duration:.3f}",
             "-vf", vf, "-r", str(fps), "-pix_fmt", "yuv420p",
             "-c:v", "libx264", output_path])
        os.remove(keyframe)
        if not ok:
            raise RuntimeError(f"SD 运动合成失败: {err}")
        return output_path


# ============================================================
#  片段拼接（xfade 交叉淡化）+ 全局调色 + QSV 硬编
# ============================================================
def concat_with_xfade(clip_paths: list[str], out_path: str,
                      xfade: float = AIMV_XFADE,
                      resolution: tuple[int, int] = AIMV_RESOLUTION) -> bool:
    """用 xfade 交叉淡化拼接多段视频（统一分辨率/fps），失败时回退 concat 直接拼"""
    if not clip_paths:
        return False
    if len(clip_paths) == 1:
        shutil.copy(clip_paths[0], out_path)
        return True
    w, h = resolution
    # 统一每段为 yuv420p 同分辨率，便于 xfade
    norm = []
    for i, p in enumerate(clip_paths):
        np_ = f"{out_path}.norm{i}.mp4"
        ok, _ = _run_ffmpeg(
            ["-i", p, "-vf", f"scale={w//2*2}:{h//2*2},format=yuv420p,fps={FPS}",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", np_])
        if not ok:
            return False
        norm.append(np_)

    # 构建 xfade 滤镜链
    inf = []
    for i, p in enumerate(norm):
        inf.append(f"[{i}:v]")
    # 逐级两两 xfade
    prev = f"[0:v]"
    for i in range(1, len(norm)):
        outlbl = f"[v{i}]" if i < len(norm) - 1 else "[vcat]"
        inf.append(f"{prev}[{i}:v]xfade=transition=fade:duration={xfade}:offset=ITEMP{i}{outlbl}")
        prev = outlbl
    # offset 需在运行时按累计时长计算；为简化使用固定预估：每段时长近似（xfade 会自行对齐）
    # 用精确 offset 需探测每段时长；此处采用逐步 offset = seg_dur*(i) - xfade*i
    # 简化：统一用平均 offset 不可靠，故改为探测每段真实时长
    durations = [_probe_duration(p) for p in norm]
    offsets = []
    acc = durations[0]
    for i in range(1, len(durations)):
        acc += durations[i] - xfade
        offsets.append(acc)
    # 重建滤镜链（含精确 offset）
    filt = ""
    prev = "0:v"
    for i in range(1, len(norm)):
        outlbl = f"v{i}" if i < len(norm) - 1 else "vcat"
        filt += f"[{prev}][{i}:v]xfade=transition=fade:duration={xfade}:offset={offsets[i-1]:.3f}[{outlbl}];"
        prev = outlbl
    filt = filt.rstrip(";")

    filter_arg = filt
    # 构造输入
    inputs = []
    for p in norm:
        inputs += ["-i", p]
    ok, err = _run_ffmpeg(
        inputs + ["-filter_complex", filter_arg, "-map", "[vcat]",
                  "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path],
        timeout=1800)
    for p in norm:
        if os.path.exists(p):
            os.remove(p)
    if not ok:
        print(f"⚠️ xfade 拼接失败，回退 concat: {err}", flush=True)
        return _concat_simple(clip_paths, out_path, resolution)
    return True


def _probe_duration(path: str) -> float:
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30)
        return float(probe.stdout.strip() or "0")
    except Exception:
        return 0.0


def _concat_simple(clip_paths: list[str], out_path: str,
                   resolution: tuple[int, int] = AIMV_RESOLUTION) -> bool:
    """回退：直接 concat（无交叉淡化）"""
    w, h = resolution
    list_file = out_path + ".list.txt"
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    ok, _ = _run_ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", list_file,
         "-vf", f"scale={w//2*2}:{h//2*2},format=yuv420p,fps={FPS}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path],
        timeout=1800)
    if os.path.exists(list_file):
        os.remove(list_file)
    return ok


def color_grade(input_path: str, out_path: str) -> bool:
    """全局调色统一：轻微提亮 + 暖色平衡，统一色调连贯性"""
    vf = "eq=brightness=0.02:saturation=1.08:contrast=1.03,colormatrix=bt601:bt709"
    ok, err = _run_ffmpeg(
        ["-i", input_path, "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path],
        timeout=900)
    if not ok:
        print(f"⚠️ 调色失败，使用原片: {err}", flush=True)
        shutil.copy(input_path, out_path)
    return True


def encode_final(input_path: str, out_path: str) -> bool:
    """最终编码：优先 h264_qsv 硬编，失败回退 libx264"""
    if HAS_QSV:
        ok, err = _run_ffmpeg(
            ["-i", input_path, "-c:v", "h264_qsv", "-preset", "medium",
             "-pix_fmt", "nv12", out_path], timeout=900)
        if ok:
            return True
        print(f"⚠️ QSV 编码失败，回退 libx264: {err}", flush=True)
    ok, _ = _run_ffmpeg(
        ["-i", input_path, "-c:v", "libx264", "-preset", "medium",
         "-pix_fmt", "yuv420p", out_path], timeout=900)
    return ok


# ============================================================
#  主编排器（自动降级链）
# ============================================================
def resolve_aimv_priority(preferred: str = "auto") -> list[str]:
    """把 UI/API 的偏好映射为具体后端优先级链"""
    mapping = {
        "auto": T2V_BACKEND_PRIORITY,
        "api": ["api", "ltx-video", "sd-motion"],
        "ltx-video": ["ltx-video", "sd-motion"],
        "sd-motion": ["sd-motion"],
    }
    return mapping.get(preferred, T2V_BACKEND_PRIORITY)


def _build_backend_chain(priority: list[str], local_model: str) -> list[VideoBackend]:
    """按优先级构造可用后端实例列表

    local_model: 用户在模型管理 Tab 选择的本地生成模型（ltx-video / sd15 / sdxl）。
      - 选择 ltx-video → 启用 LTX 后端
      - 选择 sd15/sdxl → SD 后端使用该模型
      - P0 外部 API 若已配置仍排在最前
    """
    sd_model = local_model if local_model in AVAILABLE_SD_MODELS else "sd15"
    chain: list[VideoBackend] = []
    for name in priority:
        if name == "api":
            if EXT_VIDEO_API_ENABLED:
                chain.append(APIBackend())
        elif name == "ltx-video":
            chain.append(LTXVideoBackend())
        elif name == "sd-motion":
            chain.append(SDMotionBackend(sd_model))
    # 兜底：至少保证 SD 兜底存在
    if not chain:
        chain.append(SDMotionBackend(sd_model))
    return chain


def generate_aimv(audio_path: str, lyrics: str, song_duration: Optional[float],
                  backend_priority: Optional[list[str]] = None,
                  style: str = DEFAULT_STYLE,
                  sd_model: str = None,
                  burn_subtitles: bool = False,
                  lrc_text: str = None,
                  base_name: str = None,
                  state: dict = None) -> "Generator[str, None, None]":
    """AI MV 生成主流程（生成器，逐段 yield 进度字符串）

    Args:
        audio_path: 输入音频路径（用于命名/可选烧录字幕）
        lyrics: LRC 或纯歌词文本
        song_duration: 歌曲总时长（秒）
        backend_priority: 后端优先级链，默认 T2V_BACKEND_PRIORITY（API→LTX→SD）
        style: 风格键
        sd_model: SD 兜底模型（sd15/sdxl）
        burn_subtitles: 是否在最终视频烧录 KTV 字幕
        lrc_text: 用于烧录字幕的 LRC（默认用 lyrics）
        base_name: 输出文件名基础（默认取音频名）
        state: 可变字典，结束时写入 {'output_path':..., 'status':...}
    """
    local_model = sd_model or DEFAULT_AIMV_MODEL
    chain = _build_backend_chain(backend_priority or T2V_BACKEND_PRIORITY, local_model)
    chain_names = " → ".join(b.name for b in chain)
    yield f"🎬 AI MV 后端链: {chain_names}"

    base_name = base_name or os.path.splitext(os.path.basename(audio_path or "aimv"))[0]
    work_dir = os.path.join(OUTPUT_DIR, f"{base_name}_aimv_tmp")
    os.makedirs(work_dir, exist_ok=True)

    # 1) 歌词分段
    yield "✂️ 歌词分段中..."
    segments = group_lyric_segments(lyrics, song_duration)
    if not segments:
        yield "❌ 无法解析歌词，请检查输入"
        if state is not None:
            state["status"] = "error"
        return
    yield f"✂️ 分为 {len(segments)} 段（风格: {style}）"

    # 2) 逐段生成（带降级）
    style_prefix = STYLE_PRESETS.get(style, STYLE_PRESETS[DEFAULT_STYLE])
    clip_paths: list[str] = []
    for seg in segments:
        seg_done = False
        for bi, backend in enumerate(chain):
            try:
                yield (f"🎥 第 {seg.index+1}/{len(segments)} 段 "
                       f"[{backend.name}] {seg.text[:20]}...")
                out = os.path.join(work_dir, f"seg{seg.index}.mp4")
                start = time.time()
                clip = backend.generate_clip(seg, out, style_prefix, AIMV_RESOLUTION)
                # 时长精确匹配
                matched = os.path.join(work_dir, f"seg{seg.index}.m.mp4")
                if not match_duration(clip, seg.duration, matched, AIMV_RESOLUTION):
                    matched = clip
                seg.clip_path = matched
                clip_paths.append(matched)
                cost = time.time() - start
                yield (f"✅ 第 {seg.index+1} 段完成（{backend.name}，"
                       f"{cost/60:.1f}分钟，{seg.duration:.1f}s）")
                seg_done = True
                break
            except Exception as e:
                yield (f"⚠️ 第 {seg.index+1} 段 [{backend.name}] 失败: {str(e)[:120]}，"
                       f"降级到下一后端")
                if bi == len(chain) - 1:
                    yield f"❌ 第 {seg.index+1} 段所有后端均失败，跳过"
        if not seg_done:
            # 该段彻底失败，用纯色占位避免中断（最低保障：黑场 + 静音）
            placeholder = os.path.join(work_dir, f"seg{seg.index}.mp4")
            ok, _ = _run_ffmpeg(
                ["-f", "lavfi", "-i",
                 f"color=c=black:s={AIMV_RESOLUTION[0]}x{AIMV_RESOLUTION[1]}:d={seg.duration:.3f}:r={FPS}",
                 "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", placeholder])
            if ok:
                clip_paths.append(placeholder)
                seg.clip_path = placeholder

    if not clip_paths:
        yield "❌ 未生成任何视频片段"
        if state is not None:
            state["status"] = "error"
        return

    # 3) 拼接（xfade）
    yield "🧩 片段拼接（交叉淡化）..."
    concat_path = os.path.join(work_dir, "concat.mp4")
    if not concat_with_xfade(clip_paths, concat_path):
        yield "❌ 拼接失败"
        if state is not None:
            state["status"] = "error"
        return

    # 4) 全局调色
    yield "🎨 全局调色统一..."
    graded = os.path.join(work_dir, "graded.mp4")
    color_grade(concat_path, graded)

    # 5) 最终编码（QSV 优先）→ 静音合成片
    silent = os.path.join(work_dir, "final_silent.mp4")
    yield "📦 最终编码（QSV 硬编）..."
    if not encode_final(graded, silent):
        yield "❌ 最终编码失败"
        if state is not None:
            state["status"] = "error"
        return

    # 5.5) 复用原曲音频（AI MV 需带声音）
    final = os.path.join(OUTPUT_DIR, f"{base_name}_aimv.mp4")
    if audio_path and os.path.exists(audio_path):
        ok, err = _run_ffmpeg(
            ["-i", silent, "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-shortest", final])
        if not ok:
            yield f"⚠️ 音频复用失败，输出静音版: {err[:120]}"
            shutil.copy(silent, final)
    else:
        shutil.copy(silent, final)

    # 6) 可选烧录 KTV 字幕（单音轨 + subtitles 滤镜）
    if burn_subtitles:
        yield "🔤 烧录 KTV 字幕..."
        try:
            from video_processing import lrc_to_ass
            sub_lrc = lrc_text or lyrics
            ass = os.path.join(work_dir, "sub.ass")
            with open(ass, "w", encoding="utf-8") as f:
                f.write(lrc_to_ass(sub_lrc))
            burned = os.path.join(OUTPUT_DIR, f"{base_name}_aimv_ktv.mp4")
            ok, err = _run_ffmpeg(
                ["-i", final, "-vf", f"subtitles={ass}",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", burned],
                timeout=1800)
            if ok and os.path.exists(burned):
                final = burned
                yield "✅ 字幕烧录完成"
            else:
                yield f"⚠️ 字幕烧录失败，保留无字幕版本: {err[:120]}"
        except Exception as e:
            yield f"⚠️ 字幕烧录失败，保留无字幕版本: {str(e)[:120]}"

    yield f"✅ AI MV 生成完成: {final}"
    if state is not None:
        state["output_path"] = final
        state["status"] = "done"
