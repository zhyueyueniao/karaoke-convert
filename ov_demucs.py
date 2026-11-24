"""OpenVINO GPU hybrid 管线：demucs 官方 torch 前后处理 + OV GPU 主干。

背景（2026-11-24 深挖结论，取代旧 adowu ONNX 整模型方案）：
- adowu/htdemucs-onnx 整模型在 GPU 上慢（ISTFT 两个 ConvTranspose 各 1.06s，占 80%）
  且 fp16 NaN（NaN 产生于图内 STFT 链路，污染全网）——不可用；
- 改用 demucs 官方 PyTorch 拆分：CPU 做 STFT/归一化/mask/ISTFT（调 model._spec/_mask/_ispec，
  语义与官方 forward 逐位一致），GPU 只跑两个分支主干：
  th.jit.trace(check_trace=False) → ov.convert_model → htdemucs_bb.xml；
- 实测（Core Ultra 225H iGPU）：fp16 0.45s/段（旧 OV CPU 全模型 1.5s 的 3.3 倍，
  rel_err 0.43%，无 NaN）；fp32 0.98s（diff 7e-7，数值完美）。
- 精度切换：/app/output/ov_demucs_precision.txt 写 fp16 / fp32（缺省 fp16），改完即时生效。

分段策略与 demucs 官方 split 一致：按训练段长 343980 采样（≈7.8s）切块，
段间 overlap 用线性 ramp 淡入淡出 overlap-add，消除段边界伪影。

模型支持（2026-11-24）：仅 HTDemucs 架构（htdemucs / htdemucs_ft，读 config.DEMUCS_MODEL
运行时切换，torch 实例与 GPU 主干产物按模型各缓存一份，产物在 /app/output/{model}_bb.xml）。
mdx 系列是经典 Demucs（无时间分支），不支持本管线——调用方（processing.py）应先
用 is_supported_model() 判断并回退 demucs CLI（CPU）。
"""

import os
import subprocess
import threading

import numpy as np
import openvino as ov

OUTPUT_DIR = "/app/output"
PRECISION_FILE = "/app/output/ov_demucs_precision.txt"
SEGMENT = 343980  # htdemucs 训练段长（采样点，44.1kHz）
SR = 44100

# 支持的模型清单与 config.OV_HYBRID_SUPPORTED_DEMUCS 保持一致（兜底自持一份，
# 避免独立脚本化运行 ov_demucs 时依赖 config）。
SUPPORTED_MODELS = ("htdemucs", "htdemucs_ft")

_torch_models = {}          # 模型名 -> torch 实例（htdemucs 与 htdemucs_ft 可共存）
_torch_lock = threading.Lock()
_compiled = None            # 当前 (model, precision) 的 CompiledModel
_compiled_key = None        # (model, precision, mtime)
_compile_lock = threading.Lock()


def is_supported_model(name: str) -> bool:
    """该模型是否支持 OpenVINO GPU hybrid（仅 HTDemucs 架构）。"""
    return (name or "").strip() in SUPPORTED_MODELS


def _current_model_name() -> str:
    """读运行时切换的 DEMUCS_MODEL（config 持久化优先）；不支持的模型回退 htdemucs。"""
    try:
        import config
        name = (getattr(config, "DEMUCS_MODEL", "") or "").strip()
        if name in SUPPORTED_MODELS:
            return name
    except Exception:
        pass
    return "htdemucs"


def _bb_paths(model: str) -> tuple[str, str]:
    """每个模型的主干 XML/bin 路径（模型名已白名单校验，直接拼接安全）。"""
    return (f"{OUTPUT_DIR}/{model}_bb.xml", f"{OUTPUT_DIR}/{model}_bb.bin")


# ================= torch 侧（demucs 官方前后处理） =================

def _get_torch_model(name: str | None = None):
    """懒加载 demucs 模型（BagOfModels 取 models[0]），按名缓存进程内单例。"""
    name = name or _current_model_name()
    m = _torch_models.get(name)
    if m is None:
        with _torch_lock:
            m = _torch_models.get(name)
            if m is None:
                import torch
                torch.set_num_threads(8)
                import demucs.pretrained as pre
                mm = pre.get_model(name)
                if hasattr(mm, "models"):
                    mm = mm.models[0]
                mm.eval()
                m = mm
                _torch_models[name] = m
    return m


def _build_backbone(m):
    """复刻 HTDemucs.forward 两个分支主干：输入归一化后的 x/xt + 统计量，
    输出反归一化幅度 mask 与时间支波形。与官方 forward 逐位一致（已数值验证）。"""
    import torch as th
    from einops import rearrange

    class Backbone(th.nn.Module):
        def __init__(self, mm):
            super().__init__()
            self.mm = mm

        def forward(self, x, xt, std, mean, stdt, meant):
            m = self.mm
            B, C, Fq, T = x.shape
            length = xt.shape[-1]
            saved, saved_t, lengths, lengths_t = [], [], [], []
            for idx, encode in enumerate(m.encoder):
                lengths.append(x.shape[-1])
                inject = None
                if idx < len(m.tencoder):
                    lengths_t.append(xt.shape[-1])
                    tenc = m.tencoder[idx]
                    xt = tenc(xt)
                    if not tenc.empty:
                        saved_t.append(xt)
                    else:
                        inject = xt
                x = encode(x, inject)
                if idx == 0 and m.freq_emb is not None:
                    frs = th.arange(x.shape[-2], device=x.device)
                    emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                    x = x + m.freq_emb_scale * emb
                saved.append(x)
            if m.crosstransformer:
                if m.bottom_channels:
                    b, c, f, t = x.shape
                    x = rearrange(x, "b c f t-> b c (f t)")
                    x = m.channel_upsampler(x)
                    x = rearrange(x, "b c (f t)-> b c f t", f=f)
                    xt = m.channel_upsampler_t(xt)
                x, xt = m.crosstransformer(x, xt)
                if m.bottom_channels:
                    x = rearrange(x, "b c f t-> b c (f t)")
                    x = m.channel_downsampler(x)
                    x = rearrange(x, "b c (f t)-> b c f t", f=f)
                    xt = m.channel_downsampler_t(xt)
            for idx, decode in enumerate(m.decoder):
                skip = saved.pop(-1)
                x, pre = decode(x, skip, lengths.pop(-1))
                offset = m.depth - len(m.tdecoder)
                if idx >= offset:
                    tdec = m.tdecoder[idx - offset]
                    length_t = lengths_t.pop(-1)
                    if tdec.empty:
                        pre = pre[:, :, 0]
                        xt, _ = tdec(pre, None, length_t)
                    else:
                        skip = saved_t.pop(-1)
                        xt, _ = tdec(xt, skip, length_t)
            S = len(m.sources)
            x = x.view(B, S, -1, Fq, T)
            x = x * std[:, None] + mean[:, None]
            xt = xt.view(B, S, -1, length)
            xt = xt * stdt[:, None] + meant[:, None]
            return x, xt

    return Backbone(m)


def ensure_bb_model(model: str | None = None):
    """主干 XML 缺失时自愈：jit.trace(check_trace=False) → convert_model → serialize。
    每个模型一份产物（htdemucs_bb.xml / htdemucs_ft_bb.xml）。
    （需要 torch + demucs + 权重缓存，容器内均已具备。）"""
    name = model or _current_model_name()
    bb_xml, bb_bin = _bb_paths(name)
    if os.path.exists(bb_xml) and os.path.exists(bb_bin):
        return
    import torch as th
    m = _get_torch_model(name)
    bb = _build_backbone(m)
    bb.eval()
    Fq, T = 2048, 336  # 343980 段对应的谱形状
    x = th.zeros(1, 4, Fq, T)
    xt = th.zeros(1, 2, SEGMENT)
    std = th.zeros(1, 1, 1, 1)
    mean = th.zeros(1, 1, 1, 1)
    stdt = th.zeros(1, 1, 1, 1)
    meant = th.zeros(1, 1, 1, 1)
    jit_bb = th.jit.trace(bb, (x, xt, std, mean, stdt, meant), check_trace=False)
    ov_model = ov.convert_model(jit_bb)
    ov.serialize(ov_model, bb_xml)


# ================= OpenVINO GPU 主干 =================

def _current_precision():
    """读精度开关文件（fp16/fp32），缺省 fp16。"""
    try:
        with open(PRECISION_FILE) as f:
            p = f.read().strip().lower()
        return p if p in ("fp16", "fp32") else "fp16"
    except Exception:
        return "fp16"


def get_compiled(precision: str | None = None, model: str | None = None):
    """进程内复用 CompiledModel；模型或精度文件变更时自动重编译。"""
    global _compiled, _compiled_key
    name = model or _current_model_name()
    p = (precision or _current_precision()).lower()
    try:
        mtime = os.stat(PRECISION_FILE).st_mtime
    except Exception:
        mtime = 0.0
    key = (name, p, mtime)
    if _compiled is not None and _compiled_key == key:
        return _compiled
    with _compile_lock:
        if _compiled is not None and _compiled_key == key:
            return _compiled
        ensure_bb_model(name)
        bb_xml, _ = _bb_paths(name)
        core = ov.Core()
        ov_model = core.read_model(model=bb_xml)
        cfg = {} if p == "fp16" else {"INFERENCE_PRECISION_HINT": "f32"}
        _compiled = core.compile_model(model=ov_model, device_name="GPU", config=cfg)
        _compiled_key = key
        print(f"[ov_demucs] GPU 主干已编译 model={name} precision={p}", flush=True)
    return _compiled


# ================= 音频 IO =================

def read_stereo_f32(audio_file: str) -> np.ndarray:
    """ffmpeg 解码任意音频为 44.1kHz 立体声 float32 [-1,1]，返回 (1, 2, N)。"""
    cmd = ["ffmpeg", "-v", "error", "-i", audio_file,
           "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    raw = np.frombuffer(p.stdout, dtype=np.float32)
    if raw.size == 0:
        raise RuntimeError("ffmpeg 解码失败: " + p.stderr.decode(errors="ignore")[-200:])
    st = raw.reshape(-1, 2).T.copy()  # (2, N)
    return st[None, ...]  # (1, 2, N)


def write_stereo_mp3(stereo: np.ndarray, path: str, bitrate: str = "192k"):
    """(2, N) float32 → mp3（libmp3lame）。"""
    pcm = np.ascontiguousarray(stereo.T).reshape(-1)
    cmd = ["ffmpeg", "-v", "error", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-",
           "-codec:a", "libmp3lame", "-b:a", bitrate, "-y", path]
    p = subprocess.run(cmd, input=pcm.tobytes(), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg 编码失败: " + p.stderr.decode(errors="ignore")[-200:])


# ================= 分离主流程 =================

def separate(mix: np.ndarray, overlap: float = 0.1, precision: str | None = None):
    """生成器：yield (已完成段数, 总段数)，结束时 return stems (1, 4, 2, N)。

    stems 顺序 [drums, bass, other, vocals]；伴奏 = drums + bass + other。
    hybrid：CPU torch 做 STFT/归一化/mask/ISTFT，GPU 跑主干。
    """
    import torch as th

    name = _current_model_name()
    model = _get_torch_model(name)
    compiled = get_compiled(precision, model=name)
    total = mix.shape[-1]
    if total == 0:
        raise RuntimeError("解码得到空音频")

    # 段起点：按 overlap 步进，最后补一段右对齐保证覆盖尾部
    step = max(1, int(SEGMENT * (1 - overlap)))
    starts = list(range(0, max(total - SEGMENT, 0) + 1, step))
    if starts[-1] + SEGMENT < total:
        starts.append(total - SEGMENT)
    n = len(starts)
    fade = max(1, int(SEGMENT * overlap / 2))

    out = np.zeros((1, 4, 2, total), np.float32)
    weight = np.zeros(total, np.float32)

    with th.no_grad():
        for i, s in enumerate(starts):
            seg = mix[..., s:s + SEGMENT]
            pad = SEGMENT - seg.shape[-1]
            if pad > 0:
                seg = np.pad(seg, ((0, 0), (0, 0), (0, pad)))
            seg_t = th.from_numpy(np.ascontiguousarray(seg))  # (1,2,SEGMENT)

            # --- CPU：官方 STFT + 归一化 ---
            z = model._spec(seg_t)                       # [1,2,2048,336] complex
            mag = model._magnitude(z)
            mean = mag.mean(dim=(1, 2, 3), keepdim=True)
            std = mag.std(dim=(1, 2, 3), keepdim=True)
            x = (mag - mean) / (1e-5 + std)
            meant = seg_t.mean(dim=(1, 2), keepdim=True)
            stdt = seg_t.std(dim=(1, 2), keepdim=True)
            xt = (seg_t - meant) / (1e-5 + stdt)

            # --- GPU：两分支主干 ---
            args = [x.numpy(), xt.numpy(), std.numpy(), mean.numpy(),
                    stdt.numpy(), meant.numpy()]
            o = compiled(args)

            # --- CPU：官方 mask + ISTFT + 时间支 ---
            mask = th.from_numpy(np.asarray(o[0]).astype(np.float32))
            xt_den = th.from_numpy(np.asarray(o[1]).astype(np.float32))
            zout = model._mask(z, mask)
            x_ispec = model._ispec(zout, SEGMENT)
            stems = (xt_den + x_ispec).numpy()           # (1,4,2,SEGMENT)

            w = np.ones(SEGMENT, np.float32)
            if i > 0:
                w[:fade] = np.linspace(0.0, 1.0, fade)
            if s + SEGMENT < total:
                w[-fade:] = np.linspace(1.0, 0.0, fade)

            end = min(s + SEGMENT, total)
            seglen = end - s
            out[..., s:end] += stems[..., :seglen] * w[:seglen]
            weight[s:end] += w[:seglen]
            yield i + 1, n

    weight = np.maximum(weight, 1e-6)
    out /= weight
    return out


def split_accompaniment(stems: np.ndarray) -> np.ndarray:
    """(1,4,2,N) → 伴奏 (2,N) = drums + bass + other（即 non-vocals，等价 demucs --two-stems）。"""
    return stems[0, :3].sum(axis=0)


def split_vocals(stems: np.ndarray) -> np.ndarray:
    """(1,4,2,N) → 人声 (2,N)。"""
    return stems[0, 3]
