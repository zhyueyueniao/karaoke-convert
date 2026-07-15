"""模型下载、删除、缓存检测、UI 按钮包装器"""
import os
import re
import shutil
import subprocess
import time
import gradio as gr
from config import (HF_CACHE, HF_ENDPOINT, TORCH_CHECKPOINTS,TORCH_MARKERS, AVAILABLE_WHISPER, AVAILABLE_DEMUCS,
                    WHISPER_SIZES, DEMUCS_SIZES)


# ---- 工具函数 ----

def _fmt_elapsed(start: float) -> str:
    """[格式化] 将耗时秒数转为 MM:SS"""
    elapsed = int(time.time() - start)
    return f"{elapsed // 60:02d}:{elapsed % 60:02d}"

def _is_done(msg: str) -> bool:
    """[判断] 状态消息是否为终止状态（成功/失败/警告）"""
    return "✅" in msg or "❌" in msg or "⚠️" in msg

def _format_size(size_bytes: int) -> str:
    """[格式化] 字节数转可读大小（GB/MB）"""
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.1f}GB"
    return f"{size_bytes / 1024**2:.0f}MB"


# ---- 缓存状态检测 ----
# 所有 Demucs 模型（HTDemucs / MDX 系列）均通过 torch.hub 下载到：
#   /root/.cache/torch/hub/checkpoints/ （文件名是 UUID 哈希，无法按模型名识别）
# → 标记文件与模型同目录: {TORCH_MARKERS}/{model}.files

def is_whisper_cached(model_name: str) -> bool:
    """[检测] Whisper 模型是否已下载到 HuggingFace 缓存"""
    path = os.path.join(HF_CACHE, "models--Systran--faster-whisper-" + model_name.replace(".", "--"), "refs")
    return os.path.exists(path)

def is_demucs_cached(model_name: str) -> bool:
    """[检测] 检查 .files 记录文件是否存在"""
    return os.path.exists(os.path.join(TORCH_MARKERS, f"{model_name}.files"))


# ---- 缓存大小 ----

def get_whisper_cache_size(model_name: str) -> str:
    """[大小] 获取 Whisper 模型在 HuggingFace blobs 目录的实际磁盘占用"""
    cache_dir = os.path.join(HF_CACHE, "models--Systran--faster-whisper-" + model_name.replace(".", "--"), "blobs")
    if not os.path.isdir(cache_dir):
        return WHISPER_SIZES.get(model_name, "")
    total = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in os.listdir(cache_dir)
                if os.path.isfile(os.path.join(cache_dir, f)))
    return _format_size(total)

def get_demucs_cache_size(model_name: str) -> str:
    """[大小] 从 .files 记录直接读取文件大小（写入时已计算），无记录时返回预估值"""
    files_record = os.path.join(TORCH_MARKERS, f"{model_name}.files")
    if os.path.exists(files_record):
        total = 0
        for line in open(files_record).read().strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    total += int(parts[1])
                except ValueError:
                    pass
        if total > 0:
            return _format_size(total)
    return DEMUCS_SIZES.get(model_name, "")


# ---- 删除模型 ----

def delete_whisper_model(model_name: str):
    """[删除] 删除 Whisper 模型 HuggingFace 缓存目录（释放磁盘空间）"""
    if model_name not in AVAILABLE_WHISPER:
        yield f"❌ 无效模型: {model_name}"
        return
    cache_dir = os.path.join(HF_CACHE, "models--Systran--faster-whisper-" + model_name.replace(".", "--"))
    if not os.path.exists(cache_dir):
        yield f"⚠️ {model_name} 未缓存，无需删除"
        return
    yield f"🗑 删除 {model_name}..."
    try:
        shutil.rmtree(cache_dir)
        yield f"✅ {model_name} 已删除"
    except Exception as e:
        yield f"❌ 删除失败: {str(e)}"

def delete_demucs_model(model_name: str):
    """[删除] 删除 .files 记录 + UUID 模型文件"""
    if model_name not in AVAILABLE_DEMUCS:
        yield f"❌ 无效模型: {model_name}"
        return
    files_record = os.path.join(TORCH_MARKERS, f"{model_name}.files")
    if not os.path.exists(files_record):
        yield f"⚠️ 未找到 {model_name} 缓存"
        return
    yield f"🗑 删除 {model_name} 缓存..."
    for line in open(files_record).read().strip().split("\n"):
        f = line.strip().split()[0] if line.strip() else ""
        fp = os.path.join(TORCH_CHECKPOINTS, f)
        if f and os.path.isfile(fp):
            try:
                os.remove(fp)
            except OSError:
                pass
    os.remove(files_record)
    yield f"✅ {model_name} 已删除"


# ---- 下载模型 ----

def download_whisper_model(model_name: str, force: bool = False):
    """[下载] 从 HuggingFace 镜像站下载 faster-whisper 模型
    
    - force=False: 已缓存则跳过
    - force=True: 删除旧缓存后重新下载
    - 子进程带心跳线程，避免网络空闲时 UI 假死
    """
    if model_name not in AVAILABLE_WHISPER:
        yield f"❌ 无效模型: {model_name}"
        return
    repo_id = f"Systran/faster-whisper-{model_name}"
    if not force and is_whisper_cached(model_name):
        yield f"✅ {model_name} 已缓存，无需下载"
        return
    if force:
        yield f"🗑 清除旧缓存..."
        for _ in delete_whisper_model(model_name):
            pass
    yield f"📥 开始下载 {repo_id}..."
    start = time.time()
    try:
        cmd = """\
import os, time, threading
os.environ['HF_ENDPOINT'] = '{hf_endpoint}'
from huggingface_hub import snapshot_download
start = time.time()
stop = threading.Event()
def heartbeat():
    while not stop.wait(5):
        e = int(time.time() - start)
        msg = '[' + str(e//60).zfill(2) + ':' + str(e%60).zfill(2) + '] downloading...'
        print(msg, flush=True)
hb = threading.Thread(target=heartbeat, daemon=True)
hb.start()
try:
    print('connecting...', flush=True)
    snapshot_download('{repo_id}', cache_dir='{cache_dir}', resume_download=True)
    print('DONE', flush=True)
finally:
    stop.set()
    hb.join(timeout=1)
""".format(hf_endpoint=HF_ENDPOINT, repo_id=repo_id, cache_dir=HF_CACHE)
        proc = subprocess.Popen(["python3", "-u", "-c", cmd], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            line = line.strip()
            if not line: continue
            if "%" in line:
                m = re.search(r"(\d+)%", line)
                if m:
                    eta_m = re.search(r"(\d{1,2}:\d{2})", line)
                    eta = f" 预计 {eta_m.group(1)}" if eta_m else ""
                    yield f"⏱️ {_fmt_elapsed(start)} | 📥 下载 {model_name} ... {m.group(1)}%{eta}"
                else:
                    yield f"⏱️ {_fmt_elapsed(start)} | 📥 {line}"
            else:
                yield f"⏱️ {_fmt_elapsed(start)} | 📥 {line}"
        proc.wait(timeout=1800)
        elapsed = time.time() - start
        if proc.returncode != 0:
            yield f"⏱️ {_fmt_elapsed(start)} | ❌ 下载失败 (exit={proc.returncode})，请重试"
            return
        if elapsed < 10:
            yield f"⏱️ {_fmt_elapsed(start)} | ✅ {model_name} 模型就绪（缓存命中）"
        else:
            yield f"⏱️ {_fmt_elapsed(start)} | ✅ {model_name} 下载完成"
    except subprocess.TimeoutExpired:
        yield f"❌ 下载超时（30分钟），请检查网络后重试"
    except Exception as e:
        yield f"❌ 下载异常: {str(e)}"


def download_demucs_model(model_name: str, force: bool = False):
    """[下载] 通过 Demucs pretrained.get_model() 下载预训练模型
    
    - HTDemucs 系列：通过 torch hub 从 Facebook CDN 下载
    - MDX 系列：通过 HuggingFace/ONNX 下载
    - force=True: 清除标记后重新校验下载
    """
    if model_name not in AVAILABLE_DEMUCS:
        yield f"❌ 无效模型: {model_name}"
        return
    if not force and is_demucs_cached(model_name):
        yield f"✅ {model_name} 已缓存，无需下载"
        return
    # 下载前记录 checkpoints 已有文件，下载后 diff 确定新文件
    if force:
        yield f"🗑 清除旧缓存..."
        files_record = os.path.join(TORCH_MARKERS, f"{model_name}.files")
        print("清除旧缓存"+files_record,flush=True)
        if os.path.exists(files_record):
            for line in open(files_record).read().strip().split("\n"):
                f = line.strip().split()[0] if line.strip() else ""
                fp = os.path.join(TORCH_CHECKPOINTS, f)
                print("清除旧缓存"+fp,flush=True)
                if f and os.path.exists(fp):
                    try:
                        os.remove(fp)
                    except OSError as e:
                        print("清除旧缓存报错："+str(e),flush=True)
                        pass
            os.remove(files_record)
            print("删除旧缓存"+files_record,flush=True)
    before_files = set(os.listdir(TORCH_CHECKPOINTS)) if os.path.isdir(TORCH_CHECKPOINTS) else set()

    yield f"📥 开始下载 Demucs {model_name}..."
    start = time.time()
    try:
        cmd = """\
import time, threading
start = time.time()
stop = threading.Event()
def heartbeat():
    while not stop.wait(5):
        e = int(time.time() - start)
        msg = '[' + str(e//60).zfill(2) + ':' + str(e%60).zfill(2) + '] downloading...'
        print(msg, flush=True)
hb = threading.Thread(target=heartbeat, daemon=True)
hb.start()
try:
    print('connecting...', flush=True)
    from demucs.pretrained import get_model
    get_model('{model_name}')
    print('DONE', flush=True)
finally:
    stop.set()
    hb.join(timeout=1)
""".format(model_name=model_name)
        proc = subprocess.Popen(["python3", "-u", "-c", cmd], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        out_lines = []
        for line in proc.stdout:
            line = line.strip()
            out_lines.append(line)
            if not line: continue
            if "%|" in line or "%" in line:
                m = re.search(r"(\d+)%", line)
                yield f"⏱️ {_fmt_elapsed(start)} | 📥 下载 {model_name} ... {m.group(1)}%" if m else \
                     f"⏱️ {_fmt_elapsed(start)} | 📥 {line}"
            else:
                yield f"⏱️ {_fmt_elapsed(start)} | 📥 {line}"
        proc.wait(timeout=1800)
        elapsed = time.time() - start
        if proc.returncode == 0:
            os.makedirs(TORCH_MARKERS, exist_ok=True)
            after_files = set(os.listdir(TORCH_CHECKPOINTS)) if os.path.isdir(TORCH_CHECKPOINTS) else set()
            new_files = after_files - before_files
            # .files 格式: 每行 "UUID文件名 文件大小(字节)"，同时作为缓存标记
            lines = []
            for nf in sorted(new_files):
                fp = os.path.join(TORCH_CHECKPOINTS, nf)
                size = os.path.getsize(fp) if os.path.isfile(fp) else 0
                lines.append(f"{nf} {size}")
            with open(os.path.join(TORCH_MARKERS, f"{model_name}.files"), "w") as f:
                f.write("\n".join(lines))
            if force:
                if new_files:
                    yield f"⏱️ {_fmt_elapsed(start)} | ✅ {model_name} 重新下载完成"
                else:
                    yield f"⏱️ {_fmt_elapsed(start)} | ⚠️ {model_name} 未检测到新文件，可能已在其他位置缓存"
            else:
                if elapsed < 10:
                    yield f"⏱️ {_fmt_elapsed(start)} | ✅ {model_name} 就绪（缓存命中）"
                else:
                    yield f"⏱️ {_fmt_elapsed(start)} | ✅ {model_name} 下载完成"
        else:
            err_tail = "\n".join(out_lines[-5:]) if out_lines else "无输出"
            print(f"❌ Demucs 下载失败 (exit={proc.returncode})\n{err_tail}", flush=True)
            yield f"⏱️ {_fmt_elapsed(start)} | ❌ 下载失败: {err_tail}"
    except subprocess.TimeoutExpired:
        yield f"❌ 下载超时（30分钟），请检查网络后重试"
    except Exception as e:
        yield f"❌ 下载异常: {str(e)}"


def switch_demucs_model(model_name: str) -> str:
    """[运行时切换] 仅允许切换到已下载的 Demucs 模型，立即生效"""
    import config
    if model_name not in AVAILABLE_DEMUCS:
        gr.Warning(f"无效模型: {model_name}")
        return config.DEMUCS_MODEL
    if model_name == config.DEMUCS_MODEL:
        gr.Info(f"已在使用 {model_name}，无需切换")
        return model_name
    if not is_demucs_cached(model_name):
        gr.Warning(f"{model_name} 未下载，请在下方模型列表点击 📥下载")
        return config.DEMUCS_MODEL
    config.DEMUCS_MODEL = model_name
    config._save_model_pref("DEMUCS_MODEL", model_name)
    gr.Info(f"已切换到 {model_name}")
    return model_name


# ---- UI 按钮包装生成器（控制按钮禁用/显示状态 + 完成后更新模型大小显示） ----

def whisper_dl_btn(model_name):
    """[📥 下载] 未缓存模型的一键下载，完成后切换为 🗑+🔄 并更新大小"""
    for msg in download_whisper_model(model_name):
        done, ok = _is_done(msg), "✅" in msg
        size_update = gr.update(value=f"**{model_name}**  {get_whisper_cache_size(model_name)}") if ok else gr.update()
        yield msg, gr.update(interactive=done, visible=not ok), gr.update(interactive=done, visible=ok), gr.update(interactive=done, visible=ok), size_update

def whisper_redl_btns(model_name):
    """[🔄 重新下载] 已缓存模型的强制重下，下载按钮始终隐藏"""
    for msg in download_whisper_model(model_name, force=True):
        done, ok = _is_done(msg), "✅" in msg
        size_update = gr.update(value=f"**{model_name}**  {get_whisper_cache_size(model_name)}") if ok else gr.update()
        yield msg, gr.update(visible=False), gr.update(interactive=done, visible=True), gr.update(interactive=done, visible=True), size_update

def whisper_del_btns(model_name):
    """[🗑 删除] 已缓存模型的删除，完成后切换为 📥 下载按钮 + 恢复预估值"""
    for msg in delete_whisper_model(model_name):
        done, ok = _is_done(msg), "✅" in msg
        size_update = gr.update(value=f"**{model_name}**  {WHISPER_SIZES.get(model_name, '')}") if ok else gr.update()
        yield msg, gr.update(interactive=done, visible=ok), gr.update(interactive=done, visible=not ok), gr.update(interactive=done, visible=not ok), size_update

def demucs_dl_btn(model_name):
    """[📥 下载] Demucs 未缓存模型的一键下载"""
    for msg in download_demucs_model(model_name):
        done, ok = _is_done(msg), "✅" in msg
        size_update = gr.update(value=f"**{model_name}**  {get_demucs_cache_size(model_name)}") if ok else gr.update()
        yield msg, gr.update(interactive=done, visible=not ok), gr.update(interactive=done, visible=ok), gr.update(interactive=done, visible=ok), size_update

def demucs_redl_btns(model_name):
    """[🔄 重新下载] Demucs 已缓存模型的强制重下"""
    for msg in download_demucs_model(model_name, force=True):
        done, ok = _is_done(msg), "✅" in msg
        size_update = gr.update(value=f"**{model_name}**  {get_demucs_cache_size(model_name)}") if ok else gr.update()
        yield msg, gr.update(visible=False), gr.update(interactive=done, visible=True), gr.update(interactive=done, visible=True), size_update

def demucs_del_btns(model_name):
    """[🗑 删除] Demucs 已缓存模型的删除"""
    for msg in delete_demucs_model(model_name):
        done, ok = _is_done(msg), "✅" in msg
        size_update = gr.update(value=f"**{model_name}**  {DEMUCS_SIZES.get(model_name, '')}") if ok else gr.update()
        yield msg, gr.update(interactive=done, visible=ok), gr.update(interactive=done, visible=not ok), gr.update(interactive=done, visible=not ok), size_update
