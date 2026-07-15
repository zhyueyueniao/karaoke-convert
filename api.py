"""REST API - 异步任务模式 + SSE 流式推送

【轮询模式】提交后定时查状态（兼容旧调用方）
POST /submit/karaoke   → 一键处理（可选 lyrics 参数启用对齐）
POST /submit/separate  → 人声分离
POST /submit/align     → 歌词生成/对齐（无歌词听写，有歌词对齐）
POST /submit/mv2ktv    → MV转KTV视频
GET  /status/{job_id}  → 查询任务状态/进度
GET  /result/{job_id}/{filename} → 下载结果文件
GET  /jobs             → 列出所有任务

【SSE 流式模式】一次请求实时推送进度，无需轮询（推荐）
POST /stream/karaoke   → 一键处理（SSE）
POST /stream/separate  → 人声分离（SSE）
POST /stream/align     → 歌词生成/对齐（SSE）
POST /stream/mv2ktv    → MV转KTV视频（SSE）
  事件类型: started / progress / done / error
  done 事件含 job_id，可用 /result/{job_id}/{filename} 下载文件
"""
import os
import json
import uuid
import threading
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn
import config


app = FastAPI(title="卡拉OK工作站 API", version="2.0")
jobs: dict[str, dict] = {}
_lock = threading.Lock()

# 确保输出目录存在
os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def _update_job(job_id: str, **kwargs):
    with _lock:
        jobs[job_id].update(kwargs)


def _error_job(job_id: str, error: str):
    with _lock:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = error


# ======================== 后台处理 ========================

def _run_karaoke_job(job_id: str, audio_path: str, lyrics_text: str | None = None):
    """一键处理，可选模式：
    - lyrics_text=None: 分离人声 → 听写歌词（原有流程）
    - 传入 lyrics_text:  分离人声 → 听写歌词 → 对齐歌词（更精准的逐词/行级 LRC）
    """
    from processing import one_click_karaoke, align_lyrics
    try:
        # 第1步：分离 + 听写
        gen = one_click_karaoke(audio_path)
        vocal_path = None
        lrc_raw = lrc_line_raw = None
        for vocal, accomp, lrc, lrc_line, status, *_ in gen:
            _update_job(job_id, progress=status)
            if vocal:
                _update_job(job_id, vocal_path=vocal)
                vocal_path = vocal
            if accomp:
                _update_job(job_id, accomp_path=accomp)
            if lrc:
                lrc_raw = lrc
            if lrc_line:
                lrc_line_raw = lrc_line
        # 第2步（可选）：用已有歌词重新对齐，替换听写结果
        if lyrics_text and vocal_path and os.path.exists(vocal_path):
            _update_job(job_id, progress="🎯 对齐歌词中...")
            try:
                align_gen = align_lyrics(vocal_path, lyrics_text)
                for lrc_w, lrc_l, status, *_ in align_gen:
                    _update_job(job_id, progress=status)
                    if lrc_w:
                        lrc_raw = lrc_w
                        _update_job(job_id, lrc_word=lrc_w)
                    if lrc_l:
                        lrc_line_raw = lrc_l
                        _update_job(job_id, lrc_line=lrc_l)
            except Exception:
                pass  # 对齐失败保留听写原始结果
        # 确保听写结果也被记录（对齐可能跳过）
        if lrc_raw:
            _update_job(job_id, lrc_word=lrc_raw)
        if lrc_line_raw:
            _update_job(job_id, lrc_line=lrc_line_raw)
        _update_job(job_id, status="done")
    except Exception as e:
        _error_job(job_id, str(e))


def _run_separate_job(job_id: str, audio_path: str):
    from processing import _run_demucs
    try:
        gen = _run_demucs(audio_path)
        for vocal, accomp, status in gen:
            _update_job(job_id, progress=status)
            if vocal:
                _update_job(job_id, vocal_path=vocal)
            if accomp:
                _update_job(job_id, accomp_path=accomp)
        _update_job(job_id, status="done")
    except Exception as e:
        _error_job(job_id, str(e))


def _run_align_job(job_id: str, audio_path: str, lyrics_text: str | None = None):
    """歌词生成/对齐：无歌词→听写，有歌词→对齐"""
    from processing import align_lyrics
    try:
        gen = align_lyrics(audio_path, lyrics_text or "")
        for lrc_word, lrc_line, status, *_ in gen:
            _update_job(job_id, progress=status)
            if lrc_word:
                _update_job(job_id, lrc_word=lrc_word)
            if lrc_line:
                _update_job(job_id, lrc_line=lrc_line)
        _update_job(job_id, status="done")
    except Exception as e:
        _error_job(job_id, str(e))


def _run_mv2ktv_job(job_id: str, video_path: str, lyrics_text: str | None, style_preset: str):
    """MV转KTV视频：提取音频 → 分离 → 歌词 → ASS字幕 → 多音轨视频合成"""
    from video_processing import mv_to_ktv
    try:
        gen = mv_to_ktv(video_path, lyrics_text or "", style_preset)
        for vocal, accomp, lrc, ass_text, video_out, status, *_ in gen:
            _update_job(job_id, progress=status)
            if vocal:
                _update_job(job_id, vocal_path=vocal)
            if accomp:
                _update_job(job_id, accomp_path=accomp)
            if lrc:
                _update_job(job_id, lrc_word=lrc)
            if ass_text:
                _update_job(job_id, ass_subtitle=ass_text)
            if video_out:
                _update_job(job_id, video_path=video_out)
        _update_job(job_id, status="done")
    except Exception as e:
        _error_job(job_id, str(e))


# ======================== SSE 流式处理 ========================
# 一次请求全程实时推送进度，断线即任务终止。
# 内部仍创建 job_id 写入 jobs 字典，断线后可用 /status/{job_id} 查残余状态。
# done 事件返回 job_id，客户端可用 /result/{job_id}/{filename} 下载结果文件。

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 禁止 Nginx/反代缓冲，保证实时性
}


def _sse(event: str, data: dict) -> str:
    """构造一条 SSE 事件字符串"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _create_stream_job(job_type: str) -> str:
    """创建 SSE 任务的 job_id（不启动后台线程，任务在请求线程内同步执行）"""
    job_id = str(uuid.uuid4())[:12]
    with _lock:
        jobs[job_id] = {"id": job_id, "type": job_type, "status": "running", "progress": ""}
    return job_id


def _stream_karaoke(job_id: str, audio_path: str, lyrics_text: str | None = None):
    """[SSE] 一键处理：分离 + 听写（+ 可选对齐）"""
    from processing import one_click_karaoke, align_lyrics
    yield _sse("started", {"job_id": job_id, "type": "karaoke"})
    result: dict = {}
    try:
        for vocal, accomp, lrc, lrc_line, status, *_ in one_click_karaoke(audio_path):
            _update_job(job_id, progress=status)
            if vocal: result["vocal_path"] = vocal
            if accomp: result["accomp_path"] = accomp
            if lrc: result["lrc_word"] = lrc
            if lrc_line: result["lrc_line"] = lrc_line
            yield _sse("progress", {"progress": status})
        # 可选：用已有歌词重新对齐
        if lyrics_text and result.get("vocal_path") and os.path.exists(result["vocal_path"]):
            for lrc_w, lrc_l, status, *_ in align_lyrics(result["vocal_path"], lyrics_text):
                _update_job(job_id, progress=status)
                if lrc_w: result["lrc_word"] = lrc_w
                if lrc_l: result["lrc_line"] = lrc_l
                yield _sse("progress", {"progress": status})
        _update_job(job_id, status="done", **result)
        yield _sse("done", {"job_id": job_id, **result})
    except Exception as e:
        _error_job(job_id, str(e))
        yield _sse("error", {"job_id": job_id, "error": str(e)})


def _stream_separate(job_id: str, audio_path: str):
    """[SSE] 人声分离"""
    from processing import _run_demucs
    yield _sse("started", {"job_id": job_id, "type": "separate"})
    result: dict = {}
    try:
        for vocal, accomp, status in _run_demucs(audio_path):
            _update_job(job_id, progress=status)
            if vocal: result["vocal_path"] = vocal
            if accomp: result["accomp_path"] = accomp
            yield _sse("progress", {"progress": status})
        _update_job(job_id, status="done", **result)
        yield _sse("done", {"job_id": job_id, **result})
    except Exception as e:
        _error_job(job_id, str(e))
        yield _sse("error", {"job_id": job_id, "error": str(e)})


def _stream_align(job_id: str, audio_path: str, lyrics_text: str | None = None):
    """[SSE] 歌词生成/对齐：无歌词→听写，有歌词→对齐"""
    from processing import align_lyrics
    yield _sse("started", {"job_id": job_id, "type": "align"})
    result: dict = {}
    try:
        for lrc_word, lrc_line, status, *_ in align_lyrics(audio_path, lyrics_text or ""):
            _update_job(job_id, progress=status)
            if lrc_word: result["lrc_word"] = lrc_word
            if lrc_line: result["lrc_line"] = lrc_line
            yield _sse("progress", {"progress": status})
        _update_job(job_id, status="done", **result)
        yield _sse("done", {"job_id": job_id, **result})
    except Exception as e:
        _error_job(job_id, str(e))
        yield _sse("error", {"job_id": job_id, "error": str(e)})


def _stream_mv2ktv(job_id: str, video_path: str, lyrics_text: str, style_preset: str):
    """[SSE] MV转KTV视频"""
    from video_processing import mv_to_ktv
    yield _sse("started", {"job_id": job_id, "type": "mv2ktv"})
    result: dict = {}
    try:
        for vocal, accomp, lrc, ass_text, video_out, status, *_ in \
                mv_to_ktv(video_path, lyrics_text or "", style_preset):
            _update_job(job_id, progress=status)
            if vocal: result["vocal_path"] = vocal
            if accomp: result["accomp_path"] = accomp
            if lrc: result["lrc_word"] = lrc
            if ass_text: result["ass_subtitle"] = ass_text
            if video_out: result["video_path"] = video_out
            yield _sse("progress", {"progress": status})
        _update_job(job_id, status="done", **result)
        yield _sse("done", {"job_id": job_id, **result})
    except Exception as e:
        _error_job(job_id, str(e))
        yield _sse("error", {"job_id": job_id, "error": str(e)})


def _save_upload(file: UploadFile) -> str:
    """保存上传的文件到临时目录"""
    suffix = Path(file.filename or "audio").suffix or ".mp3"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(file.file.read())
    return path


def _start_job(job_type: str, fn):
    """创建任务并启动后台线程"""
    job_id = str(uuid.uuid4())[:12]
    with _lock:
        jobs[job_id] = {"id": job_id, "type": job_type, "status": "queued", "progress": ""}
    threading.Thread(target=fn, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "status_url": f"/status/{job_id}"}


@app.post("/submit/karaoke", tags=["一键处理：分离人声+伴奏 → 听写歌词"])
async def submit_karaoke(file: UploadFile = File(...), lyricsText: str | None = Form(None)):
    """一键处理：分离人声+伴奏 → 听写歌词
    可选传入 lyrics 文本，将在听写后自动对齐生成更精准的逐词/行级 LRC
    """
    path = _save_upload(file)
    return _start_job("karaoke", lambda jid: _run_karaoke_job(jid, path, lyricsText))


@app.post("/submit/separate", tags=["人声伴奏分离"])
async def submit_separate(file: UploadFile = File(...)):
    path = _save_upload(file)
    return _start_job("separate", lambda jid: _run_separate_job(jid, path))


@app.post("/submit/align", tags=["歌词生成/对齐：无歌词听写，有歌词对齐"])
async def submit_align(file: UploadFile = File(...), lyricsText: str | None = Form(None)):
    """歌词生成/对齐：
    - 不传 lyricsText：自动听写生成歌词
    - 传入 lyricsText：用已有歌词进行 CTC 强制对齐（更精准）
    """
    path = _save_upload(file)
    return _start_job("align", lambda jid: _run_align_job(jid, path, lyricsText))


@app.post("/submit/mv2ktv", tags=["MV转KTV视频：提取音频→分离→歌词→ASS字幕→多音轨视频"])
async def submit_mv2ktv(file: UploadFile = File(...),
                        lyricsText: str | None = Form(None),
                        stylePreset: str = Form("经典KTV")):
    """MV转KTV视频：提取音频 → 分离人声伴奏 → 歌词生成(有歌词则对齐,无则听写) → ASS字幕 → 多音轨KTV视频
    
    输出视频包含两条音轨：音轨1=原唱(人声+伴奏)，音轨2=纯伴奏
    """
    path = _save_upload(file)
    return _start_job("mv2ktv", lambda jid: _run_mv2ktv_job(jid, path, lyricsText, stylePreset))


# ======================== SSE 流式端点 ========================
# 注意：用 def（非 async def），同步生成器由 FastAPI 线程池执行，不阻塞事件循环。
# 客户端断开连接时生成器自动停止，任务终止；但 jobs 字典中已写入的结果仍保留。

@app.post("/stream/karaoke", tags=["SSE流式：一键处理（分离+听写+可选对齐）"])
def stream_karaoke(file: UploadFile = File(...), lyricsText: str | None = Form(None)):
    """SSE 流式一键处理，实时推送进度
    
    事件流：
      event: started  → data: {"job_id":"...","type":"karaoke"}
      event: progress → data: {"progress":"⏱️ 00:15 | 🔊 分离中... 45%"}
      event: done     → data: {"job_id":"...","vocal_path":"...","accomp_path":"...","lrc_word":"..."}
      event: error     → data: {"job_id":"...","error":"..."}
    """
    path = _save_upload(file)
    job_id = _create_stream_job("karaoke")
    return StreamingResponse(
        _stream_karaoke(job_id, path, lyricsText),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@app.post("/stream/separate", tags=["SSE流式：人声伴奏分离"])
def stream_separate(file: UploadFile = File(...)):
    """SSE 流式人声分离，实时推送进度"""
    path = _save_upload(file)
    job_id = _create_stream_job("separate")
    return StreamingResponse(
        _stream_separate(job_id, path),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@app.post("/stream/align", tags=["SSE流式：歌词生成/对齐（无歌词听写，有歌词对齐）"])
def stream_align(file: UploadFile = File(...), lyricsText: str | None = Form(None)):
    """SSE 流式歌词生成/对齐，实时推送进度
    
    - 不传 lyricsText：自动听写生成歌词
    - 传入 lyricsText：用已有歌词进行 CTC 强制对齐（更精准）
    """
    path = _save_upload(file)
    job_id = _create_stream_job("align")
    return StreamingResponse(
        _stream_align(job_id, path, lyricsText),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@app.post("/stream/mv2ktv", tags=["SSE流式：MV转KTV视频"])
def stream_mv2ktv(file: UploadFile = File(...),
                  lyricsText: str | None = Form(None),
                  stylePreset: str = Form("经典KTV")):
    """SSE 流式 MV转KTV视频，实时推送进度
    
    事件流：
      event: started  → data: {"job_id":"...","type":"mv2ktv"}
      event: progress → data: {"progress":"⏱️ 03:15 | 🎬 渲染中... 45%"}
      event: done     → data: {"job_id":"...","video_path":"...","vocal_path":"...","ass_subtitle":"..."}
      event: error     → data: {"job_id":"...","error":"..."}
    
    done 后用 GET /result/{job_id}/{filename} 下载结果文件
    """
    path = _save_upload(file)
    job_id = _create_stream_job("mv2ktv")
    return StreamingResponse(
        _stream_mv2ktv(job_id, path, lyricsText or "", stylePreset),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@app.get("/status/{job_id}", tags=["查询任务"])
async def get_status(job_id: str):
    with _lock:
        if job_id not in jobs:
            raise HTTPException(404, "任务不存在")
        job = dict(jobs[job_id])
    return {k: v for k, v in job.items()
            if not k.startswith("_")}


@app.get("/result/{job_id}/{filename}", tags=["下载结果"])
async def download_result(job_id: str, filename: str):
    with _lock:
        if job_id not in jobs:
            raise HTTPException(404, "任务不存在")
        job = jobs[job_id]
    # 从输出目录查找匹配文件
    result_dir = Path(config.OUTPUT_DIR)
    candidates = list(result_dir.glob(f"*{filename}*"))
    if candidates:
        return FileResponse(str(candidates[0]), filename=filename)
    raise HTTPException(404, f"文件 {filename} 未找到")


@app.get("/jobs", tags=["列出任务"])
async def list_jobs():
    with _lock:
        return [{"id": jid, "type": j["type"], "status": j["status"],
                 "progress": j.get("progress", "")[-80:]}
                for jid, j in jobs.items()]


if __name__ == "__main__":
    print(f"🚀 API 启动 http://0.0.0.0:7861/docs")
    uvicorn.run(app, host="0.0.0.0", port=7861)
