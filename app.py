"""卡拉OK工作站 - 主入口 / Gradio UI 构建

Tab 结构：
  🚀 一键处理 -> 分离 + 听写
  🎵 单独分离 -> 仅分离人声+伴奏
  🎯 歌词生成对齐 -> 无歌词听写 / 有歌词对齐（统一入口）
  🎬 MV转KTV -> 视频提取音频→分离→歌词→ASS字幕→多音轨KTV视频
  ⚙️ 模型管理 -> 下载/删除/切换模型
"""
import os
import gradio as gr
from config import AVAILABLE_WHISPER, AVAILABLE_DEMUCS, WHISPER_SIZES, DEMUCS_SIZES
import config
from model_loader import get_status_html, switch_whisper_model
from model_manager import (is_whisper_cached, is_demucs_cached,
                           get_whisper_cache_size, get_demucs_cache_size,
                           whisper_dl_btn, whisper_redl_btns, whisper_del_btns,
                           demucs_dl_btn, demucs_redl_btns, demucs_del_btns,
                           switch_demucs_model)
from processing import separate_vocals, align_lyrics, one_click_karaoke
from video_processing import mv_to_ktv, PRESET_CHOICES

# API 文档外部端口（容器内 API 跑在 7861，映射到宿主机的端口通过该变量告知前端）
API_PORT = os.environ.get("API_PORT", "7861")


# 自定义 CSS
CUSTOM_CSS = """
    .tab-content { padding-top: 8px !important; }
    .column>.form{margin-top: -10px;background: none;border: none;}
    .model-row { align-items: center !important; gap: 8px !important; }
    .model-row > .column { min-width: min(160px,100%) !important; flex: 0 0 auto !important; }
    .model-row button { width: 42px !important; min-width: 42px !important; padding: 4px 4px !important; }
    .download-btn { background-color: #22c55e !important; color: white !important; }
    .delete-btn { background-color: #f97316 !important; color: white !important; }
    /* 状态 Textbox 去掉外层和 textarea 边框，做成内嵌行内提示 */
    .status-box {padding:0; border: none !important; background: transparent !important; box-shadow: none !important;}
    .status-box textarea {border: none !important; background: transparent !important; box-shadow: none !important; color: #888 !important; font-size: 12px !important; min-height: 24px !important; padding: 4px 0 !important; resize: none !important;}
    .status-box input { border: none !important; background: transparent !important; box-shadow: none !important; padding: 4px 0 !important; min-height: 24px !important; color: #888 !important; font-size: 12px !important; }
    /* MV 歌词文本框与视频框等高：Gradio Textbox 无 height 参数，用 CSS 固定 textarea 高度匹配视频(player=320) */
    .mv-lyrics textarea { height: 270px !important; resize: none !important; }
    .rw-audio{height: 320px !important; }
    """

# Gradio 6 起 css/theme 等应用级参数从 gr.Blocks() 移到 launch()；旧版本则相反。
# 这里按大版本自适应，保证 4.x / 5.x / 6.x 都能启动。
GRADIO_MAJOR = int(gr.__version__.split(".")[0])

blocks_kwargs = {}
if GRADIO_MAJOR < 6:
    blocks_kwargs["css"] = CUSTOM_CSS

with gr.Blocks(
    title="卡拉OK工作站",
    **blocks_kwargs
) as demo:
    gr.Markdown("# 🎤 卡拉OK智能工作站 v2\n一键MV转KTV / 分离人声 / **歌词生成对齐**")
    status_md = gr.Markdown(get_status_html(), elem_id="status-bar")

    # ==== 🎬 一键MV转KTV（移至最前） ====
    with gr.Tab("🎬 一键MV转KTV"):
        gr.Markdown(
            "上传MV视频 → 提取音频 → 分离人声伴奏 → 生成逐词歌词 → 烧录卡拉OK字幕 → **多音轨KTV视频**\n\n"
            "💡 **多音轨说明**：输出视频包含两条音轨（音轨1=原唱，音轨2=纯伴奏），播放器可切换\n\n"
            "💡 **歌词来源**：填写歌词文本则用对齐（更精准），留空则自动听写"
        )
        with gr.Row():
            mv_video = gr.Video(label="上传MV视频", height=320)
            mv_lyrics = gr.Textbox(
                label="歌词文本（可选）", lines=6,
                placeholder="粘贴歌词文本以获得更精准的字幕对齐\n留空则自动听写生成歌词",
                elem_classes="mv-lyrics"
            )
        mv_style = gr.Dropdown(
            PRESET_CHOICES, value="经典KTV", label="字幕样式", interactive=True
            #,info="卡拉OK字幕的视觉风格"
        )
        mv_btn = gr.Button("🎬 一键生成KTV", variant="primary", interactive=False)
        mv_output = gr.Video(label="KTV视频输出")
        mv_status = gr.Textbox(label="状态", interactive=False)
        mv_video.change(fn=lambda v: gr.update(interactive=v is not None),
                        inputs=[mv_video], outputs=[mv_btn])

        def _mv_to_ktv_ui(video_file, lyrics_text, style_preset):
            """包装 mv_to_ktv，UI 只需输出视频+状态+按钮"""
            for vocal, accomp, lrc, ass_text, video_out, status, btn_update in \
                    mv_to_ktv(video_file, lyrics_text, style_preset):
                yield video_out, status, btn_update

        mv_btn.click(fn=_mv_to_ktv_ui, inputs=[mv_video, mv_lyrics, mv_style],
                     outputs=[mv_output, mv_status, mv_btn])

    # ==== 🚀 一键处理 ====
    with gr.Tab("🚀 MP3一键提取伴奏歌词"):
        gr.Markdown("上传完整歌曲 → 自动分离人声和伴奏 → 自动听写生成 LRC 歌词")
        one_audio = gr.Audio(type="filepath", label="上传歌曲")
        one_btn = gr.Button("⚡ 一键开始", variant="primary", interactive=False)
        with gr.Row():
            one_vocal = gr.Audio(label="人声", type="filepath")
            one_accomp = gr.Audio(label="伴奏", type="filepath")
        with gr.Row():
            one_lrc = gr.Textbox(label="逐词 LRC", lines=12)
            one_lrc_line = gr.Textbox(label="行级 LRC", lines=12)
        one_status = gr.Textbox(label="状态", interactive=False)
        one_audio.change(fn=lambda f: gr.update(interactive=f is not None), inputs=[one_audio], outputs=[one_btn])
        one_btn.click(fn=one_click_karaoke, inputs=[one_audio],
                      outputs=[one_vocal, one_accomp, one_lrc, one_lrc_line, one_status, one_btn])

    # ==== 🎵 单独分离人声 ====
    with gr.Tab("🎵 单独分离人声伴奏"):
        gr.Markdown("上传歌曲 → 分离人声 + 伴奏")
        sep_audio = gr.Audio(type="filepath", label="上传音频")
        sep_btn = gr.Button("🎙️ 开始分离", variant="secondary", interactive=False)
        with gr.Row():
            sep_vocal = gr.Audio(label="人声输出", type="filepath")
            sep_accomp = gr.Audio(label="伴奏输出", type="filepath")
        sep_status = gr.Textbox(label="状态", interactive=False)
        sep_audio.change(fn=lambda f: gr.update(interactive=f is not None), inputs=[sep_audio], outputs=[sep_btn])
        sep_btn.click(fn=separate_vocals, inputs=[sep_audio],
                      outputs=[sep_vocal, sep_accomp, sep_status, sep_btn])

    # ==== 🎯 歌词生成/对齐（统一入口）====
    with gr.Tab("🎯 歌词生成对齐"):
        gr.Markdown(
            "上传音频 → 自动生成逐词 LRC 歌词\n\n"
            "💡 **无歌词**：自动听写生成歌词（适合没有歌词文本的情况）\n"
            "💡 **有歌词**：粘贴歌词文本则用 CTC 强制对齐（更精准）"
        )
        with gr.Row():
            align_audio = gr.Audio(type="filepath", label="上传音频/人声", elem_classes="rw-audio")
            align_lyrics_input = gr.Textbox(
                label="歌词文本（可选）", lines=6,
                placeholder="粘贴歌词文本以获得更精准的字幕对齐\n留空则自动听写生成歌词",
                elem_classes="mv-lyrics"
            )
        align_btn = gr.Button("🎯 开始生成", variant="primary", interactive=False)
        with gr.Row():
            align_lrc_word = gr.Textbox(label="逐词 LRC（卡拉OK变色）", lines=15)
            align_lrc_line = gr.Textbox(label="行级 LRC", lines=15)
        align_status = gr.Textbox(label="状态", interactive=False)
        align_audio.change(fn=lambda a: gr.update(interactive=a is not None),
                         inputs=[align_audio], outputs=[align_btn])
        align_btn.click(fn=align_lyrics, inputs=[align_audio, align_lyrics_input],
                       outputs=[align_lrc_word, align_lrc_line, align_status, align_btn])

    # ==== ⚙️ 模型管理 ====
    with gr.Tab("⚙️ 模型管理"):
        gr.Markdown("下载/管理模型，仅已下载的模型可切换")
        with gr.Row():
            whisper_dd = gr.Dropdown(AVAILABLE_WHISPER, value=config.WHISPER_MODEL, label="🎙️ Whisper 模型", interactive=True, info="仅已下载的模型可切换")
            demucs_dd = gr.Dropdown(AVAILABLE_DEMUCS, value=config.DEMUCS_MODEL, label="🎵 Demucs 模型", interactive=True, info="仅已下载的模型可切换")
        whisper_dd.change(fn=switch_whisper_model, inputs=[whisper_dd], outputs=[whisper_dd])
        demucs_dd.change(fn=switch_demucs_model, inputs=[demucs_dd], outputs=[demucs_dd])
        def _render_model_list(title, models, sizes_map, is_cached_fn, get_size_fn, dl_fn, redl_fn, del_fn):
            """[UI 构建] 每个模型占两行：第1行 名称+大小 | 按钮，第2行 操作状态"""
            gr.Markdown(title)
            for m in models:
                cached = is_cached_fn(m)
                size = get_size_fn(m) if cached else sizes_map[m]
                btn_state = gr.State(m)
                with gr.Row(equal_height=True, elem_classes="model-row"):
                    with gr.Column(scale=0):
                        size_md = gr.Markdown(f"**{m}**  ({size})")
                    with gr.Column(scale=0):
                        with gr.Row():
                            dl_btn = gr.Button("📥下载", variant="primary", size="sm", visible=not cached, elem_classes="download-btn")
                            redl_btn = gr.Button("🔄重下", variant="secondary", size="sm", visible=cached)
                            del_btn = gr.Button("🗑删除", variant="secondary", size="sm", visible=cached, elem_classes="delete-btn")
                st = gr.Textbox(show_label=False, lines=1, max_lines=1, elem_classes="status-box")
                if cached:
                    redl_btn.click(fn=redl_fn, inputs=[btn_state], outputs=[st, dl_btn, redl_btn, del_btn, size_md])
                    del_btn.click(fn=del_fn, inputs=[btn_state], outputs=[st, dl_btn, redl_btn, del_btn, size_md])
                else:
                    dl_btn.click(fn=dl_fn, inputs=[btn_state], outputs=[st, dl_btn, redl_btn, del_btn, size_md])

        with gr.Row():
            with gr.Column():
                _render_model_list("**🎙️ Whisper 模型**", AVAILABLE_WHISPER, WHISPER_SIZES,
                                   is_whisper_cached, get_whisper_cache_size,
                                   whisper_dl_btn, whisper_redl_btns, whisper_del_btns)
            with gr.Column():
                _render_model_list("**🎵 Demucs 模型**", AVAILABLE_DEMUCS, DEMUCS_SIZES,
                                   is_demucs_cached, get_demucs_cache_size,
                                   demucs_dl_btn, demucs_redl_btns, demucs_del_btns)

    # 自定义底部 API 链接（替换 Gradio 默认的 /view_api 链接）
    gr.HTML(
        f"""
        <div style="text-align:center;padding:0;font-size:13px;opacity:0.7;">
            <a href="#" onclick="window.open('http://' + window.location.hostname + ':{API_PORT}/docs', '_blank');return false;" style="text-decoration:underline;">
                📡 REST API 文档 / 异步任务接口
            </a>
        </div>
        """
    )

if __name__ == "__main__":
    import threading, uvicorn
    # 后台启动 API（与 Gradio 同进程共享模型，设 KARAOKE_API=0 可禁用）
    api_enabled = os.environ.get("KARAOKE_API", "1").strip().lower() not in ("0", "false", "no")
    if api_enabled:
        print(f"🚀 API 后台启动 http://0.0.0.0:{API_PORT}/docs")
        from api import app as api_app
        threading.Thread(
            target=lambda: uvicorn.run(api_app, host="0.0.0.0", port=API_PORT, log_level="warning"),
            daemon=True
        ).start()
    print("🚀 Gradio 正在启动，监听 0.0.0.0:7860 ...")
    demo.queue(default_concurrency_limit=1)
    launch_kwargs = {}
    if GRADIO_MAJOR >= 6:
        launch_kwargs["css"] = CUSTOM_CSS
        launch_kwargs["footer_links"] = ["gradio", "settings"]
    else:
        launch_kwargs["show_api"] = False

    demo.launch(
        server_name="0.0.0.0", server_port=7860, share=False,
        **launch_kwargs
    )
