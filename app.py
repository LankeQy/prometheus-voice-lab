# app.py (终极官方文档复现版)

import gradio as gr
import os
import uuid
import traceback
import torch
from TTS.api import TTS
from pydub import AudioSegment
import soundfile as sf

# ---- 1. 启动时加载 XTTS 模型 ----
print("应用脚本启动，开始加载 Coqui XTTS v2 模型...")
print("这将首先检查并下载模型（如果需要），然后加载到内存...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用的设备: {DEVICE}")

try:
    # *** 终极修复：分两步走，绝对稳健 ***

    # 步骤 1: 创建一个轻量级 TTS 对象，用它来安全地触发下载
    # 我们知道这个标识符在 TTS==0.22.0 中是有效的
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    print(f"正在使用模型标识符: {model_name}")
    print("正在检查模型是否存在，如果不存在将开始下载...")

    # 创建一个不加载模型的管理器，只用于下载
    manager = TTS().manager
    # download_model 是一个更底层的、只负责下载的函数
    model_path, config_path, _ = manager.download_model(model_name)

    # 检查文件是否真的下载下来了
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError(f"模型文件下载失败。在路径 {model_path} 或 {config_path} 未找到文件。")

    print("✅ 模型文件已确认存在于本地缓存中。")
    print("正在将模型加载到内存...")

    # 步骤 2: 现在我们确信模型已在本地，再创建完整的 TTS 对象来加载它
    # 直接使用 model_name，它现在会从缓存加载而不是网络
    TTS_MODEL = TTS(model_name=model_name, progress_bar=False).to(DEVICE)

    print("✅ Coqui XTTS v2 模型加载成功！")

except Exception as e:
    print(f"🔴 XTTS 模型加载失败: {e}");
    raise e


# ---- 2. 核心功能辅助函数 ----
# ... [这部分代码与上一版完全相同，无需改动] ...
def convert_to_wav(filepath):
    temp_wav_path = f"temp_converted_{uuid.uuid4().hex}.wav"
    try:
        audio = AudioSegment.from_file(filepath)
        audio = audio.set_frame_rate(24000).set_channels(1)
        audio.export(temp_wav_path, format="wav")
        return temp_wav_path
    except Exception as e:
        raise IOError(f"Pydub 转换音频失败: {e}")


def _process_audio_source(audio_file, mic_input, youtube_input):
    source_to_process = None
    if audio_file is not None:
        source_to_process = audio_file.name
    elif mic_input is not None:
        source_to_process = mic_input
    elif youtube_input:
        import yt_dlp
        temp_filename = f"temp_yt_{uuid.uuid4().hex}"
        ydl_opts = {'format': 'bestaudio/best',
                    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
                    'outtmpl': temp_filename, 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_input])
        downloaded_file = next((f for f in os.listdir('.') if f.startswith(temp_filename)), None)
        if downloaded_file: source_to_process = downloaded_file
    if source_to_process and os.path.exists(source_to_process):
        return convert_to_wav(source_to_process)
    return None


# ---- 3. Gradio 事件处理函数 ----
# ... [这部分代码与上一版完全相同，无需改动] ...
def clone_and_synthesize(audio_file, mic_input, youtube_input, text, language, progress=gr.Progress()):
    temp_files = []
    try:
        progress(0.1, desc="检查输入...")
        if not text: raise gr.Error("请输入要合成的文本。")
        progress(0.2, desc="处理音频源...")
        source_wav_path = _process_audio_source(audio_file, mic_input, youtube_input)
        if source_wav_path is None: raise gr.Error("请提供一个有效的音频源。")
        temp_files.append(source_wav_path)
        progress(0.5, desc="正在克隆声音并合成语音...")
        output_wav_path = f"synthesized_{uuid.uuid4().hex}.wav"
        TTS_MODEL.tts_to_file(text=text, file_path=output_wav_path, speaker_wav=source_wav_path, language=language)
        progress(1.0, desc="合成完毕！")
        temp_files.append(output_wav_path)
        return output_wav_path
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"处理失败: {e}")
    finally:
        for f in temp_files:
            if f and os.path.exists(f) and ("temp_" in f or "synthesized_" in f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"清理临时文件失败: {f}, 错误: {e}")


# ---- 4. Gradio 界面定义 ----
# [界面代码与上一版完全相同]
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 普罗米修斯旗舰声音实验室 (多语言版)")
    gr.Markdown("一个支持中、日、英等多种语言的高质量在线声音克隆工具。由 Coqui XTTS v2 驱动。")
    gr.Markdown("✅ **环境已就绪**，XTTS 模型已加载完毕。")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 提供声音源")
            with gr.Tabs():
                with gr.TabItem("📁 上传文件"):
                    audio_file_input = gr.File(label="支持 WAV, MP3, M4A 等 (推荐5-15秒清晰人声)")
                with gr.TabItem("🔗 视频平台链接"):
                    youtube_input = gr.Textbox(label="粘贴 URL")
                with gr.TabItem("🎤 麦克风录制"):
                    mic_input = gr.Audio(sources=["microphone"], type="filepath", label="点击录制")
        with gr.Column(scale=2):
            gr.Markdown("### 2. 输入文本并选择语言")
            text_input = gr.Textbox(label="输入要合成的文本", value="你好，世界。こんにちは、世界。Hello world.", lines=4)
            lang_dropdown = gr.Dropdown(
                choices=["zh-cn", "ja", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "hu",
                         "ko"], value="zh-cn", label="选择语言")
            synthesize_btn = gr.Button("克隆并合成语音", variant="primary")
    gr.Markdown("---")
    gr.Markdown("### 3. 合成结果试听")
    audio_output = gr.Audio(label="合成结果", type="filepath")
    synthesize_btn.click(fn=clone_and_synthesize,
                         inputs=[audio_file_input, mic_input, youtube_input, text_input, lang_dropdown],
                         outputs=[audio_output])

# ---- 5. 启动应用 ----
print("所有模型加载完毕，正在启动Gradio服务...")
demo.launch(server_name="0.0.0.0", server_port=7860)
print("✅ Gradio 服务已启动。")