# app.py (最终版 - Coqui XTTS v2)

import gradio as gr
import os
import uuid
import traceback
import torch
from TTS.api import TTS
from pydub import AudioSegment

# ---- 1. 启动时直接加载 XTTS 模型 (如果不存在则自动下载) ----
print("应用脚本启动，开始加载 Coqui XTTS v2 模型...")
print("首次启动时会自动下载模型(约2GB)，可能需要几分钟，请耐心等待...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用的设备: {DEVICE}")

try:
    print("正在初始化 TTS 对象，这将触发模型下载（如果需要）...")
    # 这是最顶层、最稳定的 API，它会自动处理模型的下载、缓存和加载
    TTS_MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
    print("✅ Coqui XTTS v2 模型加载成功！")
except Exception as e:
    print(f"🔴 XTTS 模型加载失败: {e}")
    # 在 Gradio 界面启动前就抛出异常，以便在 Hugging Face 的日志中清晰地看到失败原因
    raise e


# ---- 2. 核心功能辅助函数 ----
def convert_to_wav(filepath):
    """
    使用 pydub 将任何音频格式转换为临时的 WAV 文件，
    并将其标准化为 XTTS 所需的格式 (24000Hz, 单声道)。
    """
    temp_wav_path = f"temp_converted_{uuid.uuid4().hex}.wav"
    try:
        audio = AudioSegment.from_file(filepath)
        # XTTS v2 在 24000Hz 采样率下训练，以获得最佳效果
        audio = audio.set_frame_rate(24000).set_channels(1)
        audio.export(temp_wav_path, format="wav")
        return temp_wav_path
    except Exception as e:
        raise IOError(f"Pydub 转换音频失败: {e}")


def _process_audio_source(audio_file, mic_input, youtube_input):
    """
    统一处理所有三种音频源，返回一个清理过的、可用的 WAV 文件路径。
    """
    source_to_process = None
    if audio_file is not None:
        print(f"处理上传的文件: {audio_file.name}")
        source_to_process = audio_file.name
    elif mic_input is not None:
        print(f"处理麦克风录音: {mic_input}")
        source_to_process = mic_input
    elif youtube_input:
        print(f"处理 YouTube 链接: {youtube_input}")
        import yt_dlp
        temp_filename = f"temp_yt_{uuid.uuid4().hex}"
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
            'outtmpl': temp_filename,
            'quiet': True,
            'nocheckcertificate': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_input])

        # 查找下载的文件，因为后缀可能不确定
        downloaded_file = None
        for f in os.listdir('.'):
            if f.startswith(temp_filename):
                downloaded_file = f
                break
        if downloaded_file:
            print(f"YouTube 音频已下载至: {downloaded_file}")
            source_to_process = downloaded_file

    if source_to_process and os.path.exists(source_to_process):
        # 无论来源是什么，都通过 pydub 进行标准化处理
        return convert_to_wav(source_to_process)

    return None


# ---- 3. Gradio 事件处理函数 ----
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
        temp_files.append(output_wav_path)

        # XTTS 的核心 API 调用，简洁而强大
        TTS_MODEL.tts_to_file(
            text=text,
            file_path=output_wav_path,
            speaker_wav=source_wav_path,
            language=language
        )

        progress(1.0, desc="合成完毕！")
        return output_wav_path
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"处理失败: {e}")
    finally:
        # 无论成功还是失败，都清理所有本次运行产生的临时文件
        for f in temp_files:
            if f and os.path.exists(f) and ("temp_" in f or "synthesized_" in f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"清理临时文件失败: {f}, 错误: {e}")


# ---- 4. Gradio 界面定义 ----
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