# app.py (XTTS 最终版)

import gradio as gr
import os
import uuid
import traceback
import torch
import torchaudio
from TTS.api import TTS
from pydub import AudioSegment

# ---- 1. 启动时直接加载 XTTS 模型 ----
print("应用脚本启动，开始加载 Coqui XTTS v2 模型...")
print("这可能需要几分钟，请耐心等待 Gradio 界面出现...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用的设备: {DEVICE}")

try:
    print("正在加载 XTTS 模型到内存...")
    # XTTS 会自动从缓存加载预下载的模型
    TTS_MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
    print("✅ Coqui XTTS v2 模型加载成功！")
except Exception as e:
    print(f"🔴 XTTS 模型加载失败: {e}");
    raise e


# ---- 2. 核心功能辅助函数 ----
def convert_to_wav(filepath):
    """使用 pydub 将任何音频格式转换为临时的 WAV 文件，并进行标准化处理"""
    temp_wav_path = f"temp_{uuid.uuid4().hex}.wav"
    try:
        audio = AudioSegment.from_file(filepath)
        # XTTS 需要 22050Hz 的采样率
        audio = audio.set_frame_rate(22050).set_channels(1)
        audio.export(temp_wav_path, format="wav")
        return temp_wav_path
    except Exception as e:
        raise IOError(f"Pydub 转换音频失败: {e}")


def _process_audio_source(audio_file, mic_input, youtube_input):
    """统一处理所有音频源，返回一个可用的 WAV 文件路径"""
    if audio_file is not None:
        return convert_to_wav(audio_file.name)
    elif mic_input is not None:
        return convert_to_wav(mic_input)
    elif youtube_input:
        import yt_dlp
        temp_filename = f"temp_yt_{uuid.uuid4().hex}"
        ydl_opts = {'format': 'bestaudio/best',
                    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
                    'outtmpl': temp_filename, 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_input])
        downloaded_file = f"{temp_filename}.wav"
        if not os.path.exists(downloaded_file):
            # 兼容 yt-dlp 可能输出其他格式的情况
            for f in os.listdir('.'):
                if f.startswith(temp_filename):
                    downloaded_file = f
                    break
        return convert_to_wav(downloaded_file)
    return None


# ---- 3. Gradio 事件处理函数 ----
def clone_and_synthesize(audio_file, mic_input, youtube_input, text, language, progress=gr.Progress()):
    try:
        progress(0.1, desc="检查输入...")
        if not text: raise gr.Error("请输入要合成的文本。")
        source_wav_path = _process_audio_source(audio_file, mic_input, youtube_input)
        if source_wav_path is None: raise gr.Error("请提供一个音频源。")

        progress(0.5, desc="正在克隆声音并合成语音...")
        output_wav_path = f"synthesized_{uuid.uuid4().hex}.wav"

        # XTTS 的核心 API 调用，极其简洁
        TTS_MODEL.tts_to_file(
            text=text,
            file_path=output_wav_path,
            speaker_wav=source_wav_path,
            language=language
        )

        # 清理临时文件
        if os.path.exists(source_wav_path) and source_wav_path.startswith("temp_"):
            os.remove(source_wav_path)

        progress(1.0, desc="合成完毕！")
        return output_wav_path
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"处理失败: {e}")


# ---- 4. Gradio 界面定义 ----
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 普罗米修斯旗舰声音实验室 (多语言版)")
    gr.Markdown("一个支持中、日、英三种语言的高质量在线声音克隆工具。由 Coqui XTTS v2 驱动。")
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
            lang_dropdown = gr.Dropdown(choices=["zh-cn", "ja", "en"], value="zh-cn", label="选择语言")
            synthesize_btn = gr.Button("克隆并合成语音", variant="primary")

    gr.Markdown("---")
    gr.Markdown("### 3. 合成结果试听")
    audio_output = gr.Audio(label="合成结果", type="filepath")

    synthesize_btn.click(
        fn=clone_and_synthesize,
        inputs=[audio_file_input, mic_input, youtube_input, text_input, lang_dropdown],
        outputs=[audio_output]
    )

# ---- 5. 启动应用 ----
print("所有模型加载完毕，正在启动Gradio服务...")
demo.launch(server_name="0.0.0.0", server_port=7860)
print("✅ Gradio 服务已启动。")