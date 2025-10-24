# app.py
import gradio as gr
import torch
import torchaudio

from speechbrain.inference.classifiers import EncoderClassifier
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import yt_dlp
import os
import uuid
import traceback

# ---- 1. 模型加载 ----
print("正在加载所有模型，这将需要几分钟...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

try:
    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb",
        run_opts={"device": device}
    )
    print("✅ 声纹提取模型加载成功！")
except Exception as e:
    print(f"🔴 声纹提取模型加载失败: {e}")
    speaker_model = None

try:
    tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", language="zh-cn")
    tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    print("✅ TTS 试听模型加载成功 (已配置中文)！")
except Exception as e:
    print(f"🔴 TTS 试听模型加载失败: {e}")
    tts_model = None

print("✅ 所有模型加载完毕，应用准备就绪！")


# ---- 2. 核心功能函数
def process_audio_and_get_name(filepath, source_info="file"):
    """
    加载、重采样并清理音频文件，返回波形和基本名称。
    """
    if filepath is None: return None, None
    print(f"正在处理来自 '{source_info}' 的音频: {filepath}")
    try:
        signal, fs = torchaudio.load(filepath)
        # 确保采样率为 16kHz
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)
        # 转换为单声道
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        source_name = os.path.splitext(os.path.basename(filepath))[0]

        # 清理临时文件
        if source_info in ["YouTube", "microphone_temp"]:
            try:
                os.remove(filepath)
                print(f"已清理临时文件: {filepath}")
            except Exception as e:
                print(f"清理临时文件失败 '{filepath}': {e}")

        return signal, source_name
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"音频处理失败: {e}")


def download_youtube_audio(youtube_url):
    """
    使用 yt-dlp 从 URL 下载音频并返回文件路径。
    """
    if not youtube_url: return None
    print(f"正在从 URL 下载: {youtube_url}")
    # 创建一个唯一的临时文件名
    temp_filename = os.path.join("/tmp", f"temp_audio_{uuid.uuid4().hex}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': temp_filename,  # 指定输出模板，不带扩展名
        'quiet': True,
        'nocheckcertificate': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        output_path = f"{temp_filename}.wav"  # yt-dlp 会自动添加扩展名
        if not os.path.exists(output_path):
            # 有时扩展名可能是其他格式，做一个后备检查
            possible_files = [f for f in os.listdir("/tmp") if f.startswith(os.path.basename(temp_filename))]
            if not possible_files: raise FileNotFoundError("yt-dlp 下载后未找到任何音频文件。")
            # 将找到的第一个文件重命名为期望的 .wav 文件
            found_file = os.path.join("/tmp", possible_files[0])
            os.rename(found_file, output_path)

        print(f"URL 音频已下载到: {output_path}")
        return output_path
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"URL 下载或处理失败: {e}")


def generate_and_test(audio_file, mic_input, youtube_url, text_to_speak):
    """
    主逻辑函数：接收输入，生成声纹和试听音频。
    """
    # 检查输入
    if not any([audio_file, mic_input, youtube_url]):
        raise gr.Error("请提供一个音频源：上传文件、录音或视频链接。")
    if not text_to_speak:
        raise gr.Error("请输入要试听的文本。")
    if speaker_model is None or tts_model is None:
        raise gr.Error("核心模型未能加载，应用无法工作。请检查启动日志。")

    waveform, source_name = None, "audio"

    # 根据输入源处理音频
    if youtube_url:
        youtube_filepath = download_youtube_audio(youtube_url)
        waveform, source_name = process_audio_and_get_name(youtube_filepath, "YouTube")
    elif audio_file is not None:
        # 关键修复：Gradio 4.x 直接返回文件路径字符串，而不是文件对象
        waveform, source_name = process_audio_and_get_name(audio_file, "file")
    elif mic_input is not None:
        waveform, source_name = process_audio_and_get_name(mic_input, "microphone_temp")
        source_name = "mic_recording"

    if waveform is None:
        raise gr.Error("无法从提供的源加载音频。")

    print("正在生成声纹...")
    with torch.no_grad():
        # SpeechBrain 的 encode_batch 期望 (batch, time)，我们的 waveform 是 (1, time)，正好符合
        embedding = speaker_model.encode_batch(waveform.to(device))
        embedding = torch.nn.functional.normalize(embedding, dim=2)
        # 输出形状是 (1, 1, 512)，提取出 (512,) 的向量
        final_embedding = embedding.squeeze()

    pt_filename = f"{source_name}_embedding.pt"
    torch.save(final_embedding, pt_filename)
    print(f"声纹文件已保存: {pt_filename}")

    print("正在进行TTS试听...")
    inputs = tts_processor(text=text_to_speak, return_tensors="pt").to(device)
    # SpeechT5 需要 (1, 512) 形状的 embedding
    speaker_embeddings_for_tts = final_embedding.unsqueeze(0).to(device)

    speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings_for_tts, vocoder=tts_vocoder)

    test_audio_filename = f"test_{source_name}.wav"
    # torchaudio.save 需要 (channels, time) 格式
    torchaudio.save(test_audio_filename, speech.cpu().unsqueeze(0), 16000)
    print(f"试听音频已生成: {test_audio_filename}")

    return pt_filename, test_audio_filename


# ---- 3. Gradio 界面定义 ----
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 普罗米修斯旗舰声音实验室")
    gr.Markdown("在这里生产、并即时测试用于您 AI 大cha脑的任何声音。")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 提供声音源 (三选一)")
            with gr.Tabs():
                with gr.TabItem("📁 上传文件"):
                    # Gradio 4.x, type='filepath' 是默认值，返回字符串路径
                    audio_file_input = gr.File(label="支持 WAV, MP3, M4A 等格式")
                with gr.TabItem("🔗 视频平台链接"):
                    youtube_input = gr.Textbox(label="粘贴 YouTube, Bilibili, 抖音等 URL",
                                               placeholder="https://www.bilibili.com/video/BV...")
                with gr.TabItem("🎤 麦克风录制"):
                    mic_input = gr.Audio(sources=["microphone"], type="filepath", label="点击录制你的声音")

            gr.Markdown("### 2. 输入试听文本")
            text_input = gr.Textbox(label="输入要试听的中文", value="你好，我是普罗米修斯。这是我的新声音。")

            generate_btn = gr.Button("生成并试听", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 3. 获取结果")
            pt_output = gr.File(label="下载声纹 (.pt 文件)")
            audio_output = gr.Audio(label="试听克隆效果", type="filepath")

    generate_btn.click(
        fn=generate_and_test,
        inputs=[audio_file_input, mic_input, youtube_input, text_input],
        outputs=[pt_output, audio_output],
        api_name="generate"
    )

# ---- 4. 启动应用 ----
if __name__ == "__main__":
    demo.launch()