
import gradio as gr
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import yt_dlp
import os
import uuid
import traceback

# ---- 1. 模型加载 ----
print("正在加载所有模型，这将需要几分钟...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

try:
    speaker_model = EncoderClassifier.from_huggingface("speechbrain/spkrec-xvect-voxceleb", run_opts={"device": device})
    print("✅ 声纹提取模型加载成功！")
except Exception as e:
    print(f"🔴 声纹提取模型加载失败: {e}")
    speaker_model = None

try:
    tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    print("✅ TTS 试听模型加载成功！")
except Exception as e:
    print(f"🔴 TTS 试听模型加载失败: {e}")
    tts_model = None

print("✅ 所有模型加载完毕，应用准备就绪！")


# ---- 2. 核心功能函数
def process_audio_and_get_name(filepath, source_info="file"):
    if filepath is None: return None, None
    print(f"正在处理来自 '{source_info}' 的音频: {filepath}")
    try:
        signal, fs = torchaudio.load(filepath)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        source_name = os.path.splitext(os.path.basename(filepath))[0]
        if source_info in ["YouTube", "microphone_temp"]:
            try:
                os.remove(filepath)
                print(f"已清理临时文件: {filepath}")
            except Exception as e:
                print(f"清理临时文件失败: {e}")
        return signal, source_name
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"音频处理失败: {e}")

def download_youtube_audio(youtube_url):
    if not youtube_url: return None
    print(f"正在从 URL 下载: {youtube_url}")
    temp_filename = f"temp_audio_{uuid.uuid4().hex}"
    ydl_opts = {'format': 'bestaudio/best', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}], 'outtmpl': temp_filename, 'quiet': True, 'nocheckcertificate': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        output_path = f"{temp_filename}.wav"
        if not os.path.exists(output_path):
             possible_files = [f for f in os.listdir('.') if f.startswith(temp_filename)]
             if not possible_files: raise FileNotFoundError("yt-dlp 下载后未找到任何音频文件。")
             os.rename(possible_files[0], output_path)
        print(f"URL 音频已下载到: {output_path}")
        return output_path
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"URL 下载失败: {e}")

def generate_and_test(audio_file, mic_input, youtube_url, text_to_speak):
    if not any([audio_file, mic_input, youtube_url]):
        raise gr.Error("请提供一个音频源：上传文件、录音或视频链接。")
    if not text_to_speak:
        raise gr.Error("请输入要试听的文本。")
    if speaker_model is None or tts_model is None:
        raise gr.Error("核心模型未能加载，应用无法工作。")
    waveform, source_name = None, "audio"
    if youtube_url:
        youtube_filepath = download_youtube_audio(youtube_url)
        waveform, source_name = process_audio_and_get_name(youtube_filepath, "YouTube")
    elif audio_file is not None:
        waveform, source_name = process_audio_and_get_name(audio_file.name, "file")
    elif mic_input is not None:
        waveform, source_name = process_audio_and_get_name(mic_input, "microphone_temp")
        source_name = "mic_recording"
    print("正在生成声纹...")
    with torch.no_grad():
        embedding = speaker_model.encode_batch(waveform.to(device))
        embedding = torch.nn.functional.normalize(embedding, dim=2)
        final_embedding = embedding[0][0]
    pt_filename = f"{source_name}_embedding.pt"
    torch.save(final_embedding, pt_filename)
    print(f"声纹文件已保存: {pt_filename}")
    print("正在进行TTS试听...")
    inputs = tts_processor(text=text_to_speak, return_tensors="pt").to(device)
    speaker_embeddings_for_tts = final_embedding.unsqueeze(0).to(device)
    speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings_for_tts, vocoder=tts_vocoder)
    test_audio_filename = f"test_{source_name}.wav"
    torchaudio.save(test_audio_filename, speech.cpu().unsqueeze(0), 16000)
    print(f"试听音频已生成: {test_audio_filename}")
    return pt_filename, test_audio_filename

# ---- 3. Gradio 界面定义
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 普罗米修斯旗舰声音实验室")
    gr.Markdown("在这里生产、并即时测试用于您 AI 大脑的任何声音。")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 提供声音源 (三选一)")
            with gr.Tabs():
                with gr.TabItem("📁 上传文件"):
                    audio_file_input = gr.File(label="支持 WAV, MP3, M4A 等格式")
                with gr.TabItem("🔗 视频平台链接"):
                    youtube_input = gr.Textbox(label="粘贴 YouTube, Bilibili, 抖音等 URL", placeholder="https://www.bilibili.com/video/BV...")
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
demo.launch()