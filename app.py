# app.py (最终修复版 - 移除懒加载)

import gradio as gr
import os
import uuid
import traceback
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# ---- 1. 启动时直接加载所有模型 ----
print("应用脚本启动，开始加载所有模型...")
print("这可能需要2-5分钟，请耐心等待 Gradio 界面出现...")

# 自动检测设备
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"使用的设备: {DEVICE}")

# 加载声纹提取模型 (ECAPA-TDNN)
try:
    print("正在加载 ECAPA-TDNN 声纹模型...")
    EMBEDDING_MODEL = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE}
    )
    print("✅ ECAPA-TDNN 声纹模型加载成功！")
except Exception as e:
    print(f"🔴 声纹模型加载失败: {e}")
    raise e

# 加载语音合成模型 (SpeechT5)
try:
    print("正在加载 SpeechT5 语音合成模型...")
    TTS_PROCESSOR = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    TTS_MODEL = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(DEVICE)
    VOCODER = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)
    print("✅ SpeechT5 语音合成模型加载成功！")
except Exception as e:
    print(f"🔴 SpeechT5 模型加载失败: {e}")
    raise e


# ---- 2. 核心功能辅助函数 ----
def _process_audio(filepath, source_info):
    signal, fs = torchaudio.load(filepath)
    if fs != 16000:
        signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(signal)
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    source_name = os.path.splitext(os.path.basename(filepath))[0]
    if source_info in ["YouTube", "microphone_temp"]:
        try:
            os.remove(filepath)
        except Exception:
            pass
    return signal, source_name


def _download_youtube(youtube_url):
    import yt_dlp
    temp_filename = f"temp_audio_{uuid.uuid4().hex}"
    ydl_opts = {'format': 'bestaudio/best', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
                'outtmpl': temp_filename, 'quiet': True, 'nocheckcertificate': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    output_path = f"{temp_filename}.wav"
    if not os.path.exists(output_path):
        possible_files = [f for f in os.listdir('.') if f.startswith(temp_filename)]
        if not possible_files: raise FileNotFoundError("yt-dlp下载后未找到任何音频文件。")
        os.rename(possible_files[0], output_path)
    return output_path


# ---- 3. Gradio 事件处理函数 ----
def generate_embedding_wrapper(audio_file, mic_input, youtube_url, progress=gr.Progress()):
    try:
        progress(0.1, desc="检查输入...")
        if not any([audio_file, mic_input, youtube_url]): raise gr.Error("请提供一个音频源。")

        waveform, source_name = None, "audio"

        progress(0.2, desc="处理音频源...")
        if youtube_url:
            filepath = _download_youtube(youtube_url)
            waveform, source_name = _process_audio(filepath, "YouTube")
        elif audio_file is not None:
            waveform, source_name = _process_audio(audio_file.name, "file")
        elif mic_input is not None:
            waveform, source_name = _process_audio(mic_input, "microphone_temp")
            source_name = "mic_recording"
        if waveform is None: raise gr.Error("无法加载音频。")

        progress(0.6, desc="正在生成兼容性声纹...")
        with torch.no_grad():
            # ECAPA-TDNN模型期望一个3D张量 (batch, time, channels)
            # 我们的 waveform 是 2D (1, time)，所以需要增加一个批次维度
            waveform_3d = waveform.unsqueeze(0).to(DEVICE)

            #  3D 张量喂给模型
            model_out = EMBEDDING_MODEL.mods.embedding_model(waveform_3d)

            if isinstance(model_out, tuple):
                embedding_512d = model_out[-2].squeeze(0)
            else:
                embedding_512d = model_out.squeeze(0)
            embedding = torch.nn.functional.normalize(embedding_512d, dim=-1).squeeze()

        validation_report = ""
        shape = embedding.shape
        if len(shape) == 1 and shape[0] == 512:
            validation_report += f"✅ 形状正确: {shape}\n"
        else:
            validation_report += f"❌ 形状错误: {shape} (应为 512)\n"
        if not (torch.isnan(embedding).any() or torch.isinf(embedding).any()):
            validation_report += "✅ 数值有效 (无NaN或Inf)\n"
        else:
            validation_report += "❌ 向量中包含无效值\n"
        norm = torch.linalg.norm(embedding).item()
        if 0.99 < norm < 1.01:
            validation_report += f"✅ 归一化成功 (模长 ≈ {norm:.4f})\n"
        else:
            validation_report += f"❌ 归一化失败 (模长 = {norm:.4f})\n"

        pt_filename = f"{source_name}_embedding.pt"
        torch.save(embedding, pt_filename)
        progress(1.0, desc="完成！")
        return pt_filename, validation_report, pt_filename, gr.update(visible=True)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"处理失败: {e}")


def synthesize_speech_wrapper(text_to_speak, pt_filepath, progress=gr.Progress()):
    try:
        if not text_to_speak: raise gr.Error("请输入要合成的文本。")
        if not pt_filepath or not os.path.exists(pt_filepath): raise gr.Error("未找到有效的声纹文件。")

        progress(0.3, desc="加载声纹并处理文本...")
        inputs = TTS_PROCESSOR(text=text_to_speak, return_tensors="pt").to(DEVICE)
        speaker_embedding = torch.load(pt_filepath, map_location=DEVICE).unsqueeze(0)

        progress(0.6, desc="正在生成语音频谱...")
        with torch.no_grad():
            spectrogram = TTS_MODEL.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embedding)
            progress(0.8, desc="通过声码器合成最终音频...")
            speech = VOCODER(spectrogram)

        output_wav_path = f"synthesized_{uuid.uuid4().hex}.wav"
        sf.write(output_wav_path, speech.cpu().numpy(), samplerate=16000)
        progress(1.0, desc="合成完毕！")
        return output_wav_path
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"语音合成失败: {e}")


# ---- 4. Gradio 界面定义 ----
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 普罗米修斯旗舰声音实验室")
    gr.Markdown("一个专业的在线声音克隆工具，您可以在这里生产、并即时测试用于您 AI 大脑的任何声音。")
    gr.Markdown("✅ **环境已就绪**，所有模型均已加载完毕，您可以立即开始使用。")

    pt_file_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 提供声音源")
            with gr.Tabs():
                with gr.TabItem("📁 上传文件"):
                    audio_file_input = gr.File(label="支持 WAV, MP3, M4A 等")
                with gr.TabItem("🔗 视频平台链接"):
                    # 修复：将变量名从 youtube_url 改为 youtube_input
                    youtube_input = gr.Textbox(label="粘贴来自 YouTube, Bilibili, 抖音等网站的 URL")
                with gr.TabItem("🎤 麦克风录制"):
                    mic_input = gr.Audio(sources=["microphone"], type="filepath", label="点击录制")
            generate_btn = gr.Button("生成并验证声纹文件", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 2. 下载并验证结果")
            pt_output = gr.File(label="下载声纹 (.pt 文件)")
            validation_output = gr.Textbox(label="声纹质量报告", lines=5, interactive=False)

    with gr.Group(visible=False) as tts_box:
        gr.Markdown("---")
        gr.Markdown("### 3. 即时试听克隆效果 (由 Microsoft SpeechT5 驱动)")
        with gr.Row():
            text_input = gr.Textbox(label="输入要合成的文本 (支持中英文)",
                                    value="你好，世界。这是一个由微软语音模型克隆的声音。")
            synthesize_btn = gr.Button("合成并试听", variant="primary")
        audio_output = gr.Audio(label="合成结果试听", type="filepath")

    # 修复：在 inputs 列表中使用正确的变量名 youtube_input
    generate_btn.click(
        fn=generate_embedding_wrapper,
        inputs=[audio_file_input, mic_input, youtube_input],
        outputs=[pt_output, validation_output, pt_file_state, tts_box]
    )

    synthesize_btn.click(
        fn=synthesize_speech_wrapper,
        inputs=[text_input, pt_file_state],
        outputs=[audio_output]
    )

# ---- 5. 启动应用 ----
print("所有模型加载完毕，正在启动Gradio服务...")
demo.launch(server_name="0.0.0.0", server_port=7860)

print("✅ Gradio 服务已启动，应用正在等待用户操作。")