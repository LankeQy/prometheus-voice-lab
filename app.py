# app.py
import gradio as gr
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
import yt_dlp
import os
import uuid
import traceback

# ---- 1. 模型加载 ----
print("正在加载声纹提取模型...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

try:
    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb",
        run_opts={"device": device}
    )
    print("✅ 声纹提取模型加载成功！应用准备就绪。")
except Exception as e:
    print(f"🔴 声纹提取模型加载失败: {e}")
    speaker_model = None


# ---- 2. 核心功能函数 ----
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
    ydl_opts = {'format': 'bestaudio/best', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
                'outtmpl': temp_filename, 'quiet': True, 'nocheckcertificate': True}
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


def generate_embedding_only(audio_file, mic_input, youtube_url, progress=gr.Progress()):
    """
    主函数：生成声纹文件，并附带质量验证报告。
    """
    progress(0, desc="检查输入...")
    if not any([audio_file, mic_input, youtube_url]):
        raise gr.Error("请提供一个音频源：上传文件、录音或视频链接。")
    if speaker_model is None:
        raise gr.Error("核心模型未能加载，应用无法工作。请检查启动日志。")

    waveform, source_name = None, "audio"
    progress(0.2, desc="处理音频源...")
    if youtube_url:
        youtube_filepath = download_youtube_audio(youtube_url)
        waveform, source_name = process_audio_and_get_name(youtube_filepath, "YouTube")
    elif audio_file is not None:
        waveform, source_name = process_audio_and_get_name(audio_file, "file")
    elif mic_input is not None:
        waveform, source_name = process_audio_and_get_name(mic_input, "microphone_temp")
        source_name = "mic_recording"

    if waveform is None:
        raise gr.Error("无法从提供的源加载音频。")

    progress(0.6, desc="正在生成声纹...")
    with torch.no_grad():
        embedding = speaker_model.encode_batch(waveform.to(device))
        embedding = torch.nn.functional.normalize(embedding, dim=2)
        final_embedding = embedding.squeeze()

    # ---- 新增：验证步骤 ----
    validation_report = ""
    is_healthy = True
    try:
        # 1. 检查形状
        shape = final_embedding.shape
        if len(shape) == 1 and shape[0] == 512:
            validation_report += f"✅ 形状正确: {shape}\n"
        else:
            validation_report += f"❌ 形状错误: {shape} (应为 [512])\n"
            is_healthy = False

        # 2. 检查数值
        if torch.isnan(final_embedding).any() or torch.isinf(final_embedding).any():
            validation_report += "❌ 向量中包含无效值 (NaN/inf)\n"
            is_healthy = False
        else:
            validation_report += "✅ 数值有效 (无 NaN/inf)\n"

        # 3. 检查范数 (模长)
        norm = torch.linalg.norm(final_embedding).item()
        if 0.99 < norm < 1.01:
            validation_report += f"✅ 归一化成功 (向量模长 ≈ {norm:.4f})\n"
        else:
            validation_report += f"❌ 归一化失败 (向量模长 = {norm:.4f}，应接近1)\n"
            is_healthy = False

        if not is_healthy:
            validation_report += "\n⚠️ 警告: 声纹文件可能无效，请检查源音频质量（如是否静音、噪音过大等）。"

    except Exception as e:
        validation_report = f"验证过程中出现异常: {e}"
    # ---- 验证结束 ----

    pt_filename = f"{source_name}_embedding.pt"
    torch.save(final_embedding, pt_filename)
    print(f"声纹文件已保存: {pt_filename}")
    progress(1.0, desc="完成！")

    return pt_filename, validation_report


# ---- 3. Gradio 界面定义 (带验证报告) ----
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 普罗米修斯声纹提取器")
    gr.Markdown("一个专注、高效的工具，用于为您的 AI 助手生产 `.pt` 声纹文件。")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 提供声音源 (三选一)")
            gr.Markdown("建议使用 **5-30秒** 的**清晰、无背景噪音**的音频以获得最佳效果。")
            with gr.Tabs():
                with gr.TabItem("📁 上传文件"):
                    audio_file_input = gr.File(label="支持 WAV, MP3, M4A 等格式")
                with gr.TabItem("🔗 视频平台链接"):
                    youtube_input = gr.Textbox(label="粘贴 YouTube, Bilibili, 抖音等 URL",
                                               placeholder="https://www.bilibili.com/video/BV...")
                with gr.TabItem("🎤 麦克风录制"):
                    mic_input = gr.Audio(sources=["microphone"], type="filepath", label="点击录制你的声音")

            generate_btn = gr.Button("生成并验证声纹文件", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 2. 下载并验证结果")
            pt_output = gr.File(label="下载声纹 (.pt 文件)")
            # 新增一个文本框来显示验证报告
            validation_output = gr.Textbox(
                label="声纹质量报告 (自动生成)",
                lines=5,
                interactive=False
            )

    generate_btn.click(
        fn=generate_embedding_only,
        inputs=[audio_file_input, mic_input, youtube_input],
        # 将输出绑定到两个组件上
        outputs=[pt_output, validation_output],
        api_name="generate_embedding"
    )

# ---- 4. 启动应用 ----
demo.launch()