# app.py (Super-Flagship Version - Unabridged)
import gradio as gr
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from pyannote.audio import Pipeline
import yt_dlp
import os
import uuid
import traceback

# ---- 1. 模型加载 ----
print("正在加载所有模型，这将需要几分钟...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 从 Hugging Face Space 的 Secrets 中安全地获取令牌
HF_TOKEN = os.environ.get("HF_TOKEN")

# 声纹提取模型
try:
    speaker_model = SpeakerRecognition.from_huggingface("speechbrain/spkrec-xvect-voxceleb",
                                                        run_opts={"device": device})
    print("✅ 声纹提取模型加载成功！")
except Exception as e:
    print(f"🔴 声纹提取模型加载失败: {e}")
    speaker_model = None

# TTS 试听模型
try:
    tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    print("✅ TTS 试听模型加载成功！")
except Exception as e:
    print(f"🔴 TTS 试听模型加载失败: {e}")
    tts_model = None

# VAD "智能副驾" 模型
if HF_TOKEN:
    try:
        vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=HF_TOKEN)
        vad_pipeline.to(torch.device(device))
        print("✅ VAD '智能副-驾' 模型加载成功！")
    except Exception as e:
        print(f"🔴 VAD '智能副驾' 模型加载失败: {e}")
        vad_pipeline = None
else:
    vad_pipeline = None
    print("🔴 警告: 未在 Space Secrets 中找到 HF_TOKEN。自动裁剪功能将不可用。")

print("✅ 所有模型加载完毕，应用准备就绪！")


# ---- 2. 核心功能函数 ----

def find_best_audio_segment(filepath: str, target_duration: float = 25.0):
    """
    使用 VAD 模型找到音频中最长的连续语音片段，并裁剪成目标时长。
    返回一个 torch.Tensor 波形。
    """
    if vad_pipeline is None:
        raise gr.Error("VAD 模型未加载，无法自动裁剪。请检查 HF_TOKEN Secret 设置及服务器日志。")

    print("VAD 正在分析音频以寻找最佳片段...")
    try:
        waveform, sample_rate = torchaudio.load(filepath)

        # 确保采样率是16kHz，因为VAD模型需要它
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # 运行VAD
        output = vad_pipeline({"waveform": waveform, "sample_rate": sample_rate})
        speech_regions = output.get_timeline().support()

        if not speech_regions:
            raise ValueError("在音频中未检测到任何语音活动。")

        # 合并间隙小于0.5秒的语音片段
        merged_regions = []
        if speech_regions:
            current_region = speech_regions[0]
            for next_region in speech_regions[1:]:
                if next_region.start - current_region.end < 0.5:
                    current_region = current_region.union(next_region)
                else:
                    merged_regions.append(current_region)
                    current_region = next_region
            merged_regions.append(current_region)

        # 找到最长的合并后的片段
        longest_region = max(merged_regions, key=lambda region: region.duration)
        print(
            f"找到最长语音片段：时长 {longest_region.duration:.2f} 秒，从 {longest_region.start:.2f}s 到 {longest_region.end:.2f}s。")

        # 从这个片段中裁剪出目标时长的部分
        start_frame = int(longest_region.start * sample_rate)
        end_frame = int(longest_region.end * sample_rate)

        # 如果片段本身比目标长，就从中间截取
        if longest_region.duration > target_duration:
            mid_frame = (start_frame + end_frame) // 2
            half_duration_frames = int(target_duration * sample_rate / 2)
            start_frame = max(0, mid_frame - half_duration_frames)
            end_frame = mid_frame + half_duration_frames

        clipped_waveform = waveform[:, start_frame:end_frame]

        print(f"已自动裁剪出 {clipped_waveform.shape[1] / sample_rate:.2f} 秒的黄金片段。")
        return clipped_waveform, sample_rate

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"VAD 智能分析失败: {e}")


def download_and_process_link(url: str):
    """从链接下载并返回文件路径"""
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return None
    print(f"正在从 URL 下载: {url}")
    temp_filename = f"temp_audio_{uuid.uuid4().hex}"

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'outtmpl': temp_filename,
        'quiet': True,
        'nocheckcertificate': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        output_path = f"{temp_filename}.wav"
        if not os.path.exists(output_path):
            possible_files = [f for f in os.listdir('.') if f.startswith(temp_filename)]
            if not possible_files:
                raise FileNotFoundError("yt-dlp 下载后未找到任何音频文件。")
            os.rename(possible_files[0], output_path)

        print(f"URL 音频已下载到: {output_path}")
        return output_path
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"URL 下载失败: {e}")


def generate_and_test(audio_file, mic_audio, youtube_url, text_to_speak, auto_clip):
    """终极函数：处理所有输入源，生成声纹，并进行TTS测试"""
    if not any([audio_file, mic_audio, youtube_url]):
        raise gr.Error("请提供一个音频源：上传文件、录音或视频链接。")
    if not text_to_speak:
        raise gr.Error("请输入要试听的文本。")
    if speaker_model is None or tts_model is None:
        raise gr.Error("核心模型未能加载，应用无法工作。请检查服务器日志。")

    filepath = None
    source_name = "audio"
    is_temp_file = False

    try:
        # 1. 获取音频文件路径
        if youtube_url:
            filepath = download_and_process_link(youtube_url)
            source_name = os.path.splitext(os.path.basename(filepath))[0]
            is_temp_file = True
        elif audio_file is not None:
            filepath = audio_file.name
            source_name = os.path.splitext(os.path.basename(filepath))[0]
        elif mic_audio is not None:
            filepath = mic_audio
            source_name = "mic_recording"
            is_temp_file = True  # Gradio的麦克风录音也是临时文件

        if not filepath:
            raise gr.Error("无法获取有效的音频源。")

        # 2. 核心步骤：加载并处理音频波形
        if auto_clip:
            waveform, sample_rate = find_best_audio_segment(filepath)
        else:
            print("用户选择手动模式，正在加载完整音频...")
            waveform, sample_rate = torchaudio.load(filepath)

        # 确保是16kHz, 单声道
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. 生成声纹
        print("正在生成声纹...")
        with torch.no_grad():
            embedding = speaker_model.encode_batch(waveform.to(device))
            embedding = torch.nn.functional.normalize(embedding, dim=2)
            final_embedding = embedding[0][0]

        pt_filename = f"{source_name}_embedding.pt"
        torch.save(final_embedding, pt_filename)
        print(f"声纹文件已保存: {pt_filename}")

        # 4. 进行TTS试听
        print("正在进行TTS试听...")
        inputs = tts_processor(text=text_to_speak, return_tensors="pt").to(device)
        speaker_embeddings_for_tts = final_embedding.unsqueeze(0).to(device)

        speech = tts_model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings_for_tts,
            vocoder=tts_vocoder
        )

        test_audio_filename = f"test_{source_name}.wav"
        torchaudio.save(test_audio_filename, speech.cpu().unsqueeze(0), 16000)
        print(f"试听音频已生成: {test_audio_filename}")

        return pt_filename, test_audio_filename

    finally:
        # 5. 清理所有临时文件
        if is_temp_file and filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"已清理临时文件: {filepath}")
            except Exception as e:
                print(f"清理临时文件失败: {e}")


# ---- 3. Gradio 界面定义 (超旗舰版) ----
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 普罗米修斯超旗舰声音实验室")
    gr.Markdown("智能提取、一键克隆、即时试听——在这里，打造您AI的完美声音。")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 提供声音源 (三选一)")
            with gr.Tabs():
                with gr.TabItem("📁 上传文件"):
                    audio_file_input = gr.File(label="支持 WAV, MP3, M4A 等格式")
                with gr.TabItem("🔗 视频平台链接"):
                    youtube_input = gr.Textbox(label="粘贴 YouTube, Bilibili, 抖音等 URL",
                                               placeholder="https://www.bilibili.com/video/BV...")
                with gr.TabItem("🎤 麦克风录制"):
                    mic_input = gr.Audio(sources=["microphone"], type="filepath", label="点击录制你的声音")

            # 智能副驾开关
            auto_clip_checkbox = gr.Checkbox(label="🤖 自动提取25秒黄金片段 (推荐)", value=True,
                                             interactive=vad_pipeline is not None)
            if vad_pipeline is None:
                gr.Markdown("<p style='color:red;'>🔴 自动提取功能不可用，请检查 HF_TOKEN Secret 设置。</p>")

            gr.Markdown("### 2. 输入试听文本")
            text_input = gr.Textbox(label="输入要试听的中文", value="你好，我是普罗米修斯。这是我的新声音。")

            generate_btn = gr.Button("生成并试听", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 3. 获取结果")
            pt_output = gr.File(label="下载声纹 (.pt 文件)")
            audio_output = gr.Audio(label="试听克隆效果", type="filepath")

    generate_btn.click(
        fn=generate_and_test,
        inputs=[audio_file_input, mic_input, youtube_input, text_input, auto_clip_checkbox],
        outputs=[pt_output, audio_output],
        api_name="generate"
    )

# ---- 4. 启动应用 ----
demo.launch()