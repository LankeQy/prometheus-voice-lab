# app.py (最终功能实现版 - 绝对成功)

import gradio as gr
import os
import uuid
import traceback
import torch
import soundfile as sf
from pydub import AudioSegment

# *** 终极修复：直接导入并使用底层的 XTTS 模型类，以精确控制每一步 ***
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager

# ---- 1. 启动时加载 XTTS 模型 ----
print("应用脚本启动，开始加载 Coqui XTTS v2 模型...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用的设备: {DEVICE}")

try:
    print("正在从本地缓存初始化 XTTS 模型...")
    # 动态获取 TTS 库的缓存路径
    mm = ModelManager()
    model_path_in_cache = os.path.join(mm.get_models_file_path(), "..",
                                       "tts_models--multilingual--multi-dataset--xtts_v2")

    config_path = os.path.join(model_path_in_cache, "config.json")
    if not os.path.exists(config_path):
        # 如果缓存找不到，就用高层 API 触发一次下载
        print(f"在 {model_path_in_cache} 未找到模型，尝试从网络下载...")
        from TTS.api import TTS

        TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
        # 再次检查
        if not os.path.exists(config_path):
            raise FileNotFoundError("模型文件下载失败或缓存路径不正确。")

    # 使用底层 API 精确加载
    config = XttsConfig()
    config.load_json(config_path)
    TTS_MODEL = Xtts.init_from_config(config)
    TTS_MODEL.load_checkpoint(config, checkpoint_dir=model_path_in_cache, eval=True)
    TTS_MODEL.to(DEVICE)
    print("✅ Coqui XTTS v2 模型通过底层 API 加载成功！")
except Exception as e:
    print(f"🔴 XTTS 模型加载失败: {e}");
    raise e


# ---- 2. 核心功能辅助函数 ----
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
def generate_embedding_wrapper(audio_file, mic_input, youtube_input, progress=gr.Progress()):
    temp_files = []
    try:
        progress(0.2, desc="处理音频源...")
        source_wav_path = _process_audio_source(audio_file, mic_input, youtube_input)
        if source_wav_path is None: raise gr.Error("请提供一个有效的音频源。")
        temp_files.append(source_wav_path)

        progress(0.6, desc="正在提取 XTTS 完整声纹...")
        with torch.no_grad():
            gpt_cond_latent, speaker_embedding = TTS_MODEL.get_conditioning_latents(audio_path=source_wav_path)

        # 移动到 CPU 以便保存
        gpt_cond_latent_cpu = gpt_cond_latent.squeeze(0).cpu()
        speaker_embedding_cpu = speaker_embedding.squeeze(0).cpu()

        source_name = os.path.splitext(os.path.basename(audio_file.name if audio_file else "recording"))[0]
        pt_filename = f"{source_name}_xtts_embedding.pt"
        # 将两个关键向量都保存在一个字典里
        torch.save({"gpt_cond_latent": gpt_cond_latent_cpu, "speaker_embedding": speaker_embedding_cpu}, pt_filename)

        progress(1.0, desc="声纹提取完毕！")
        report = f"✅ 声纹提取成功！\n"
        report += f"✅ Speaker Embedding 形状: {speaker_embedding_cpu.shape}\n"
        report += f"✅ 文件已保存为: {pt_filename}"
        return pt_filename, report, pt_filename, gr.update(visible=True)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"声纹提取失败: {e}")
    finally:
        for f in temp_files:
            if f and os.path.exists(f): os.remove(f)


def synthesize_speech_wrapper(pt_filepath, text, language, progress=gr.Progress()):
    try:
        progress(0.1, desc="检查输入...")
        if not text: raise gr.Error("请输入要合成的文本。")
        if not pt_filepath or not os.path.exists(pt_filepath):
            raise gr.Error("未找到声纹文件。请先在步骤1中生成一个。")

        progress(0.3, desc="加载声纹并准备合成...")
        latents = torch.load(pt_filepath, map_location=DEVICE)
        gpt_cond_latent = latents["gpt_cond_latent"].unsqueeze(0)
        speaker_embedding = latents["speaker_embedding"].unsqueeze(0)

        progress(0.6, desc="正在合成语音...")
        output_wav_path = f"synthesized_{uuid.uuid4().hex}.wav"

        with torch.no_grad():
            # 调用模型的底层 inference 方法
            out = TTS_MODEL.inference(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                temperature=0.7,
            )
            wav = out["wav"]

        # 使用 soundfile 库保存，这是最稳健的方式
        sf.write(output_wav_path, wav, 24000)

        progress(1.0, desc="合成完毕！")
        return output_wav_path
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"语音合成失败: {e}")


# ---- 4. Gradio 界面定义 ----
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 普罗米修斯旗舰声音实验室 (多语言最终版)")
    gr.Markdown("一个支持中、日、英等多种语言的高质量在线声音克隆工具。由 Coqui XTTS v2 驱动。")
    gr.Markdown("✅ **环境已就绪**，XTTS 模型已加载完毕。")
    pt_file_state = gr.State(value=None)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 提供声音源并生成声纹")
            with gr.Tabs():
                with gr.TabItem("📁 上传文件"): audio_file_input = gr.File(label="支持 WAV, MP3, M4A 等")
                with gr.TabItem("🔗 视频平台链接"): youtube_input = gr.Textbox(label="粘贴 URL")
                with gr.TabItem("🎤 麦克风录制"): mic_input = gr.Audio(sources=["microphone"], type="filepath",
                                                                      label="点击录制")
            generate_btn = gr.Button("生成声纹 (.pt 文件)", variant="primary")
        with gr.Column(scale=1):
            gr.Markdown("### 2. 下载声纹文件")
            pt_output = gr.File(label="下载声纹文件")
            report_output = gr.Textbox(label="处理报告", lines=5, interactive=False)
    with gr.Group(visible=False) as tts_box:
        gr.Markdown("---")
        gr.Markdown("### 3. 使用声纹合成语音")
        text_input = gr.Textbox(label="输入要合成的文本", value="你好，世界。こんにちは、世界。Hello world.", lines=3)
        lang_dropdown = gr.Dropdown(
            choices=["zh-cn", "ja", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "hu", "ko"],
            value="zh-cn", label="选择语言")
        synthesize_btn = gr.Button("使用生成的声纹合成", variant="primary")
        audio_output = gr.Audio(label="合成结果试听", type="filepath")
    generate_btn.click(fn=generate_embedding_wrapper, inputs=[audio_file_input, mic_input, youtube_input],
                       outputs=[pt_output, report_output, pt_file_state, tts_box])
    synthesize_btn.click(fn=synthesize_speech_wrapper, inputs=[pt_file_state, text_input, lang_dropdown],
                         outputs=[audio_output])

# ---- 5. 启动应用 ----
print("所有模型加载完毕，正在启动Gradio服务...")
demo.launch(server_name="0.0.0.0", server_port=7860)
print("✅ Gradio 服务已启动。")