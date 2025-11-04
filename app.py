# app.py

import gradio as gr
import os
import uuid
import traceback
import soundfile as sf  # 用于保存合成的音频文件

# ---- 1. 初始状态 ----
# 打印启动信息，告知用户应用正在初始化。
print("应用脚本启动，将进行深度懒加载。")
print("Gradio界面将在一分钟内启动。")

# 全局变量，用于缓存已加载的模型和库，避免重复加载，提升性能。
CACHED_LIBS = {}
CACHED_EMBEDDING_MODEL = None
EMBEDDING_MODEL_LOADED = False
CACHED_TTS_COMPONENTS = {}  # 使用字典存储TTS模型的多个组件（处理器、主模型、声码器）
TTS_MODEL_LOADED = False


# ---- 2. 深度懒加载函数 ----
# 只有在用户第一次点击相关按钮时，才会执行这些耗时的加载操作。

def _load_heavy_libs_and_embedding_model():
    """
    加载用于“生成声纹”功能的库和模型。
    这个函数只在首次点击“生成”按钮时被调用一次。
    """
    global CACHED_LIBS, CACHED_EMBEDDING_MODEL, EMBEDDING_MODEL_LOADED
    if EMBEDDING_MODEL_LOADED:
        return

    print("首次操作：开始导入重量级库并加载声纹模型...")
    import torch
    import torchaudio
    from speechbrain.inference.classifiers import EncoderClassifier

    # 将导入的库缓存到全局变量中
    CACHED_LIBS['torch'] = torch
    CACHED_LIBS['torchaudio'] = torchaudio

    # 自动检测是否有可用的GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    try:
        # 加载预训练的声纹提取模型
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb",
            run_opts={"device": device}
        )
        CACHED_EMBEDDING_MODEL = model
        EMBEDDING_MODEL_LOADED = True
        print("✅ 声纹模型加载成功！")
    except Exception as e:
        raise gr.Error(f"声纹模型加载失败: {e}")


def _load_tts_model():
    """
    加载用于“语音合成试听”功能的库和模型 (Microsoft SpeechT5)。
    这个函数只在首次点击“试听”按钮时被调用一次。
    """
    global CACHED_TTS_COMPONENTS, TTS_MODEL_LOADED
    if TTS_MODEL_LOADED:
        return

    print("首次试听：开始加载 SpeechT5 语音合成模型...")
    import torch
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    try:
        # 加载 SpeechT5 的三个核心组件
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

        # 将组件存入缓存字典
        CACHED_TTS_COMPONENTS['processor'] = processor
        CACHED_TTS_COMPONENTS['model'] = model
        CACHED_TTS_COMPONENTS['vocoder'] = vocoder
        TTS_MODEL_LOADED = True
        print("✅ SpeechT5 语音合成模型加载成功！")
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"语音合成模型加载失败: {e}")


# ---- 3. 核心功能辅助函数 ----

def _process_audio(filepath, source_info):
    """
    统一处理所有来源的音频：转换为16kHz单声道，并返回PyTorch张量。
    """
    torch = CACHED_LIBS['torch']
    torchaudio = CACHED_LIBS['torchaudio']

    signal, fs = torchaudio.load(filepath)
    # 如果采样率不是16kHz，进行重采样
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)
    # 如果是多声道，混合为单声道
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    # 从文件名中提取基础名称
    source_name = os.path.splitext(os.path.basename(filepath))[0]

    # 删除临时的YouTube下载或麦克风录音文件
    if source_info in ["YouTube", "microphone_temp"]:
        try:
            os.remove(filepath)
        except Exception:
            pass

    return signal, source_name


def _download_youtube(youtube_url):
    """
    使用yt-dlp从YouTube等视频网站链接下载并提取音频。
    """
    import yt_dlp
    temp_filename = f"temp_audio_{uuid.uuid4().hex}"
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'outtmpl': temp_filename,
        'quiet': True,
        'nocheckcertificate': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    output_path = f"{temp_filename}.wav"
    # yt-dlp有时会生成不同的后缀，这里做一下兼容性处理
    if not os.path.exists(output_path):
        possible_files = [f for f in os.listdir('.') if f.startswith(temp_filename)]
        if not possible_files:
            raise FileNotFoundError("yt-dlp下载后未找到任何音频文件。")
        os.rename(possible_files[0], output_path)

    return output_path


# ---- 4. Gradio 事件处理函数 ----
# 这些是直接由Gradio界面按钮点击触发的函数。

def generate_embedding_wrapper(audio_file, mic_input, youtube_url, progress=gr.Progress()):
    """
    生成声纹文件的完整流程封装。
    """
    try:
        progress(0, desc="准备环境中（首次点击会较慢）...")
        # 确保声纹模型已加载
        _load_heavy_libs_and_embedding_model()

        progress(0.1, desc="检查输入...")
        if not any([audio_file, mic_input, youtube_url]):
            raise gr.Error("请提供一个音频源（上传文件、粘贴链接或麦克风录制）。")

        torch = CACHED_LIBS['torch']
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        waveform, source_name = None, "audio"

        progress(0.2, desc="处理音频源...")
        if youtube_url:
            filepath = _download_youtube(youtube_url)
            waveform, source_name = _process_audio(filepath, "YouTube")
        elif audio_file is not None:
            waveform, source_name = _process_audio(audio_file.name, "file")  # 使用 .name 获取gr.File的路径
        elif mic_input is not None:
            waveform, source_name = _process_audio(mic_input, "microphone_temp")
            source_name = "mic_recording"

        if waveform is None:
            raise gr.Error("无法加载音频，请检查输入源。")

        progress(0.6, desc="正在生成声纹...")
        with torch.no_grad():
            # 使用加载的模型生成声纹嵌入
            embedding = CACHED_EMBEDDING_MODEL.encode_batch(waveform.to(device))
            # 标准化处理
            embedding = torch.nn.functional.normalize(embedding, dim=2).squeeze()

        # 生成详细的质量验证报告
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

        # 保存声纹为.pt文件
        pt_filename = f"{source_name}_embedding.pt"
        torch.save(embedding, pt_filename)
        progress(1.0, desc="完成！")

        # 返回结果给UI：下载文件、报告文本、状态文件路径，并显示试听区
        return pt_filename, validation_report, pt_filename, gr.update(visible=True)

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"处理失败: {e}")


def synthesize_speech_wrapper(text_to_speak, pt_filepath, progress=gr.Progress()):
    """
    使用生成的声纹文件和SpeechT5合成语音的完整流程封装。
    """
    try:
        if not text_to_speak:
            raise gr.Error("请输入要合成的文本。")
        if not pt_filepath or not os.path.exists(pt_filepath):
            raise gr.Error("未找到有效的声纹文件。请先生成一个。")

        progress(0, desc="准备语音合成引擎（首次点击会较慢）...")
        # 确保TTS模型已加载
        _load_tts_model()

        progress(0.3, desc="加载声纹并处理文本...")
        torch = CACHED_LIBS['torch']
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # 从缓存中获取模型组件
        processor = CACHED_TTS_COMPONENTS['processor']
        model = CACHED_TTS_COMPONENTS['model']
        vocoder = CACHED_TTS_COMPONENTS['vocoder']

        # 步骤1: 使用处理器将文本转换为输入ID
        inputs = processor(text=text_to_speak, return_tensors="pt").to(device)

        # 步骤2: 加载我们生成的声纹文件
        speaker_embedding = torch.load(pt_filepath, map_location=device)
        # SpeechT5需要(1, 512)的形状，而我们的声纹是(512,)，所以用unsqueeze增加一个维度
        speaker_embedding = speaker_embedding.unsqueeze(0)

        progress(0.6, desc="正在生成语音频谱...")
        with torch.no_grad():
            # 步骤3: 模型根据文本和声纹生成频谱图
            spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embedding)

            progress(0.8, desc="通过声码器合成最终音频...")
            # 步骤4: 声码器将频谱图转换为实际的音频波形
            speech = vocoder(spectrogram)

        output_wav_path = f"synthesized_{uuid.uuid4().hex}.wav"
        # 使用 soundfile 库保存音频，指定16kHz采样率
        sf.write(output_wav_path, speech.cpu().numpy(), samplerate=16000)

        progress(1.0, desc="合成完毕！")
        return output_wav_path

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"语音合成失败: {e}")


# ---- 5. Gradio 界面定义 ----
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 普罗米修斯旗舰声音实验室")
    gr.Markdown("一个专业的在线声音克隆工具，您可以在这里生产、并即时测试用于您 AI 大脑的任何声音。")
    gr.Markdown("⚠️ **提示**: 首次点击“生成”或“试听”会初始化相应模型，**可能需要等待1-3分钟**。后续使用会很快。")

    # 定义一个隐藏的 State 组件，用于在“生成”和“试听”功能之间传递生成的 .pt 文件路径
    pt_file_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 提供声音源")
            with gr.Tabs():
                with gr.TabItem("📁 上传文件"):
                    audio_file_input = gr.File(label="支持 WAV, MP3, M4A 等")
                with gr.TabItem("🔗 视频平台链接"):
                    youtube_input = gr.Textbox(label="粘贴来自 YouTube, Bilibili, 抖音等网站的 URL")
                with gr.TabItem("🎤 麦克风录制"):
                    mic_input = gr.Audio(sources=["microphone"], type="filepath", label="点击录制")

            generate_btn = gr.Button("生成并验证声纹文件", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 2. 下载并验证结果")
            pt_output = gr.File(label="下载声纹 (.pt 文件)")
            validation_output = gr.Textbox(label="声纹质量报告", lines=5, interactive=False)

    # 试听区域，默认隐藏，在生成声纹后显示
    with gr.Group(visible=False) as tts_box:
        gr.Markdown("---")
        gr.Markdown("### 3. 即时试听克隆效果 (由 Microsoft SpeechT5 驱动)")
        with gr.Row():
            text_input = gr.Textbox(
                label="输入要合成的文本 (支持中英文)",
                value="你好，世界。这是一个由微软语音模型克隆的声音。"
            )
            synthesize_btn = gr.Button("合成并试听", variant="primary")

        audio_output = gr.Audio(label="合成结果试听", type="filepath")

    # 定义按钮点击事件的逻辑
    generate_btn.click(
        fn=generate_embedding_wrapper,
        inputs=[audio_file_input, mic_input, youtube_input],
        # 输出会更新：下载文件组件、报告文本框、隐藏的State，以及试听区的可见性
        outputs=[pt_output, validation_output, pt_file_state, tts_box]
    )

    synthesize_btn.click(
        fn=synthesize_speech_wrapper,
        inputs=[text_input, pt_file_state],
        outputs=[audio_output]
    )

# ---- 6. 启动应用 ----
# 使用 server_name="0.0.0.0" 以便在 Docker 容器和 Hugging Face Spaces 中正确运行
demo.launch(server_name="0.0.0.0", server_port=7860)

print("✅ Gradio 服务已启动，应用正在等待用户操作。")