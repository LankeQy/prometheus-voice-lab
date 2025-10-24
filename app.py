# app.py
import gradio as gr
import os
import uuid
import traceback

# ---- 1. 初始状态 ----
# 注意：这里我们只导入最轻量级的标准库。
# 所有重量级库 (torch, torchaudio, speechbrain) 都会在需要时才导入。
print("应用脚本启动，将进行深度懒加载。")
print("Gradio界面将在一分钟内启动。")

# 全局变量来缓存已加载的模型和库
CACHED_LIBS = {}
CACHED_MODEL = None
MODEL_LOADED = False


# ---- 2. 深度懒加载函数 ----
def _load_heavy_libs_and_model():
    """
    一个函数负责所有重量级库的导入和模型的加载。
    这个函数只在用户首次点击时被调用一次。
    """
    global CACHED_LIBS, CACHED_MODEL, MODEL_LOADED

    if MODEL_LOADED:
        return

    print("首次操作：开始导入重量级库...")
    # 在函数内部导入，避免在启动时执行
    import torch
    import torchaudio
    from speechbrain.inference.classifiers import EncoderClassifier

    CACHED_LIBS['torch'] = torch
    CACHED_LIBS['torchaudio'] = torchaudio
    CACHED_LIBS['EncoderClassifier'] = EncoderClassifier
    print("✅ 重量级库导入成功。")

    print("首次操作：开始加载声纹模型...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb",
            run_opts={"device": device}
        )
        CACHED_MODEL = model
        MODEL_LOADED = True
        print("✅ 声纹模型加载成功！")
    except Exception as e:
        print(f"🔴 模型加载失败: {e}")
        raise gr.Error(f"核心模型加载失败: {e}")


# ---- 3. 核心功能函数 ----
def _process_audio(filepath, source_info):
    # 使用缓存的库
    torch = CACHED_LIBS['torch']
    torchaudio = CACHED_LIBS['torchaudio']

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


# ---- 4. Gradio 事件处理函数 ----
def generate_embedding_wrapper(audio_file, mic_input, youtube_url, progress=gr.Progress()):
    """
    这是Gradio直接调用的函数。它的唯一职责是：
    1. 确保模型和库已加载。
    2. 调用核心逻辑。
    3. 处理UI反馈。
    """
    try:
        progress(0, desc="准备环境中（首次点击会较慢）...")
        # 深度懒加载发生在此处！
        _load_heavy_libs_and_model()

        progress(0.1, desc="检查输入...")
        if not any([audio_file, mic_input, youtube_url]):
            raise gr.Error("请提供一个音频源。")

        torch = CACHED_LIBS['torch']
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        waveform, source_name = None, "audio"
        progress(0.2, desc="处理音频源...")
        if youtube_url:
            filepath = _download_youtube(youtube_url)
            waveform, source_name = _process_audio(filepath, "YouTube")
        elif audio_file is not None:
            waveform, source_name = _process_audio(audio_file, "file")
        elif mic_input is not None:
            waveform, source_name = _process_audio(mic_input, "microphone_temp")
            source_name = "mic_recording"

        if waveform is None: raise gr.Error("无法加载音频。")

        progress(0.6, desc="正在生成声纹...")
        with torch.no_grad():
            embedding = CACHED_MODEL.encode_batch(waveform.to(device))
            embedding = torch.nn.functional.normalize(embedding, dim=2)
            final_embedding = embedding.squeeze()

        # 验证
        validation_report = ""
        shape = final_embedding.shape
        if len(shape) == 1 and shape[0] == 512:
            validation_report += f"✅ 形状正确: {shape}\n"
        else:
            validation_report += f"❌ 形状错误: {shape}\n"
        if not (torch.isnan(final_embedding).any() or torch.isinf(final_embedding).any()):
            validation_report += "✅ 数值有效\n"
        else:
            validation_report += "❌ 向量中包含无效值\n"
        norm = torch.linalg.norm(final_embedding).item()
        if 0.99 < norm < 1.01:
            validation_report += f"✅ 归一化成功 (模长 ≈ {norm:.4f})\n"
        else:
            validation_report += f"❌ 归一化失败 (模长 = {norm:.4f})\n"

        pt_filename = f"{source_name}_embedding.pt"
        torch.save(final_embedding, pt_filename)
        progress(1.0, desc="完成！")

        return pt_filename, validation_report

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"处理失败: {e}")


# ---- 5. Gradio 界面定义 ----
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 普罗米修斯声纹提取器")
    gr.Markdown("一个专注、高效的工具，用于为您的 AI 助手生产 `.pt` 声纹文件。")
    gr.Markdown("⚠️ **提示**: 首次点击“生成”会初始化环境，**可能需要等待2-5分钟**。后续生成会很快。")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 提供声音源")
            with gr.Tabs():
                with gr.TabItem("📁 上传文件"): audio_file_input = gr.File(label="支持 WAV, MP3 等")
                with gr.TabItem("🔗 视频平台链接"): youtube_input = gr.Textbox(label="粘贴 URL")
                with gr.TabItem("🎤 麦克风录制"): mic_input = gr.Audio(sources=["microphone"], type="filepath",
                                                                      label="点击录制")
            generate_btn = gr.Button("生成并验证声纹文件", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 2. 下载并验证结果")
            pt_output = gr.File(label="下载声纹 (.pt 文件)")
            validation_output = gr.Textbox(label="声纹质量报告", lines=5, interactive=False)

    generate_btn.click(
        fn=generate_embedding_wrapper,
        inputs=[audio_file_input, mic_input, youtube_input],
        outputs=[pt_output, validation_output]
    )

# ---- 6. 启动应用 ----
# 现在的启动路径非常轻量，应该能在30秒内完成
demo.launch()
print("✅ Gradio 服务已启动，应用正在等待用户操作。")