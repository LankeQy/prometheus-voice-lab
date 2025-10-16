FROM python:3.10

WORKDIR /app

# 1. 安装基础“施工工具”
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装必要的库
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        # 第一部分：从 PyTorch 官方源头，安装一个保证互相兼容的全家桶
        torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
        # 第二部分：安装所有其他我们需要的库
        gradio \
        transformers>=4.30.0 \
        speechbrain==1.0.3 \
        soundfile \
        librosa \
        yt-dlp \
        ffmpeg-python \
        pyannote.audio

# 3. 复制所有应用代码
COPY . .

# 4. 设置环境变量
ENV HF_TOKEN=$HF_TOKEN

# 5. 暴露端口
EXPOSE 7860

# 6. 回归最可靠的 Python 启动方式
CMD ["python", "app.py"]