# "语法修正版" Dockerfile - 不再有任何愚蠢的错误
FROM python:3.10

WORKDIR /app

# 1. 安装基础“施工工具”
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装PyTorch 和所有依赖
# 我们在一个 RUN 命令中，完成所有事情。
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        # 所有包都在这一个 pip install 命令中，作为它的参数
        torch==2.1.0 \
        torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cpu \
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