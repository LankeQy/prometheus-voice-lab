
FROM python:3.10 as builder

WORKDIR /install

# 1. 安装基础“施工工具”
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 2. 升级 pip
RUN pip install --upgrade pip

# 3. 将所有 AI 库安装到一个临时的虚拟环境中
# 这是最关键的一步，它将所有复杂的依赖解析和编译，都隔离在这一级
RUN python -m venv /install/venv && \
    . /install/venv/bin/activate && \
    pip install --no-cache-dir \
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


FROM python:3.10-slim

WORKDIR /app

# 1. 只安装运行所必需的系统库
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# 2. 复制已经安装好的所有 Python 库
COPY --from=builder /install/venv /usr/local

# 3. 复制我们的应用代码
COPY app.py .

# 4. 设置环境变量
ENV HF_TOKEN=$HF_TOKEN

# 5. 暴露端口
EXPOSE 7860

# 6. 启动应用
CMD ["python", "app.py"]