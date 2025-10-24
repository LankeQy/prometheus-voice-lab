FROM python:3.10

WORKDIR /app

# 1. 设置环境变量，将缓存目录放在工作区内
ENV HF_HOME=/app/huggingface_cache
ENV MPLCONFIGDIR=/app/matplotlib_cache

# 2. 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# 3. 创建缓存目录
RUN mkdir -p $HF_HOME $MPLCONFIGDIR

# 4. 复制依赖文件
COPY requirements.txt .

# 5. 升级 pip 并安装所有 Python 依赖
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. 复制所有应用代码
COPY . .

# 7. 暴露 Gradio 运行端口
EXPOSE 7860

# 8. 定义容器启动命令
CMD ["python", "app.py"]