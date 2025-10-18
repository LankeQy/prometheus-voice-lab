
FROM python:3.10

WORKDIR /app

# 1.
ENV HF_HOME=/app/huggingface_cache
ENV MPLCONFIGDIR=/app/matplotlib_cache

# 2. 安装
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# 3.
RUN mkdir -p $HF_HOME $MPLCONFIGDIR

# 4.
COPY requirements.txt .

# 5. 升级 pip 并安装所有依赖
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. 复制所有应用代码
COPY . .

# 7. 暴露端口
EXPOSE 7860

# 8.  Python 启动方式
CMD ["python", "app.py"]