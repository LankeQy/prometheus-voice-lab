# 使用更小的 slim 基础镜像
FROM python:3.10-slim

WORKDIR /app

# 1. 设置环境变量，将缓存目录放在工作区内
ENV HF_HOME=/app/huggingface_cache
ENV MPLCONFIGDIR=/app/matplotlib_cache
# 确保 pip 不使用缓存，减少构建过程中的磁盘占用
ENV PIP_NO_CACHE_DIR=1

# 2. 安装系统依赖，并在同一层清理缓存，减小镜像大小
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    espeak-ng \
    --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. 创建缓存目录
RUN mkdir -p $HF_HOME $MPLCONFIGDIR

# 4. 复制依赖文件
COPY requirements.txt .

# 5. 升级 pip 并安装所有 Python 依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 6. 复制下载脚本并执行，将模型烘焙进镜像
# 这是解决存储问题的关键步骤
COPY download_models.py .
RUN python download_models.py

# 7. 复制所有应用代码
COPY . .

# 8. 暴露 Gradio 运行端口
EXPOSE 7860

# 9. 定义容器启动命令
CMD ["python", "app.py"]