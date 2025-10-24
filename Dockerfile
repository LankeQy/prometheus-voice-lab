# 使用更小的 slim 基础镜像
FROM python:3.10-slim

WORKDIR /app

# 1. 设置环境变量
ENV HF_HOME=/app/huggingface_cache
ENV MPLCONFIGDIR=/app/matplotlib_cache
ENV PIP_NO_CACHE_DIR=1
# 设定非交互式前端，避免 apt-get 在构建时卡住
ENV DEBIAN_FRONTEND=noninteractive

# 2. 安装系统依赖并清理
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    espeak-ng \
    --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. 创建所有需要的目录
# 我们一次性创建所有目录，方便后面统一授权
RUN mkdir -p $HF_HOME $MPLCONFIGDIR

# 4. 复制依赖文件
COPY requirements.txt .

# 5. 安装 Python 依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 6. 预下载模型
COPY download_models.py .
RUN python download_models.py

# 7. 复制所有应用代码
COPY . .

# 8. 【关键修复 - 通用方案】授予所有用户对工作目录的写入权限
# 这确保了无论容器以哪个用户身份运行，都能写入所需的文件
RUN chmod -R 777 /app

# 9. 暴露端口
EXPOSE 7860

# 10. 定义启动命令
CMD ["python", "app.py"]