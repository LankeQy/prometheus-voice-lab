# Dockerfile

# 使用 Python 3.10 的 slim 版本作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 1. 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=false \
    # 显式定义 Hugging Face 模型的缓存位置
    HF_HOME=/app/huggingface_cache

# 2. 以 root 身份安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. 以 root 身份复制并安装 Python 依赖
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 4. 创建普通用户 appuser
RUN groupadd -r appuser --gid=1000 && useradd -r -g appuser --uid=1000 --create-home appuser

# 5. 复制所有应用代码到工作目录
# 此时，/app 目录下的所有文件仍然归 root 所有
COPY . .

# 6.
# 在切换用户之前，将 /app 目录及其所有内容的属主更改为 appuser。
# chown 命令用于更改文件所有者，-R 表示递归地应用到所有子文件和子目录。
RUN chown -R appuser:appuser /app

# 7. 现在，切换到普通用户身份进行后续所有操作
USER appuser

# 8. 以 appuser 身份运行模型下载脚本
# 因为 appuser 现在拥有 /app 目录，所以它有权限在其中创建 huggingface_cache 文件夹
RUN python download_models.py

# 9. 暴露应用端口
EXPOSE 7860

# 10. 定义容器启动命令（也将以 appuser 身份运行）
CMD ["python", "app.py"]