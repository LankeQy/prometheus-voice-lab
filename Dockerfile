# Dockerfile

# 使用 Python 3.10 的 slim 版本作为基础镜像，体积更小
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 1. 设置环境变量，避免交互式提示并优化 Python 输出
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=false

# 2. 安装系统依赖
# 将所有 apt-get 操作合并到一层以减小镜像体积，并最后清理缓存
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. 创建一个非 root 用户来运行应用，这是解决权限问题的关键
# 创建一个名为 appuser 的用户和组
RUN groupadd -r appuser --gid=1000 && useradd -r -g appuser --uid=1000 --create-home appuser

# 4. 复制并安装 Python 依赖
# 先复制 requirements.txt 并安装，可以利用 Docker 的层缓存机制
# 只有当 requirements.txt 变化时，这一层才会重新构建
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 5. 切换到新创建的非 root 用户
USER appuser

# 6. 以 appuser 身份预下载模型
# 此时创建的所有文件和目录的所有者都将是 appuser
COPY --chown=appuser:appuser download_models.py .
RUN python download_models.py

# 7. 复制应用代码
# 使用 --chown 确保复制的文件也属于 appuser
COPY --chown=appuser:appuser . .

# 8. 暴露应用端口
EXPOSE 7860

# 9. 定义容器启动命令
CMD ["python", "app.py"]