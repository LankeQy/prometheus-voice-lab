# 使用一个标准的 Python 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 1. 更新包管理器并安装基础工具 (如 git) 和 FFmpeg
# FFmpeg 是处理音视频的底层依赖，yt-dlp 需要它
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. 复制依赖文件
COPY requirements.txt .

# 3. 核心策略：分步安装 + 强制清理
# 我们将最庞大的 PyTorch 单独安装，然后清理缓存，再安装其他的。

# -- 第一步：安装 PyTorch 和 TorchAudio --
# 使用 --no-cache-dir 参数告诉 pip 不要保存下载缓存
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    # 清理 pip 的全局缓存 (如果存在)
    rm -rf /root/.cache/pip

# -- 第二步：安装所有剩余的依赖 --
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# 4. 复制所有应用代码
COPY . .

# 5. 设置环境变量 (为 pyannote 准备)
#  Space secrets 中设置了 HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# 6. 暴露 Gradio 的默认端口
EXPOSE 7860

# 7. 启动命令
# 使用 --server-name 0.0.0.0 使其可以被外部访问
CMD ["gradio", "app.py", "--server-name", "0.0.0.0", "--server-port", "7860"]