# 使用一个标准的 Python 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 1. 更新包管理器并安装基础工具 (如 git) 和 FFmpeg
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. 复制依赖文件
COPY requirements.txt .

# 3. 核心策略：分步安装 + 强制清理
# -- 第一步：安装 PyTorch 和 TorchAudio --
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    rm -rf /root/.cache/pip

# -- 第二步：安装所有剩余的依赖 --
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# 4. 复制所有应用代码
COPY . .

# 5. 设置环境变量 (为 pyannote 准备)
# Space secrets 中设置了 HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# 6. 暴露 Gradio 的默认端口
EXPOSE 7860

# 7. 启动命令
# 使用 --host 参数
CMD ["gradio", "app.py", "--host", "0.0.0.0", "--server-port", "7860"]