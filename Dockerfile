# "战地加固版" Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 1.
# build-essential: 包含了 C/C++ 编译器、make 等所有编译源代码所需的工具。
# libsndfile1: 某些音频库在底层需要的共享库。
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 2. 复制依赖文件
COPY requirements.txt .

# 3. 升级 pip 并执行统一安装
# 在一个工具齐全的环境里，这个命令现在可以成功编译所有需要的库。
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# 4. 复制所有应用代码
COPY . .

# 5. 设置环境变量
ENV HF_TOKEN=$HF_TOKEN

# 6. 暴露端口
EXPOSE 7860

# 7. 回归最可靠的 Python 启动方式
CMD ["python", "app.py"]