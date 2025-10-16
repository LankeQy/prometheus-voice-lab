# 使用一个标准的 Python 基础镜像
FROM python:3.10-slim

WORKDIR /app

# 1. 更新包管理器并安装基础工具
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

# 2. 复制依赖文件
COPY requirements.txt .
# 3. 安装依赖
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