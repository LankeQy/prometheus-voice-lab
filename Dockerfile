
FROM python:3.10

WORKDIR /app

# 1. 在一个完整的 Debian 系统上，安装我们的“施工工具”
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 2. 复制依赖文件
COPY requirements.txt .

# 3. 升级 pip 并执行统一安装
# 在一个完整的、工具齐全的环境里，pip 将拥有解决所有依赖问题的能力。
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. 复制所有应用代码
COPY . .

# 5. 设置环境变量
ENV HF_TOKEN=$HF_TOKEN

# 6. 暴露端口
EXPOSE 7860

# 7. 回归最可靠的 Python 启动方式
CMD ["python", "app.py"]