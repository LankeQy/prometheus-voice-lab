

FROM python:3.10

WORKDIR /app


RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY . .


EXPOSE 7860


CMD ["python", "app.py"]