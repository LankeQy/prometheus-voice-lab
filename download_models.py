# download_models.py
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    AutoProcessor,
    AutoModel
)

# --- 1. 下载官方推荐的、与SpeechT5配套的声纹提取模型 ---
print("开始预下载 SpeechT5 配套的声纹提取模型...")
try:
    embedding_model_id = "speechbrain/speaker-recognition-ecapa-tdnn-voxceleb"
    AutoProcessor.from_pretrained(embedding_model_id)
    AutoModel.from_pretrained(embedding_model_id)
    print("✅ 声纹提取模型下载完毕。")
except Exception as e:
    print(f"🔴 声纹提取模型下载失败: {e}")
    raise e

# --- 2. 下载 Microsoft SpeechT5 语音合成相关模型 ---
print("\n开始预下载 SpeechT5 语音合成模型...")
try:
    tts_model_id = "microsoft/speecht5_tts"
    vocoder_model_id = "microsoft/speecht5_hifigan"
    SpeechT5Processor.from_pretrained(tts_model_id)
    SpeechT5ForTextToSpeech.from_pretrained(tts_model_id)
    SpeechT5HifiGan.from_pretrained(vocoder_model_id)
    print("✅ SpeechT5 处理器、主模型和声码器下载完毕。")
except Exception as e:
    print(f"🔴 SpeechT5 模型下载失败: {e}")
    raise e