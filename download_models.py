# download_models.py
import os
from speechbrain.inference.classifiers import EncoderClassifier
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# 确保缓存目录存在
hf_home = os.environ.get("HF_HOME", "/app/huggingface_cache")
os.makedirs(hf_home, exist_ok=True)

print("开始预下载所有模型...")

try:
    EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb",
    )
    print("✅ 声纹提取模型下载完毕。")
except Exception as e:
    print(f"🔴 声纹提取模型下载失败: {e}")

try:
    SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", language="zh-cn")
    SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    print("✅ TTS 相关模型下载完毕。")
except Exception as e:
    print(f"🔴 TTS 模型下载失败: {e}")

print("✅ 所有模型预下载完成！")