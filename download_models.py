# download_models.py (最终版)

from speechbrain.inference.classifiers import EncoderClassifier
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

# --- 1. 下载 SpeechT5 官方范例所使用的 x-vect 声纹模型 ---
print("开始预下载 SpeechT5 官方推荐的 x-vect 声纹模型...")
try:
    EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb",
    )
    print("✅ x-vect 声纹提取模型下载完毕。")
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

# --- 3. 下载并缓存官方声纹范例数据集 (我们的 "黄金标准") ---
print("\n开始预下载官方声纹范例数据集...")
try:
    # trust_remote_code=True 是新版 datasets 库需要的
    load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)
    print("✅ 官方声纹范例数据集下载完毕。")
except Exception as e:
    print(f"🔴 官方声纹范例数据集下载失败: {e}")
    # 这是一个非关键性步骤，即使失败，应用也能运行
    pass