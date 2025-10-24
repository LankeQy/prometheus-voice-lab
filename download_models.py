# download_models.py
from speechbrain.inference.classifiers import EncoderClassifier
import os

# 确保缓存目录存在
hf_home = os.environ.get("HF_HOME", "/app/huggingface_cache")
os.makedirs(hf_home, exist_ok=True)


print("开始预下载声纹提取模型...")

try:
    EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb",
    )
    print("✅ 声纹提取模型下载完毕。")
except Exception as e:
    print(f"🔴 声纹提取模型下载失败: {e}")