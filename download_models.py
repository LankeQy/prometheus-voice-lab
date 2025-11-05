# download_models.py (XTTS 最终版)
from TTS.utils.manage import ModelManager
import os

print("\n开始预下载 Coqui XTTS v2 多语言语音合成模型...")
print("这个模型大约 2GB，下载过程可能需要几分钟...")

try:
    # 指定模型路径
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

    # 创建模型管理器实例
    mm = ModelManager()

    # 下载模型。这个函数会自动处理缓存，如果已下载则跳过。
    mm.download_model(model_name)

    print(f"✅ Coqui XTTS v2 模型已成功下载或验证。")

except Exception as e:
    print(f"🔴 Coqui XTTS v2 模型下载失败: {e}")
    raise e