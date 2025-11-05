# download_models.py (最终手动下载版)
from huggingface_hub import snapshot_download
import os

print("\n开始预下载 Coqui XTTS v2 多语言语音合成模型...")
print("这个模型大约 2GB，下载过程可能需要几分钟...")

try:
    # 模型的 Hugging Face Hub 仓库 ID
    repo_id = "coqui/XTTS-v2"
    # 我们希望将模型文件下载到本地的哪个目录
    local_dir = "./XTTS-v2-main/"

    # 使用 snapshot_download 下载整个仓库
    # ignore_patterns 避免下载不需要的大文件
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # 在 Spaces 环境中必须为 False
        ignore_patterns=["*.png", "*.jpg", "*.jpeg", "*.md", "samples/*"]
    )

    print(f"✅ Coqui XTTS v2 模型文件已成功下载至: {local_dir}")

except Exception as e:
    print(f"🔴 Coqui XTTS v2 模型下载失败: {e}")
    raise e