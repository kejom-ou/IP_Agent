"""
本地模型状态检查器 — 验证 pretrained_models/ 下的模型是否就绪

用法:
    python local_models/model_downloader.py          # 检查全部
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "pretrained_models"

# 模型清单: (名称, 子目录, 必需)
MODELS = [
    ("ASR — faster-whisper-small", "faster-whisper-small", True),
    ("LLM — Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct", True),
    ("LLM — Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct", False),
    ("LLM — Qwen2.5-3B-Instruct",   "Qwen2.5-3B-Instruct",   False),
    ("TTS — CosyVoice-300M",        "CosyVoice-300M",         True),
    ("LipSync — MuseTalk",          "MuseTalk",               False),
]


def check_all():
    print(f"\n模型目录: {MODELS_DIR}\n")
    ok = 0
    for name, subdir, required in MODELS:
        path = MODELS_DIR / subdir
        if path.is_dir():
            # 简单判断：目录存在且有文件
            files = list(path.rglob("*"))
            size_gb = sum(f.stat().st_size for f in files if f.is_file()) / (1024 ** 3)
            print(f"  ✅ {name} ({size_gb:.1f} GB)")
            ok += 1
        elif required:
            print(f"  ❌ {name} — 缺失！请放入 {path}")
        else:
            print(f"  ⬜ {name} — 未安装（可选）")
    print(f"\n共 {ok}/{len(MODELS)} 个模型就绪。")


if __name__ == "__main__":
    check_all()
