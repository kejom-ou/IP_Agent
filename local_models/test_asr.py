"""
ASR 单项测试 — 使用本地 faster-whisper 模型
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from local_models.asr_engine import WhisperASR

if __name__ == "__main__":
    audio_file = input("音频文件路径: ").strip()
    if not os.path.exists(audio_file):
        print(f"❌ 文件不存在: {audio_file}")
        sys.exit(1)

    asr = WhisperASR()
    if asr.load():
        text = asr.transcribe(audio_file)
        print(f"\n转写结果:\n{text}")
        asr.unload()
    else:
        print("模型加载失败")
