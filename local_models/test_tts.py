"""
TTS 单项测试 — 使用本地 CosyVoice 模型（纯本地 Python 推理）
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from local_models.tts_engine import CosyVoiceEngine

if __name__ == "__main__":
    engine = CosyVoiceEngine()
    text = input("合成文本: ").strip()
    if not text:
        print("❌ 文本为空")
        sys.exit(1)

    output_path = os.path.join(os.path.dirname(__file__), "test_output.wav")
    result = engine.synthesize(text=text, speaker="default", speed=1.0, output_path=output_path)
    if result:
        print(f"✅ 合成完成: {result}")
    else:
        print("❌ 合成失败")
    engine.unload()
