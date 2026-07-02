"""
LipSync 单项测试 — 使用本地 MuseTalk 模型
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from local_models.lipsync import MuseTalkEngine

if __name__ == "__main__":
    video_path = input("口播视频路径: ").strip()
    audio_path = input("音频路径: ").strip()

    if not os.path.exists(video_path):
        print(f"❌ 视频不存在: {video_path}")
        sys.exit(1)
    if not os.path.exists(audio_path):
        print(f"❌ 音频不存在: {audio_path}")
        sys.exit(1)

    engine = MuseTalkEngine()
    print("加载本地 MuseTalk 模型...")
    if not engine.load():
        print("❌ 加载失败")
        sys.exit(1)

    output = engine.generate(video_path=video_path, audio_path=audio_path)
    engine.unload()
    if output:
        print(f"✅ 生成完成: {output}")
    else:
        print("❌ 生成失败")
