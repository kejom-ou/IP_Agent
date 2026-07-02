"""
LLM 单项测试 — 使用本地 Qwen2.5 模型
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from local_models.llm_engine import LocalLLMEngine

if __name__ == "__main__":
    engine = LocalLLMEngine()
    print("加载本地 LLM 模型...")
    if not engine.init():
        print("❌ 加载失败")
        sys.exit(1)

    print("\n模型就绪，输入文案进行仿写（输入 q 退出）")
    while True:
        text = input("\n文案: ").strip()
        if text.lower() == "q":
            break
        result = engine.rewrite(text, mode="AI自动仿写")
        print(f"仿写结果:\n{result}")

    engine.unload()
