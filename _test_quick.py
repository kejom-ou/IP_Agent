"""Quick LLM test"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_models.llm_engine import TransformersLLM

print("Creating TransformersLLM...")
llm = TransformersLLM()
print("Loading model...")
if llm.load():
    print("Generating...")
    msgs = [
        {"role": "system", "content": "改写以下句子，保持意思不变"},
        {"role": "user", "content": "改写：你好世界"},
    ]
    r = llm.generate(msgs, max_tokens=32)
    print(f"OK: {r}")
    llm.unload()
else:
    print("LOAD FAILED")
print("DONE")
