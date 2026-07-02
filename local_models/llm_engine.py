"""
本地 LLM 引擎 — Transformers 原生推理 Qwen2.5（从本地加载模型）
"""

import gc
import logging
import warnings
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from local_models.config import get_llm_model

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 系统提示词
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_AUTO = """你是一个专业的口播文案改写助手。请根据以下规则改写文案：
1. 保留原文核心信息和关键卖点
2. 优化语言表达，使文案更口语化、更有感染力
3. 保持原文的段落结构
4. 可以适当调整语序和用词，但不能改变原意
5. 输出只包含改写后的文案，不要添加任何解释说明"""

SYSTEM_PROMPT_CUSTOM = """你是一个专业的口播文案改写助手。请严格按照用户给出的指令改写文案。
输出只包含改写后的文案，不要添加任何解释说明。"""


# ---------------------------------------------------------------------------
# Transformers 原生推理引擎（仅本地加载）
# ---------------------------------------------------------------------------

class TransformersLLM:
    """基于 HuggingFace Transformers 的原生推理 — 仅从本地路径加载"""

    def __init__(self, local_path: str = None):
        cfg = get_llm_model()
        self.local_path = local_path or cfg["local_path"]
        self.model_name = cfg["name"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # LLM 优先 GPU
        self.model = None
        self.tokenizer = None

    def load(self) -> bool:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"加载模型: {self.model_name} 从: {self.local_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation="sdpa",
            )
            self.model.eval()

            params_b = sum(p.numel() for p in self.model.parameters()) / 1e9
            logger.info(f"模型加载完成 ({params_b:.1f}B 参数, {self.device})")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def generate(
        self, messages: list, temperature: float = 0.7, max_tokens: int = 2048,
    ) -> str:
        if self.model is None:
            return ""

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            with torch.amp.autocast(self.device, dtype=torch.float16) if self.device == "cuda" else torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# LLM 统一入口
# ---------------------------------------------------------------------------

class LocalLLMEngine:
    """本地 LLM 入口（文案仿写）"""

    def __init__(self, local_path: str = None):
        self.local_path = local_path
        self.engine: Optional[TransformersLLM] = None

    def init(self) -> bool:
        self.engine = TransformersLLM(local_path=self.local_path)
        return self.engine.load()

    def rewrite(
        self,
        original_text: str,
        mode: str = "AI自动仿写",
        custom_prompt: Optional[str] = None,
    ) -> str:
        if self.engine is None and not self.init():
            return original_text

        if mode == "AI自动仿写":
            system_prompt = SYSTEM_PROMPT_AUTO
            user_prompt = f"请改写以下文案：\n\n{original_text}"
        else:
            system_prompt = SYSTEM_PROMPT_CUSTOM
            instruction = custom_prompt or "请优化这段文案的表达"
            user_prompt = f"指令：{instruction}\n\n原文案：\n{original_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = self.engine.generate(messages, temperature=0.7, max_tokens=2048)
        return result if result else original_text

    def unload(self):
        if self.engine:
            self.engine.unload()
