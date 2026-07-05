"""
本地 LLM 引擎 — Transformers 原生推理 Qwen2.5（从本地加载模型）
支持 FP16 / INT4，FP16 ~1GB 显存，INT4 ~0.5GB 显存
"""

import gc
import logging
import re
import warnings
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from local_models.config import get_llm_model

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# bitsandbytes INT4 量化（可选，无此包则回退 FP16）
try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    logger.info("bitsandbytes 未安装，LLM 将使用 FP16（~2GB 显存）")

# ---------------------------------------------------------------------------
# 文本清洗：过滤 emoji 和特殊字符（仿写后处理，确保不给 TTS 喂怪字符）
# ---------------------------------------------------------------------------

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U00002B00-\U00002BFF"
    "\U0001F1E6-\U0001F1FF"
    "\U0001F000-\U0001F02F"
    "\U0001F0A0-\U0001F0FF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "\U00002702-\U000027B0"
    "]+",
    flags=re.UNICODE,
)

_SPECIAL_SYMBOLS_PATTERN = re.compile(
    "["
    "★☆◆◇■□●○▲△▼▽→←↑↓⇒⇐"
    "①②③④⑤⑥⑦⑧⑨⑩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽"
    "❶❷❸❹❺❻❼❽❾❿"
    "─━│┃…·•・【】〖〗⟨⟩©®™"
    "✅❌❎🔗🎬🚀📥📹🎭✍️🔊📌"
    "]+",
    flags=re.UNICODE,
)


def clean_text_for_tts(text: str) -> str:
    """去除 emoji 和特殊装饰符号，保留中文/英文/数字和标准标点。"""
    if not text:
        return text
    text = _EMOJI_PATTERN.sub("", text)
    text = _SPECIAL_SYMBOLS_PATTERN.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 系统提示词
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_AUTO = """你是一个专业的短视频口播文案改写师。你的任务是对用户提供的文案进行深度改写，绝对不能直接复制原文。

改写要求：
1. 调整句子语序，把原文的前后句顺序打乱重组
2. 至少替换 60% 的用词（保留关键名词即可），用不同但等价的表达方式
3. 把书面化表达改成口语化短句，适合口播朗读
4. 可以增减语气词（"说真的""你想想""对不对"等）来增强互动感
5. 核心信息和数据必须保留，但表达方式必须完全不同

禁止事项：
- 禁止使用任何 emoji 表情符号（如 😊🎉👍 等）
- 禁止使用特殊 Unicode 字符、装饰符号（如 ★◆→①② 等）
- 只使用简体中文、英文、数字和标准中文标点符号（，。！？；：""''）

输出格式（严格遵守，每行一种内容）：
▼标题
一句吸引人的视频标题（15字以内）
▼标签
#标签1 #标签2 #标签3
▼正文
改写后的完整文案（正文严格控制在200字以内）"""

SYSTEM_PROMPT_CUSTOM = """你是一个专业的口播文案改写助手。请严格按照用户给出的指令改写文案。

输出格式（严格遵守，每行一种内容）：
▼标题
一句吸引人的视频标题（15字以内）
▼标签
#标签1 #标签2 #标签3
▼正文
改写后的完整文案（正文严格控制在200字以内）"""


# ---------------------------------------------------------------------------
# Transformers 原生推理引擎（仅本地加载）
# ---------------------------------------------------------------------------

class TransformersLLM:
    """基于 HuggingFace Transformers 的原生推理 — 仅从本地路径加载，支持 INT4"""

    def __init__(self, local_path: str = None):
        cfg = get_llm_model()
        self.local_path = local_path or cfg["local_path"]
        self.model_name = cfg["name"]
        self.quantization = cfg.get("quantization", "fp16")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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

            # INT4 量化（bitsandbytes）
            if self.device == "cuda" and self.quantization == "int4" and HAS_BNB:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.local_path,
                    quantization_config=quant_config,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                    attn_implementation="sdpa",
                )
                quant_tag = "INT4"
            else:
                # FP16 (GPU) / FP32 (CPU) 回退
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.local_path,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                    attn_implementation="sdpa",
                )
                quant_tag = f"FP16" if self.device == "cuda" else "FP32"

            self.model.eval()

            params_b = sum(p.numel() for p in self.model.parameters()) / 1e9
            vram_mb = torch.cuda.memory_allocated() / (1024**2) if self.device == "cuda" else 0
            logger.info(f"模型加载完成 ({params_b:.2f}B 参数, {quant_tag}, GPU={vram_mb:.0f}MB, {self.device})")
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
                    repetition_penalty=1.15,
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
    ) -> dict:
        """仿写文案，返回 {"text": 正文, "title": 标题, "tags": 标签文本}"""
        if self.engine is None and not self.init():
            return {"text": original_text, "title": "", "tags": ""}

        if mode == "AI自动仿写":
            system_prompt = SYSTEM_PROMPT_AUTO
            user_prompt = f"请对下面的文案进行深度改写，调整语序、替换用词、口语化表达，但保留核心信息。输出必须和原文完全不同：\n\n{original_text}"
        else:
            system_prompt = SYSTEM_PROMPT_CUSTOM
            instruction = custom_prompt or "请优化这段文案的表达"
            user_prompt = f"指令：{instruction}\n\n原文案：\n{original_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = self.engine.generate(messages, temperature=0.9, max_tokens=2048)
        if not result:
            return {"text": original_text, "title": "", "tags": ""}

        # 先解析结构化输出（依赖 ▼ 分隔符），再清洗各字段
        parsed = self._parse_structured_output(result, original_text)
        parsed["text"] = clean_text_for_tts(parsed["text"])
        parsed["title"] = clean_text_for_tts(parsed.get("title", ""))
        parsed["tags"] = clean_text_for_tts(parsed.get("tags", ""))
        return parsed

    def _parse_structured_output(self, raw: str, fallback_text: str) -> dict:
        """从 LLM 输出中解析 标题 / 标签 / 正文（解析失败时自动兜底）"""
        title = ""
        tags = ""
        text = raw  # 默认使用原始输出

        # 尝试按 ▼分隔符 解析
        # 格式: ▼标题\nxxx\n▼标签\n#tag1 #tag2\n▼正文\nxxx
        title_match = re.search(r'▼标题\s*\n\s*(.+?)\s*(?=▼标签|▼正文|$)', raw, re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()

        tags_match = re.search(r'▼标签\s*\n\s*(.+?)\s*(?=▼正文|$)', raw, re.DOTALL)
        if tags_match:
            tags_raw = tags_match.group(1).strip()
            # 提取所有 # 开头的中文/英文标签，合并成单行空格分隔
            tag_list = re.findall(r'#[\w\u4e00-\u9fff]+', tags_raw)
            tags = " ".join(tag_list) if tag_list else tags_raw

        body_match = re.search(r'▼正文\s*\n(.+)', raw, re.DOTALL)
        if body_match:
            text = body_match.group(1).strip()
        else:
            # 兜底：如果解析失败，去掉标题/标签行，剩余作为正文
            text = re.sub(r'▼标题\s*\n.+?(?=▼标签|$)', '', text, flags=re.DOTALL)
            text = re.sub(r'▼标签\s*\n.+?(?=▼正文|$)', '', text, flags=re.DOTALL)
            text = text.strip()

        if not text or len(text) < 5:
            text = fallback_text

        # 兜底：若 LLM 没有按格式输出 ▼标题，则用正文首句作为标题
        if not title and text and text != fallback_text:
            first = re.split(r'[。！？!?\n]', text.strip(), maxsplit=1)[0].strip()
            title = first[:30] if first else ""

        # 兜底：若 LLM 没有输出标签，尝试从正文里抓 #xxx
        if not tags and text:
            tag_list = re.findall(r'#[\w\u4e00-\u9fff]+', text)
            tags = " ".join(tag_list[:5]) if tag_list else ""

        return {"text": text, "title": title, "tags": tags}

    def unload(self):
        if self.engine:
            self.engine.unload()
