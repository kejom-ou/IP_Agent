"""
============================================================
本地大语言模型引擎（local_models/llm_engine.py）
============================================================
将 DeepSeek API 调用替换为本地运行的 Qwen2.5 模型。

自动分级策略：
  - 8GB+  → Qwen2.5-3B-Instruct (int4, ~3.5GB)   ⭐⭐⭐⭐
  - 6GB+  → Qwen2.5-1.5B-Instruct (int4, ~1.8GB) ⭐⭐⭐
  - <6GB  → Qwen2.5-0.5B-Instruct (int4, ~0.8GB) ⭐⭐
  - CPU   → Qwen2.5-1.5B-Instruct (GGUF)         ⭐⭐⭐

支持：
  - ollama（最简单，推荐）
  - llama.cpp / GGUF（最省显存）
  - vLLM（高性能服务模式，可选）

接口兼容原 ai_processing/text_rewriter.py 的 execute_rewrite
============================================================
"""

import os
import gc
import json
import logging
import subprocess
from typing import Optional, Literal

import torch

from local_models.config import (
    get_llm_config,
    detect_vram_gb,
    LLM_CONFIG_TIERS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 显存管理
# ---------------------------------------------------------------------------

def _clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 方案 1：Ollama 后端（推荐，最简单）
# ---------------------------------------------------------------------------

class OllamaLLM:
    """
    基于 Ollama 的本地 LLM 推理。
    优点：一键安装、自动量化、API 兼容 OpenAI 格式。
    """

    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name
        self.base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._ready = False

    def check_and_pull(self) -> bool:
        """检查 Ollama 服务是否就绪，必要时拉取模型"""
        import requests

        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                logger.error(f"❌ Ollama 服务异常: {resp.status_code}")
                return False

            models = [m["name"] for m in resp.json().get("models", [])]
            if self.model_name not in models:
                logger.info(f"⬇️  正在拉取模型 {self.model_name}...")
                subprocess.run(
                    ["ollama", "pull", self.model_name],
                    check=True,
                )
                logger.info(f"✅ 模型拉取完成: {self.model_name}")

            self._ready = True
            return True

        except requests.ConnectionError:
            logger.error(
                "❌ Ollama 服务未运行。\n"
                "   安装: curl -fsSL https://ollama.com/install.sh | sh\n"
                "   启动: ollama serve"
            )
            return False
        except Exception as e:
            logger.error(f"❌ Ollama 检查失败: {e}")
            return False

    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """调用 Ollama Chat API"""
        import requests

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            logger.error(f"❌ Ollama 调用失败: {e}")
            return ""


# ---------------------------------------------------------------------------
# 方案 2：llama.cpp / GGUF（最省显存）
# ---------------------------------------------------------------------------

class LlamaCppLLM:
    """
    基于 llama-cpp-python 的 GGUF 量化模型推理。
    优点：显存占用最小，支持 GPU offloading。
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,    # -1 = 全部放到 GPU
        use_mlock: bool = False,    # 低配关闭以节省 RAM
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.use_mlock = use_mlock
        self.model = None
        self._ready = False

    def load(self) -> bool:
        """加载 GGUF 模型"""
        try:
            _clear_vram()
            from llama_cpp import Llama

            logger.info(f"🔄 加载 GGUF 模型: {self.model_path}")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                use_mlock=self.use_mlock,
                verbose=False,
            )
            self._ready = True
            logger.info(f"✅ GGUF 模型加载完成")
            return True
        except ImportError:
            logger.error(
                "❌ llama-cpp-python 未安装。\n"
                "   安装: pip install llama-cpp-python"
            )
            return False
        except Exception as e:
            logger.error(f"❌ GGUF 模型加载失败: {e}")
            return False

    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """调用 GGUF 模型生成"""
        if not self._ready:
            return ""

        # 构建 prompt（ChatML 格式）
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        try:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"❌ GGUF 生成失败: {e}")
            return ""

    def unload(self):
        """释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
        self._ready = False
        _clear_vram()


# ---------------------------------------------------------------------------
# 统一 LLM 引擎
# ---------------------------------------------------------------------------

class LocalLLMEngine:
    """
    本地 LLM 统一入口，自动选择最优后端。

    优先级：
      1. Ollama（如果已安装并运行）
      2. llama.cpp / GGUF（如果提供了模型路径）
      3. 降级到提示用户安装

    使用示例:
        engine = LocalLLMEngine()
        result = engine.rewrite(original_text, mode="auto", prompt=None)
    """

    # 文案仿写系统提示词
    SYSTEM_PROMPT_AUTO = """你是一个专业的口播文案改写助手。请根据以下规则改写文案：
1. 保留原文核心信息和关键卖点
2. 优化语言表达，使文案更口语化、更有感染力
3. 保持原文的段落结构
4. 可以适当调整语序和用词，但不能改变原意
5. 输出只包含改写后的文案，不要添加任何解释说明"""

    SYSTEM_PROMPT_CUSTOM = """你是一个专业的口播文案改写助手。请严格按照用户给出的指令改写文案。
输出只包含改写后的文案，不要添加任何解释说明。"""

    def __init__(
        self,
        backend: Literal["auto", "ollama", "llamacpp"] = "auto",
        ollama_model: str = "qwen2.5:3b",
        gguf_path: Optional[str] = None,
    ):
        self.backend = backend
        self.ollama_model = ollama_model
        self.gguf_path = gguf_path
        self.engine = None
        self._initialized = False

    def init(self) -> bool:
        """初始化 LLM 引擎"""
        # 自动检测
        if self.backend == "auto":
            # 先尝试 Ollama
            self.engine = OllamaLLM(model_name=self.ollama_model)
            if self.engine.check_and_pull():
                self._initialized = True
                logger.info(f"✅ 使用 Ollama 后端: {self.ollama_model}")
                return True

            # 再尝试 GGUF
            if self.gguf_path and os.path.exists(self.gguf_path):
                self.engine = LlamaCppLLM(model_path=self.gguf_path)
                if self.engine.load():
                    self._initialized = True
                    logger.info("✅ 使用 llama.cpp 后端")
                    return True

            logger.error(
                "❌ 无可用 LLM 后端。\n"
                "   推荐安装 Ollama: curl -fsSL https://ollama.com/install.sh | sh\n"
                "   然后拉取模型: ollama pull qwen2.5:3b"
            )
            return False

        elif self.backend == "ollama":
            self.engine = OllamaLLM(model_name=self.ollama_model)
            self._initialized = self.engine.check_and_pull()
            return self._initialized

        elif self.backend == "llamacpp":
            if not self.gguf_path:
                logger.error("❌ 需要提供 GGUF 模型路径")
                return False
            self.engine = LlamaCppLLM(model_path=self.gguf_path)
            self._initialized = self.engine.load()
            return self._initialized

        return False

    def rewrite(
        self,
        original_text: str,
        mode: str = "AI自动仿写",
        custom_prompt: Optional[str] = None,
        api_key: Optional[str] = None,    # 保留参数，本地模式下忽略
    ) -> str:
        """
        文案仿写（与原 execute_rewrite 接口兼容）。

        Args:
            original_text: 原始文案
            mode:          "AI自动仿写" 或 "根据指令仿写"
            custom_prompt: 自定义仿写指令（mode="根据指令仿写" 时使用）
            api_key:       保留参数（兼容旧接口），本地模式忽略

        Returns:
            仿写后的文案
        """
        if not self._initialized:
            if not self.init():
                return original_text    # 失败则返回原文

        # 构建消息
        if mode == "AI自动仿写":
            system_prompt = self.SYSTEM_PROMPT_AUTO
            user_prompt = f"请改写以下文案：\n\n{original_text}"
        else:
            system_prompt = self.SYSTEM_PROMPT_CUSTOM
            instruction = custom_prompt or "请优化这段文案的表达"
            user_prompt = f"指令：{instruction}\n\n原文案：\n{original_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(f"✍️  开始仿写，模式: {mode}")
        result = self.engine.chat(messages, temperature=0.7, max_tokens=2048)
        logger.info(f"✍️  仿写完成，输出 {len(result)} 字符")

        # 仿写完成后释放显存
        self.unload()
        return result if result else original_text

    def generate_description(
        self,
        text: str,
        api_key: Optional[str] = None,
    ) -> str:
        """
        生成视频描述与话题标签（与原 AI_write_descriptions 兼容）。

        Args:
            text:    视频文案
            api_key: 保留参数

        Returns:
            视频描述 + #话题标签
        """
        if not self._initialized:
            if not self.init():
                return ""

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的短视频运营。请根据视频文案，撰写一段吸引人的视频描述（30-80字），并在末尾添加3-5个话题标签（#开头）。直接输出结果。",
            },
            {
                "role": "user",
                "content": f"视频文案：\n{text}",
            },
        ]

        result = self.engine.chat(messages, temperature=0.8, max_tokens=512)
        self.unload()
        return result

    def unload(self):
        """释放显存"""
        if isinstance(self.engine, LlamaCppLLM):
            self.engine.unload()
        elif isinstance(self.engine, OllamaLLM):
            pass   # Ollama 由外部服务管理，不需要手动释放
        self._initialized = False


# ---------------------------------------------------------------------------
# 与原接口兼容的包装函数
# ---------------------------------------------------------------------------

def execute_rewrite_local(
    original_text: str,
    ai_mode: str,
    ai_prompt: Optional[str],
    api_key: Optional[str],
) -> str:
    """
    替代原 ai_processing/text_rewriter.execute_rewrite。
    接口参数完全兼容。

    Args:
        original_text: 原始文案
        ai_mode:       "AI自动仿写" 或 "根据指令仿写"
        ai_prompt:     自定义仿写指令
        api_key:       保留参数（本地模式忽略）

    Returns:
        仿写后的文案
    """
    engine = LocalLLMEngine()
    return engine.rewrite(
        original_text=original_text,
        mode=ai_mode,
        custom_prompt=ai_prompt,
        api_key=api_key,
    )


def ai_write_descriptions_local(
    text: str,
    api_key: Optional[str],
) -> str:
    """
    替代原 ai_processing/text_rewriter.AI_write_descriptions。

    Args:
        text:    视频文案
        api_key: 保留参数

    Returns:
        视频描述 + 话题标签
    """
    engine = LocalLLMEngine()
    return engine.generate_description(text, api_key)


# ---------------------------------------------------------------------------
# GGUF 模型下载辅助
# ---------------------------------------------------------------------------

# 各档位推荐模型及下载地址
RECOMMENDED_GGUF_MODELS = {
    "qwen2.5-3b": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_gb": 2.0,
        "vram": "~3.5 GB (GPU offload)",
    },
    "qwen2.5-1.5b": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_gb": 1.0,
        "vram": "~1.8 GB (GPU offload)",
    },
    "qwen2.5-0.5b": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "size_gb": 0.4,
        "vram": "~0.8 GB (GPU offload)",
    },
}


def download_gguf_model(model_key: str, save_dir: str = "./models") -> Optional[str]:
    """
    下载推荐的 GGUF 模型。

    Args:
        model_key: "qwen2.5-3b" / "qwen2.5-1.5b" / "qwen2.5-0.5b"
        save_dir:  保存目录

    Returns:
        模型文件路径
    """
    import requests

    if model_key not in RECOMMENDED_GGUF_MODELS:
        logger.error(f"❌ 未知模型: {model_key}")
        return None

    info = RECOMMENDED_GGUF_MODELS[model_key]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(info["url"]))

    if os.path.exists(save_path):
        logger.info(f"✅ 模型已存在: {save_path}")
        return save_path

    logger.info(f"⬇️  下载 {model_key} ({info['size_gb']} GB)...")
    resp = requests.get(info["url"], stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  下载进度: {pct:.1f}%", end="")

    print()
    logger.info(f"✅ 下载完成: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# 便捷测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = LocalLLMEngine()
    if engine.init():
        result = engine.rewrite(
            "今天给大家分享一个超好用的护肤小技巧，每天坚持就能看到效果。",
            mode="AI自动仿写",
        )
        print(f"\n仿写结果:\n{result}")
