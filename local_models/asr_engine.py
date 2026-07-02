"""
本地语音识别引擎（ModelScope pipeline，从本地加载模型）
"""

import gc
import logging
import torch

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from local_models.config import ASR_CONFIG

logger = logging.getLogger(__name__)


class WhisperASR:
    """ASR 语音识别 — ModelScope pipeline + 本地模型"""

    def __init__(
        self,
        language: str = "zh",
    ):
        self.model_path = ASR_CONFIG["local_path"]
        self.device = "cpu"  # ASR 固定在 CPU
        self.language = language
        self.pipeline = None

    def load(self) -> bool:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"加载 ASR pipeline 从: {self.model_path}")
            self.pipeline = pipeline(
                task=Tasks.auto_speech_recognition,
                model=self.model_path,
                device=self.device,
            )
            logger.info("ASR pipeline 加载完成")
            return True
        except ImportError:
            logger.error("modelscope 未安装 → pip install modelscope")
            return False
        except Exception as e:
            logger.error(f"ASR 加载失败: {e}")
            return False

    def transcribe(self, audio_path: str) -> str:
        if self.pipeline is None:
            return ""
        try:
            result = self.pipeline(audio_path)
            text = result.get("text", "")
            logger.info(f"转写完成: {len(text)} 字符")
            return text
        except Exception as e:
            logger.error(f"转写失败: {e}")
            return ""

    def unload(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
