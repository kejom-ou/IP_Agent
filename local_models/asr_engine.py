"""
本地语音识别引擎（FunASR AutoModel + SenseVoiceSmall，从本地加载）
"""

import gc
import logging
import torch

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from local_models.config import ASR_CONFIG

logger = logging.getLogger(__name__)


class ASREngine:
    """ASR 语音识别 — FunASR AutoModel (SenseVoiceSmall) + 本地模型"""

    def __init__(self):
        self.model_path = ASR_CONFIG["local_path"]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load(self) -> bool:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"加载 SenseVoiceSmall 从: {self.model_path}")
            self.model = AutoModel(
                model=self.model_path,
                trust_remote_code=True,
                device=self.device,
                disable_update=True,  # 强制离线，不从远程拉取
            )
            logger.info("SenseVoiceSmall 加载完成")
            return True
        except ImportError:
            logger.error("funasr 未安装 → pip install funasr")
            return False
        except Exception as e:
            logger.error(f"ASR 加载失败: {e}")
            return False

    def transcribe(self, audio_path: str, language: str = "zh") -> str:
        """转写音频文件，返回纯文本"""
        if self.model is None:
            return ""

        try:
            result = self.model.generate(
                input=audio_path,
                language=language,  # "zh" / "en" / "yue" / "ja" / "ko" / "auto"
                use_itn=True,       # 逆文本正则化（数字/日期等格式化）
                batch_size_s=60,
            )
            if not result:
                return ""

            raw_text = result[0].get("text", "")
            # 去除富文本标记（情感/事件标签），保留纯文本
            text = rich_transcription_postprocess(raw_text)
            logger.info(f"转写完成: {len(text)} 字符")
            return text
        except Exception as e:
            logger.error(f"转写失败: {e}")
            return ""

    def transcribe_with_details(self, audio_path: str, language: str = "zh") -> dict:
        """转写并返回详细信息（文本 + 情感 + 事件）"""
        if self.model is None:
            return {}

        try:
            result = self.model.generate(
                input=audio_path,
                language=language,
                use_itn=True,
                batch_size_s=60,
            )
            if not result:
                return {}

            item = result[0]
            return {
                "text": rich_transcription_postprocess(item.get("text", "")),
                "raw_text": item.get("text", ""),
                "emotion": item.get("emo", ""),       # 情感标签
                "events": item.get("event", ""),       # 音频事件
                "language": item.get("lang", ""),      # 检测到的语言
            }
        except Exception as e:
            logger.error(f"转写失败: {e}")
            return {}

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
