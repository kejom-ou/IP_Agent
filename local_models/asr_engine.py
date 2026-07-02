"""
本地语音识别引擎（faster-whisper，从本地加载模型）
"""

import gc
import logging
import torch

from local_models.config import ASR_MODEL

logger = logging.getLogger(__name__)


class WhisperASR:
    """faster-whisper 语音识别 — 仅从本地加载，不联网下载"""

    def __init__(
        self,
        model_size: str = None,
        compute_type: str = None,
        language: str = "zh",
    ):
        self.model_size = model_size or ASR_MODEL["model_size"]
        self.compute_type = compute_type or ASR_MODEL["compute_type"]
        self.local_path = ASR_MODEL["local_path"]
        self.device = "cpu"  # ASR 固定在 CPU 运行
        self.language = language
        self.model = None

    def load(self) -> bool:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            from faster_whisper import WhisperModel
            logger.info(f"加载 faster-whisper ({self.model_size}) 从: {self.local_path}")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.local_path,
                local_files_only=True,
                num_workers=1,
                cpu_threads=4,
            )
            logger.info("Whisper 加载完成")
            return True
        except Exception as e:
            logger.error(f"Whisper 加载失败: {e}")
            return False

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            return ""
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            text = " ".join(seg.text.strip() for seg in segments)
            logger.info(f"转写完成: {len(text)} 字符")
            return text
        except Exception as e:
            logger.error(f"转写失败: {e}")
            return ""

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
