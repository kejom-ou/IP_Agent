"""
本地口型合成引擎（MuseTalk，从本地模型加载）
"""

import gc
import logging
from pathlib import Path
from typing import Optional

import torch

from local_models.config import LIPSYNC_CONFIG

logger = logging.getLogger(__name__)


class MuseTalkEngine:
    """MuseTalk 口型合成 — 仅从本地目录加载，不联网下载（需 ~4GB 显存）"""

    def __init__(self, local_path: str = None):
        self.local_path = local_path or LIPSYNC_CONFIG["local_path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None

    def load(self) -> bool:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            logger.info(f"加载 MuseTalk 从: {self.local_path}")
            self.pipeline = pipeline(
                Tasks.talking_head_lip_sync,
                model=self.local_path,
                device=self.device,
            )
            logger.info("MuseTalk 加载完成")
            return True
        except ImportError:
            logger.error("modelscope 未安装 → pip install modelscope")
            return False
        except Exception as e:
            logger.error(f"MuseTalk 加载失败: {e}")
            return False

    def generate(
        self, video_path: str, audio_path: str,
        output_path: Optional[str] = None, fps: int = 25,
    ) -> Optional[str]:
        if self.pipeline is None:
            return None

        if output_path is None:
            suffix = Path(video_path).suffix or ".mp4"
            output_path = str(
                Path(video_path).parent / f"{Path(video_path).stem}_lipsync{suffix}"
            )

        try:
            logger.info(f"MuseTalk 生成中: {video_path} + {audio_path}")
            self.pipeline(dict(
                video_path=video_path,
                audio_path=audio_path,
                output_video_path=output_path,
                fps=fps,
            ))
            logger.info(f"生成完成: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"MuseTalk 生成失败: {e}")
            return None

    def unload(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
