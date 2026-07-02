"""
本地语音合成引擎（ModelScope pipeline，从本地加载模型）
"""

import gc
import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import soundfile as sf

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from local_models.config import TTS_CONFIG

logger = logging.getLogger(__name__)


class CosyVoiceEngine:
    """CosyVoice TTS 引擎 — ModelScope pipeline + 本地模型"""

    def __init__(self, local_path: str = None):
        self.model_path = local_path or TTS_CONFIG["local_path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None

    # ---- 模型加载 ----

    def load_model(self) -> bool:
        """从本地目录加载 CosyVoice 模型（GPU）"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                logger.warning("⚠️ 未检测到 CUDA，CosyVoice 将在 CPU 上运行（较慢）")

            logger.info(f"加载 CosyVoice pipeline 从: {self.model_path}")
            self.pipeline = pipeline(
                task=Tasks.text_to_speech,
                model=self.model_path,
                device=self.device,
            )
            logger.info(f"CosyVoice pipeline 加载完成 (GPU={torch.cuda.is_available()})")
            return True
        except ImportError:
            logger.error("modelscope 未安装 → pip install modelscope")
            return False
        except Exception as e:
            logger.error(f"CosyVoice 加载失败: {e}")
            return False

    # ---- 推理 ----

    def synthesize(
        self, text: str, speaker: str = "default", speed: float = 1.0,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """从本地模型推理合成语音"""
        if self.pipeline is None and not self.load_model():
            return None

        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")

        try:
            result = self.pipeline(input=text, spk_id=speaker, speed=speed)
            # ModelScope CosyVoice pipeline 返回 {'output_wav': numpy.ndarray} 或文件路径
            wav = result.get("output_wav")
            if wav is None:
                wav = result  # fallback: 可能直接返回 numpy array
            if isinstance(wav, str):
                # 返回的是文件路径，直接复制
                import shutil
                shutil.copy(wav, output_path)
            else:
                sf.write(output_path, np.array(wav), 22050)
            logger.info(f"语音合成完成: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"合成失败: {e}")
            return None

    # ---- 卸载 ----

    def unload(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 兼容原接口
# ---------------------------------------------------------------------------

def get_pt_files_local() -> List[Tuple[str, str]]:
    """获取可用音色列表"""
    return [
        ("默认女声", "中文女"),
        ("默认男声", "中文男"),
    ]


def handle_audio_creation_local(
    text: str, speaker_index: int, speed: float = 1.0,
) -> Tuple[Optional[str], str]:
    if not text or not text.strip():
        return None, "文案为空"

    pt_files = get_pt_files_local()
    speaker = pt_files[speaker_index][1] if 0 <= speaker_index < len(pt_files) else "中文女"

    engine = CosyVoiceEngine()
    audio_path = engine.synthesize(text=text, speaker=speaker, speed=speed)
    if audio_path:
        return audio_path, "✅ 语音合成完成"
    return None, "❌ 语音合成失败"


def generate_audio_only_local(text: str, api_key: Optional[str] = None) -> Tuple[Optional[str], str]:
    return handle_audio_creation_local(text, 0, 1.0)
