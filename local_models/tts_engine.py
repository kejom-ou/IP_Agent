"""
本地语音合成引擎（CosyVoice，纯本地推理）
"""

import gc
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple

import torch

from local_models.config import TTS_CONFIG

logger = logging.getLogger(__name__)


class CosyVoiceEngine:
    """CosyVoice TTS 引擎 — 纯本地 SDK 推理"""

    def __init__(self, local_path: str = None):
        self.local_path = local_path or TTS_CONFIG["local_path"]
        self.model = None

    # ---- 本地模型加载 ----

    def load_sdk(self) -> bool:
        """从本地目录加载 CosyVoice 模型（GPU）"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                logger.warning("⚠️ 未检测到 CUDA，CosyVoice 将在 CPU 上运行（较慢）")

            import sys
            cosy_path = str(Path(self.local_path).parent)
            if cosy_path not in sys.path:
                sys.path.insert(0, cosy_path)

            from cosyvoice.cli.cosyvoice import CosyVoice
            logger.info(f"加载 CosyVoice 从: {self.local_path}")
            self.model = CosyVoice(self.local_path, load_jit=False, load_trt=False, fp16=torch.cuda.is_available())
            logger.info(f"CosyVoice 加载完成 (GPU={torch.cuda.is_available()})")
            return True
        except ImportError:
            logger.error("CosyVoice SDK 未安装 → 请安装 CosyVoice")
            return False
        except Exception as e:
            logger.error(f"CosyVoice 加载失败: {e}")
            return False

    # ---- 本地推理 ----

    def synthesize(
        self, text: str, speaker: str = "default", speed: float = 1.0,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """从本地模型推理合成语音"""
        if self.model is None and not self.load_sdk():
            return None

        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")

        try:
            output = self.model.inference_sft(
                tts_text=text,
                spk_id=speaker,
                stream=False,
                speed=speed,
            )
            import soundfile as sf
            import numpy as np
            sf.write(output_path, np.array(output), 22050)
            logger.info(f"语音合成完成: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"本地推理合成失败: {e}")
            return None

    # ---- 卸载 ----

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 兼容原接口
# ---------------------------------------------------------------------------

def get_pt_files_local() -> List[Tuple[str, str]]:
    """获取可用音色列表（本地音色）"""
    return [
        ("默认女声", "default_female"),
        ("默认男声", "default_male"),
    ]


def handle_audio_creation_local(
    text: str, speaker_index: int, speed: float = 1.0,
) -> Tuple[Optional[str], str]:
    if not text or not text.strip():
        return None, "文案为空"

    pt_files = get_pt_files_local()
    speaker = pt_files[speaker_index][1] if 0 <= speaker_index < len(pt_files) else "default"

    engine = CosyVoiceEngine()
    audio_path = engine.synthesize(text=text, speaker=speaker, speed=speed)
    if audio_path:
        return audio_path, "✅ 语音合成完成"
    return None, "❌ 语音合成失败"


def generate_audio_only_local(text: str, api_key: Optional[str] = None) -> Tuple[Optional[str], str]:
    return handle_audio_creation_local(text, 0, 1.0)
