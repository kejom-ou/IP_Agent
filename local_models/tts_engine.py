"""
============================================================
本地语音合成引擎（local_models/tts_engine.py）
============================================================
封装 CosyVoice 本地 TTS + 声音克隆。

CosyVoice-300M 特性：
  - 模型仅 ~300MB，显存占用 ~2 GB
  - 支持零样本声音克隆（3 秒音频即可）
  - 支持中文/英文/日语/粤语/韩语
  - 情感控制（Instruct 版本）
  - 流式合成，低延迟

接口兼容原 utils/voice_processor 的函数：
  - handle_audio_creation
  - run_GPTvoice_command
  - get_pt_files（音色列表）
============================================================
"""

import os
import gc
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple

import torch

from local_models.config import TTS_CONFIG, detect_vram_gb

logger = logging.getLogger(__name__)


def _clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CosyVoice 引擎
# ---------------------------------------------------------------------------

class CosyVoiceEngine:
    """
    CosyVoice-300M 本地语音合成引擎。

    支持两种模式：
      1. API 模式：使用已有的 CosyVoice 服务（localhost:9880）
      2. 直接模式：通过 Python SDK 直接调用（需安装 cosyvoice 包）
    """

    def __init__(
        self,
        mode: str = "api",             # "api" / "direct"
        api_url: str = "http://localhost:9880",
        device: str = "cuda",
    ):
        self.mode = mode
        self.api_url = api_url
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self._initialized = False

    # ---- API 模式（使用已有 CosyVoice 服务） ----

    def check_api(self) -> bool:
        """检查 CosyVoice API 是否可用"""
        import requests
        try:
            resp = requests.get(f"{self.api_url}/docs", timeout=3)
            return resp.status_code == 200
        except:
            return False

    def get_speakers_api(self) -> List[str]:
        """获取可用音色列表（API 模式）"""
        import requests
        try:
            resp = requests.get(f"{self.api_url}/speakers_list", timeout=10)
            if resp.status_code == 200:
                return resp.json()
            return []
        except:
            return []

    def synthesize_api(
        self,
        text: str,
        speaker: str = "default",
        speed: float = 1.0,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        调用 CosyVoice API 合成语音。

        Args:
            text:       待合成文本
            speaker:    音色名称
            speed:      语速（0.5-2.0）
            output_path: 输出路径（None 则自动生成临时文件）

        Returns:
            音频文件路径
        """
        import requests

        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")

        try:
            resp = requests.post(
                f"{self.api_url}/tts",
                json={
                    "text": text,
                    "speaker": speaker,
                    "speed": speed,
                },
                timeout=60,
            )
            resp.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(resp.content)

            logger.info(f"✅ 语音合成完成: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"❌ CosyVoice API 合成失败: {e}")
            return None

    # ---- 直接模式（Python SDK） ----

    def load_model(self) -> bool:
        """
        直接加载 CosyVoice 模型（不依赖外部 API）。

        需要安装: pip install cosyvoice
        """
        try:
            _clear_vram()
            from cosyvoice.cli.cosyvoice import CosyVoice as CosyVoiceModel

            logger.info("🔄 加载 CosyVoice-300M 模型...")

            self.model = CosyVoiceModel(
                model_dir="pretrained_models/CosyVoice-300M-SFT",
                device=self.device,
            )
            self._initialized = True
            logger.info(f"✅ CosyVoice 模型加载完成")
            return True

        except ImportError:
            logger.warning(
                "⚠️ cosyvoice 包未安装，将使用 API 模式。\n"
                "   直接模式安装: pip install cosyvoice\n"
                "   或从源码: git clone https://github.com/FunAudioLLM/CosyVoice.git"
            )
            return False
        except Exception as e:
            logger.error(f"❌ CosyVoice 加载失败: {e}")
            return False

    def synthesize_direct(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        speed: float = 1.0,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        直接使用 CosyVoice SDK 合成（零样本声音克隆）。

        Args:
            text:           待合成文本
            ref_audio_path: 参考音频（声音克隆），None 则使用默认音色
            ref_text:       参考音频对应的文本（可选）
            speed:          语速
            output_path:    输出路径

        Returns:
            音频文件路径
        """
        if not self._initialized:
            return None

        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")

        try:
            if ref_audio_path:
                # 零样本声音克隆
                self.model.inference_zero_shot(
                    tts_text=text,
                    prompt_text=ref_text or "",
                    prompt_wav=ref_audio_path,
                    output_dir=os.path.dirname(output_path),
                    speed=speed,
                )
            else:
                # 默认音色
                self.model.inference_sft(
                    tts_text=text,
                    output_dir=os.path.dirname(output_path),
                    speed=speed,
                )

            logger.info(f"✅ 语音合成完成: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ CosyVoice 合成失败: {e}")
            return None

    def unload(self):
        """卸载模型"""
        if self.model is not None:
            del self.model
            self.model = None
        self._initialized = False
        _clear_vram()


# ---------------------------------------------------------------------------
# 与原接口兼容的包装函数
# ---------------------------------------------------------------------------

def get_pt_files_local() -> List[Tuple[str, str]]:
    """
    替代原 utils/voice_processor.get_pt_files。
    获取可用音色列表。

    Returns:
        [(display_name, internal_id), ...]
    """
    engine = CosyVoiceEngine(mode="api")

    if engine.check_api():
        speakers = engine.get_speakers_api()
        return [(name, name) for name in speakers]
    else:
        # API 不可用时的默认列表
        logger.warning("⚠️ CosyVoice API 不可用，使用默认音色列表")
        return [
            ("默认女声", "default_female"),
            ("默认男声", "default_male"),
            ("温柔女声", "gentle_female"),
            ("磁性男声", "magnetic_male"),
        ]


def handle_audio_creation_local(
    text: str,
    speaker_index: int,    # Gradio dropdown index
    speed: float = 1.0,
) -> Tuple[Optional[str], str]:
    """
    替代原 utils/voice_processor.handle_audio_creation。

    Args:
        text:          待合成文本
        speaker_index: 音色索引（来自 pt_file_dropdown）
        speed:         语速

    Returns:
        (audio_path, status_message)
    """
    if not text or not text.strip():
        return None, "❌ 文案为空，无法生成音频"

    engine = CosyVoiceEngine(mode="api")

    # 获取音色名称
    pt_files = get_pt_files_local()
    if 0 <= speaker_index < len(pt_files):
        speaker = pt_files[speaker_index][1]
    else:
        speaker = "default"

    logger.info(f"🎙️  语音合成: 音色={speaker}, 语速={speed}")
    audio_path = engine.synthesize_api(
        text=text,
        speaker=speaker,
        speed=speed,
    )

    if audio_path:
        return audio_path, f"✅ 语音合成完成（音色: {speaker}）"
    else:
        return None, "❌ 语音合成失败"


def generate_audio_only_local(text: str, api_key: Optional[str] = None) -> Tuple[Optional[str], str]:
    """
    替代原 utils/voice_processor.generate_audio_only。
    """
    return handle_audio_creation_local(text, 0, 1.0)


# ---------------------------------------------------------------------------
# CosyVoice 服务启动辅助
# ---------------------------------------------------------------------------

def start_cosyvoice_service_local() -> bool:
    """
    启动本地 CosyVoice API 服务。
    复用原 start_cosyvoice_service.py 的逻辑，增加自动下载模型的功能。
    """
    import subprocess
    import sys

    cosyvoice_dir = os.path.join(os.path.dirname(__file__), "..", "cosyvoice")

    # 检查模型是否存在
    model_dir = os.path.join(os.path.dirname(__file__), "..", "pretrained_models", "CosyVoice-300M-SFT")
    if not os.path.exists(model_dir):
        logger.warning(
            "⚠️ CosyVoice 模型未下载。\n"
            "   自动下载方式:\n"
            "   1. pip install modelscope\n"
            "   2. python -c \"from modelscope import snapshot_download; "
            "snapshot_download('iic/CosyVoice-300M-SFT', cache_dir='pretrained_models')\""
        )
        return False

    # 调用原有的启动脚本
    try:
        subprocess.Popen(
            [sys.executable, os.path.join(cosyvoice_dir, "start_cosyvoice_api.py")],
            cwd=cosyvoice_dir,
        )
        logger.info("✅ CosyVoice 服务启动中...")
        return True
    except Exception as e:
        logger.error(f"❌ 启动失败: {e}")
        return False


# ---------------------------------------------------------------------------
# 模型下载辅助
# ---------------------------------------------------------------------------

def download_cosyvoice_model(save_dir: str = "./pretrained_models") -> bool:
    """
    从 ModelScope 下载 CosyVoice-300M-SFT 模型。
    """
    try:
        from modelscope import snapshot_download

        logger.info("⬇️  下载 CosyVoice-300M-SFT 模型...")
        snapshot_download(
            "iic/CosyVoice-300M-SFT",
            cache_dir=save_dir,
        )
        logger.info("✅ CosyVoice 模型下载完成")
        return True
    except ImportError:
        logger.error(
            "❌ modelscope 未安装。\n"
            "   安装: pip install modelscope"
        )
        return False
    except Exception as e:
        logger.error(f"❌ 模型下载失败: {e}")
        return False


# ---------------------------------------------------------------------------
# 便捷测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试 API 模式
    engine = CosyVoiceEngine(mode="api")
    if engine.check_api():
        print("✅ CosyVoice API 可用")
        speakers = engine.get_speakers_api()
        print(f"   可用音色: {speakers}")
    else:
        print("⚠️  CosyVoice API 不可用，请先启动服务")
