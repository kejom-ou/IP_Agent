"""
============================================================
本地语音识别引擎（local_models/asr_engine.py）
============================================================
将云端 Whisper API 替换为本地 faster-whisper。
  - small 模型：~1.5 GB 显存，中文准确率 > 95%
  - tiny 模型：~0.5 GB 显存，适合低配显卡
  - 支持 int8 量化进一步节省显存

原接口兼容：替代 utils/video_processor.py 中的 download_and_extract_text
============================================================
"""

import os
import gc
import logging
from typing import Optional
from pathlib import Path

import torch

from local_models.config import ASR_CONFIG, detect_vram_gb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 显存清理
# ---------------------------------------------------------------------------

def _clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Whisper 引擎
# ---------------------------------------------------------------------------

class WhisperASR:
    """
    本地 Whisper 语音识别引擎。

    使用 faster-whisper（CTranslate2 加速），相比原版 Whisper：
      - 推理速度提升 4 倍
      - 显存占用降低 38%
      - 支持 int8 量化
    """

    def __init__(
        self,
        model_size: str = "small",       # tiny / base / small / medium / large-v3
        device: str = "cuda",
        compute_type: str = "int8",       # int8 / float16 / float32
        language: str = "zh",             # 中文优先
    ):
        self.model_size = model_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.compute_type = compute_type
        self.language = language
        self.model = None
        self._loaded = False

    def load(self) -> bool:
        """加载 Whisper 模型"""
        try:
            _clear_vram()
            from faster_whisper import WhisperModel

            logger.info(
                f"🔄 加载 faster-whisper ({self.model_size}, {self.compute_type})..."
            )

            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                num_workers=1,
                cpu_threads=4,
            )
            self._loaded = True
            logger.info(f"✅ Whisper 模型加载完成 ({self.model_size})")
            return True

        except ImportError:
            logger.error(
                "❌ faster-whisper 未安装。\n"
                "   安装: pip install faster-whisper"
            )
            return False
        except Exception as e:
            logger.error(f"❌ Whisper 加载失败: {e}")
            return False

    def transcribe(self, audio_path: str) -> str:
        """
        转写音频文件为文本。

        Args:
            audio_path: 音频文件路径（支持 mp3/wav/m4a 等）

        Returns:
            转写文本
        """
        if not self._loaded:
            logger.error("Whisper 模型未加载")
            return ""

        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=self.language,
                beam_size=5,
                vad_filter=True,          # 过滤静音段
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                ),
            )

            detected_lang = info.language
            logger.info(
                f"📝 检测语言: {detected_lang} "
                f"(概率: {info.language_probability:.2%})"
            )

            text = " ".join(seg.text.strip() for seg in segments)
            logger.info(f"📝 转写完成: {len(text)} 字符")
            return text

        except Exception as e:
            logger.error(f"❌ 转写失败: {e}")
            return ""

    def transcribe_with_timestamps(self, audio_path: str) -> list:
        """
        转写音频并返回带时间戳的片段（用于字幕生成）。

        Returns:
            [{"start": float, "end": float, "text": str}, ...]
        """
        if not self._loaded:
            return []

        try:
            segments, _ = self.model.transcribe(
                audio_path,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )
            return [
                {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
                for seg in segments
            ]
        except Exception as e:
            logger.error(f"❌ 时间戳转写失败: {e}")
            return []

    def unload(self):
        """释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        _clear_vram()
        logger.info("🗑️  Whisper 已卸载")


# ---------------------------------------------------------------------------
# 与原有接口兼容的提取函数
# ---------------------------------------------------------------------------

def download_and_extract_text_local(video_url: str) -> str:
    """
    替代原 utils/video_processor.download_and_extract_text。
    从视频链接下载视频，然后本地 Whisper 提取文案。

    Args:
        video_url: 视频链接（抖音/小红书/视频号）

    Returns:
        提取的口播文案文本
    """
    import tempfile
    import subprocess

    # 步骤 1：下载视频
    logger.info(f"📥 下载视频: {video_url}")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "input_video.mp4")
        audio_path = os.path.join(tmpdir, "audio.wav")

        # 使用 yt-dlp 下载
        try:
            subprocess.run(
                [
                    "yt-dlp", "-f", "best", "-o", video_path,
                    "--no-playlist", video_url
                ],
                check=True, capture_output=True, timeout=120,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 视频下载失败: {e.stderr.decode() if e.stderr else e}")
            return ""
        except FileNotFoundError:
            logger.error(
                "❌ yt-dlp 未安装。\n"
                "   安装: pip install yt-dlp"
            )
            return ""

        # 步骤 2：提取音频
        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1",
                    audio_path, "-y"
                ],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 音频提取失败: {e}")
            return ""
        except FileNotFoundError:
            logger.error("❌ FFmpeg 未安装")
            return ""

        # 步骤 3：Whisper 转写
        asr = WhisperASR(model_size=ASR_CONFIG.model_id.split("-")[-1])
        if not asr.load():
            return ""
        text = asr.transcribe(audio_path)
        asr.unload()

        return text


# ---------------------------------------------------------------------------
# 便捷测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    asr = WhisperASR()
    if asr.load():
        # 测试：对本地音频文件转写
        test_audio = "test_audio.wav"
        if os.path.exists(test_audio):
            result = asr.transcribe(test_audio)
            print(f"转写结果: {result}")
        else:
            print(f"测试音频不存在: {test_audio}")
        asr.unload()
