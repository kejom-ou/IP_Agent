"""
统一适配器 — 提供与原 app.py 完全兼容的本地模型接口

用法：在 app.py 中将原 import 替换为此模块即可完成云端 → 本地切换
"""

import logging
from typing import Optional, List, Tuple

from local_models.asr_engine import ASREngine
from local_models.llm_engine import LocalLLMEngine
from local_models.tts_engine import CosyVoiceEngine, get_pt_files_local

logger = logging.getLogger(__name__)


# ===========================================================================
# 文案提取（ASR）
# ===========================================================================

def download_and_extract_text(video_url: str) -> str:
    """从视频链接提取口播文案"""
    import tempfile, subprocess, os

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "input_video.mp4")
        audio_path = os.path.join(tmpdir, "audio.wav")

        try:
            subprocess.run(
                ["yt-dlp", "-f", "best", "-o", video_path, "--no-playlist", video_url],
                check=True, capture_output=True, timeout=120,
            )
        except Exception as e:
            logger.error(f"视频下载失败: {e}")
            return ""

        try:
            subprocess.run([
                "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", audio_path, "-y",
            ], check=True, capture_output=True)
        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            return ""

        asr = ASREngine()
        if not asr.load():
            return ""
        text = asr.transcribe(audio_path)
        asr.unload()
        return text


# ===========================================================================
# 文案仿写（LLM）
# ===========================================================================

def execute_rewrite(
    original_text: str,
    ai_mode: str,
    ai_prompt: Optional[str],
    api_key: Optional[str],
) -> str:
    engine = LocalLLMEngine()
    if not engine.init():
        return original_text
    return engine.rewrite(original_text, mode=ai_mode, custom_prompt=ai_prompt)


def AI_write_descriptions(text: str, api_key: Optional[str]) -> str:
    engine = LocalLLMEngine()
    if not engine.init():
        return ""
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个专业的短视频运营。请根据视频文案，撰写一段吸引人的视频描述"
                "（30-80字），并在末尾添加3-5个话题标签（#开头）。直接输出结果。"
            ),
        },
        {"role": "user", "content": f"视频文案：\n{text}"},
    ]
    return engine.engine.generate(messages, temperature=0.8, max_tokens=512)


# ===========================================================================
# 语音合成（TTS）
# ===========================================================================

def get_pt_files() -> List[Tuple[str, str]]:
    return get_pt_files_local()


def handle_audio_creation(
    text: str, pt_file_index: int, speed: float = 1.0,
) -> Tuple[Optional[str], str]:
    pt_files = get_pt_files_local()
    speaker = pt_files[pt_file_index][1] if 0 <= pt_file_index < len(pt_files) else "default"

    engine = CosyVoiceEngine()
    audio_path = engine.synthesize(text=text, speaker=speaker, speed=speed)
    if audio_path:
        return audio_path, "✅ 语音合成完成"
    return None, "❌ 语音合成失败"


def generate_audio_only(text: str, api_key: Optional[str] = None) -> Tuple[Optional[str], str]:
    return handle_audio_creation(text, 0, 1.0)


def run_GPTvoice_command(account: str = "") -> str:
    return "✅ 本地语音引擎已就绪"


# ===========================================================================
# 口型合成（LipSync）
# ===========================================================================

def generate_tuilionnx_video(
    face_model: str,
    video_path,
    audio_path,
    batch_size: int = 4,
    sync_offset: int = 0,
    scale_h: float = 1.6,
    scale_w: float = 3.6,
    compress: bool = False,
    beautify_teeth: bool = False,
    silence_check: bool = False,
    add_watermark: bool = True,
    bg_image=None,
    bg_image_list=None,
    check_box: bool = False,
) -> Tuple[Optional[str], str, Optional[str], Optional[str]]:
    """兼容原 TuiliONNX 接口"""
    import time, os

    from local_models.lipsync import MuseTalkEngine

    start = time.time()
    engine = MuseTalkEngine()
    if not engine.load():
        return None, "引擎加载失败", None, None

    output_path = os.path.join(
        os.path.dirname(str(audio_path)),
        f"lipsync_output_{int(start)}.mp4",
    )
    result = engine.generate(video_path=video_path, audio_path=audio_path, output_path=output_path)
    engine.unload()

    elapsed = time.time() - start
    time_str = f"生成耗时: {elapsed:.1f}s"
    return (result, time_str, result, None) if result else (None, time_str, None, None)
