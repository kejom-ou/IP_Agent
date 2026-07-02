"""
============================================================
统一适配器（local_models/adapter.py）
============================================================
提供与原 app.py 接口完全兼容的本地模型实现。

在 app.py 中只需修改 import 即可完成切换：
  原: from utils.video_processor import download_and_extract_text
  新: from local_models.adapter import download_and_extract_text

  原: from ai_processing.text_rewriter import execute_rewrite, AI_write_descriptions
  新: from local_models.adapter import execute_rewrite, AI_write_descriptions

  原: from video_tools.generate_video import generate_tuilionnx_video
  新: from local_models.adapter import generate_tuilionnx_video

  原: from utils.voice_processor import handle_audio_creation, get_pt_files, ...
  新: from local_models.adapter import handle_audio_creation, get_pt_files, ...
============================================================
"""

import os
import logging
from typing import Optional, List, Tuple

from local_models.asr_engine import download_and_extract_text_local
from local_models.llm_engine import execute_rewrite_local, ai_write_descriptions_local
from local_models.tts_engine import (
    handle_audio_creation_local,
    get_pt_files_local,
    generate_audio_only_local,
)
from local_models.lipsync import generate_lipsync_video
from local_models.config import (
    detect_vram_gb,
    get_all_configs,
    print_model_plan,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# 文案提取（ASR）
# ===========================================================================

def download_and_extract_text(video_url: str) -> str:
    """
    从视频链接提取口播文案（本地 Whisper 版本）。
    接口与原 utils/video_processor.download_and_extract_text 完全兼容。
    """
    return download_and_extract_text_local(video_url)


# ===========================================================================
# 文案仿写（LLM）
# ===========================================================================

def execute_rewrite(
    original_text: str,
    ai_mode: str,
    ai_prompt: Optional[str],
    api_key: Optional[str],
) -> str:
    """
    文案仿写（本地 Qwen2.5 版本）。
    接口与原 ai_processing/text_rewriter.execute_rewrite 完全兼容。
    """
    return execute_rewrite_local(original_text, ai_mode, ai_prompt, api_key)


def AI_write_descriptions(text: str, api_key: Optional[str]) -> str:
    """
    生成视频描述与话题标签（本地版本）。
    接口与原 ai_processing/text_rewriter.AI_write_descriptions 完全兼容。
    """
    return ai_write_descriptions_local(text, api_key)


# ===========================================================================
# 语音合成（TTS / CosyVoice）
# ===========================================================================

def get_pt_files() -> List[Tuple[str, str]]:
    """获取可用音色列表"""
    return get_pt_files_local()


def handle_audio_creation(
    text: str,
    pt_file_index: int,
    speed: float = 1.0,
) -> Tuple[Optional[str], str]:
    """生成音频"""
    return handle_audio_creation_local(text, pt_file_index, speed)


def generate_audio_only(text: str, api_key: Optional[str] = None) -> Tuple[Optional[str], str]:
    """仅生成音频"""
    return generate_audio_only_local(text, api_key)


def run_GPTvoice_command(account: str = "") -> str:
    """启动语音接口（兼容原接口，本地版无需操作）"""
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
    """
    数字人口播视频生成（本地 MuseTalk/Wav2Lip 版本）。
    接口与原 video_tools/generate_video.generate_tuilionnx_video 完全兼容。

    内部将 TuiliONNX 替换为 MuseTalk/Wav2Lip 本地引擎，
    自动根据显存选择最优方案。
    """
    return generate_lipsync_video(
        face_model=face_model,
        video_path=video_path,
        audio_path=audio_path,
        batch_size=batch_size,
        sync_offset=sync_offset,
        scale_h=scale_h,
        scale_w=scale_w,
        compress=compress,
        beautify_teeth=beautify_teeth,
        silence_check=silence_check,
        add_watermark=add_watermark,
        bg_image=bg_image,
        bg_image_list=bg_image_list,
        check_box=check_box,
    )


# ===========================================================================
# 兼容性辅助函数
# ===========================================================================

def get_trained_models() -> List[str]:
    """获取可用数字人模型（本地版，兼容原接口）"""
    return ["本地数字人（MuseTalk）", "本地数字人（Wav2Lip）"]


def get_face_list() -> List[str]:
    """获取可用人脸模型列表"""
    return ["默认人脸"]


def refresh_face_list(current_face=None):
    """刷新人脸列表"""
    import gradio as gr
    return gr.update(choices=get_face_list(), value="默认人脸")


# ===========================================================================
# 启动检查
# ===========================================================================

def check_local_environment() -> dict:
    """
    检查本地运行环境是否就绪。

    Returns:
        {
            "gpu_available": bool,
            "vram_gb": int,
            "whisper_ready": bool,
            "llm_ready": bool,
            "tts_ready": bool,
            "lipsync_ready": bool,
            "models": dict,       # 各环节模型配置
        }
    """
    import importlib

    vram = detect_vram_gb()
    configs = get_all_configs(vram)

    # 检查各模块是否可导入
    modules_ok = {
        "whisper_ready": importlib.util.find_spec("faster_whisper") is not None,
        "llm_ready": (
            importlib.util.find_spec("llama_cpp") is not None
            or _check_ollama()
        ),
        "tts_ready": importlib.util.find_spec("cosyvoice") is not None or _check_cosyvoice_api(),
        "lipsync_ready": importlib.util.find_spec("modelscope") is not None,
    }

    return {
        "gpu_available": vram > 0,
        "vram_gb": vram,
        **modules_ok,
        "models": {k: cfg.name for k, cfg in configs.items()},
    }


def _check_ollama() -> bool:
    """检查 Ollama 是否可用"""
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except:
        return False


def _check_cosyvoice_api() -> bool:
    """检查 CosyVoice API 是否可用"""
    import requests
    try:
        resp = requests.get("http://localhost:9880/docs", timeout=2)
        return resp.status_code == 200
    except:
        return False


# ===========================================================================
# 便捷入口：一键检查
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print_model_plan()
    status = check_local_environment()
    print(f"\n📊 环境状态:")
    print(f"   GPU: {'✅' if status['gpu_available'] else '❌'} ({status['vram_gb']} GB)")
    print(f"   Whisper (ASR):  {'✅' if status['whisper_ready'] else '❌'}")
    print(f"   LLM (仿写):     {'✅' if status['llm_ready'] else '❌'}")
    print(f"   CosyVoice (TTS):{'✅' if status['tts_ready'] else '❌'}")
    print(f"   LipSync (口型): {'✅' if status['lipsync_ready'] else '❌'}")
    print(f"   模型方案: {status['models']}")
