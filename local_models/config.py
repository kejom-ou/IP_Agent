"""
模型配置中心 — 所有模型均从本地 pretrained_models/ 目录加载
"""

import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 本地模型根目录
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
LOCAL_MODELS_DIR = ROOT_DIR / "pretrained_models"

# ---------------------------------------------------------------------------
# 显存检测
# ---------------------------------------------------------------------------

def detect_vram_gb() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
        logger.info(f"GPU 显存: {gb} GB")
        return gb
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# 模型配置（每个模型都指定 local_path）
# ---------------------------------------------------------------------------

# ASR — ModelScope pipeline，本地 SenseVoiceSmall 模型
ASR_CONFIG = {
    "local_path": str(LOCAL_MODELS_DIR / "SenseVoiceSmall"),
}

# LLM — Transformers 原生推理（ModelScope pipeline 不支持 chat template）
LLM_CONFIG = {
    "name": "Qwen2.5-0.5B-Instruct",
    "local_path": str(LOCAL_MODELS_DIR / "Qwen2.5-0.5B-Instruct"),
    "quantization": "int4",  # INT4 量化 ~0.5GB 显存
}

# TTS — ModelScope pipeline，本地 CosyVoice Lite 模型（~1-2GB 显存）
TTS_CONFIG = {
    "local_path": str(LOCAL_MODELS_DIR / "CosyVoice-300M-SFT"),
}

# LipSync — Wav2Lip 轻量口型合成，本地 wav2lip_gan.pth 模型
LIPSYNC_CONFIG = {
    "local_path": str(LOCAL_MODELS_DIR / "Wav2Lip"),
}


# ---------------------------------------------------------------------------
# 自动选择
# ---------------------------------------------------------------------------

def get_llm_model() -> dict:
    _check_local_or_warn(LLM_CONFIG["local_path"], LLM_CONFIG["name"])
    return LLM_CONFIG


def _check_local_or_warn(local_path: str, name: str):
    if not Path(local_path).exists():
        logger.warning(f"⚠️ 本地模型目录不存在: {local_path}")
        logger.warning(f"   请将 {name} 模型放到 {local_path}")


# ---------------------------------------------------------------------------
# 便捷测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vram = detect_vram_gb()
    print(f"显存: {vram} GB")
    print(f"LLM:  {LLM_CONFIG['name']} (固定最小参数量)")
    print(f"模型目录: {LOCAL_MODELS_DIR}")
    if LOCAL_MODELS_DIR.exists():
        for d in LOCAL_MODELS_DIR.iterdir():
            print(f"  {'✅' if d.is_dir() else '  '} {d.name}")
    else:
        print("  (目录不存在，请创建并放入模型)")
