"""
============================================================
本地模型配置中心（local_models/config.py）
============================================================
作用：集中管理所有本地模型的配置、显存预算分配、自动降级策略。
适配显卡：RTX 5060 8GB / 3060 6GB / 3050 4GB 等消费级显卡。
============================================================
"""

import os
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 自动检测 GPU 显存
# ---------------------------------------------------------------------------

def detect_vram_gb() -> int:
    """
    自动检测当前 CUDA 设备的可用显存（GB），保守取整。
    若未检测到 GPU，返回 0（纯 CPU 模式）。
    """
    if not torch.cuda.is_available():
        logger.warning("⚠️ 未检测到 CUDA GPU，将使用纯 CPU 模式。")
        return 0
    try:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        gb = round(total_bytes / (1024 ** 3))
        logger.info(f"🖥️  检测到 GPU 显存：{gb} GB")
        return gb
    except Exception as e:
        logger.warning(f"⚠️ 显存检测失败：{e}")
        return 0


# ---------------------------------------------------------------------------
# 显存预算规划
# ---------------------------------------------------------------------------
# 设计原则：8GB 显存卡需同时容纳 4 个模型（ASR + LLM + TTS + LipSync），
# 采用"分时加载"策略：不是所有模型同时驻留在显存中，而是用时加载、用完释放。
#
# 显存预算分配（以 8GB 为例）：
# ┌──────────┬──────────┬─────────────────────────────┐
# │ 模型      │ 预估显存  │ 说明                         │
# ├──────────┼──────────┼─────────────────────────────┤
# │ Whisper   │ ~1.5 GB  │ faster-whisper small (int8)  │
# │ LLM 仿写  │ ~3.5 GB  │ Qwen2.5-3B-Instruct (int4)   │
# │ CosyVoice │ ~2.0 GB  │ CosyVoice-300M (fp16)        │
# │ 口型合成  │ ~3.0 GB  │ MuseTalk / Wav2Lip (fp16)    │
# │ 系统预留  │ ~1.0 GB  │ PyTorch 开销 + OS             │
# └──────────┴──────────┴─────────────────────────────┘
#  合计       ≈ 11 GB（分时加载下峰值 ~4 GB）
#
# 关键：ASR → LLM → TTS → LipSync 串行执行，后一个模型加载前释放前一个。

@dataclass
class ModelConfig:
    """单个模型的配置"""
    name: str
    model_id: str                    # HuggingFace / ModelScope 模型 ID
    vram_estimate_gb: float          # 预估显存占用（GB）
    priority: int = 0                # 优先级（数字越小越优先）
    quantize: Optional[str] = None   # 量化方式：int4 / int8 / fp16 / None
    fallback: Optional[str] = None   # 降级模型 ID


@dataclass
class VRAMBudget:
    """显存预算管理器"""
    total_gb: int
    reserved_gb: float = 1.0         # 系统预留（PyTorch 开销 + OS）
    models: list = field(default_factory=list)

    @property
    def available_gb(self) -> float:
        return max(0.0, self.total_gb - self.reserved_gb)

    def fits(self, model: ModelConfig) -> bool:
        return model.vram_estimate_gb <= self.available_gb

    def recommend_quantize(self, model: ModelConfig) -> Optional[str]:
        """根据可用显存推荐量化等级"""
        avail = self.available_gb
        if model.vram_estimate_gb <= avail:
            return None             # fp16 原生精度
        if model.vram_estimate_gb * 0.5 <= avail:
            return "int8"
        if model.vram_estimate_gb * 0.25 <= avail:
            return "int4"
        return "cpu"                # 回退到 CPU


# ---------------------------------------------------------------------------
# 各环节模型配置表
# ---------------------------------------------------------------------------

# 环节 1：语音识别（ASR / Whisper）
ASR_CONFIG = ModelConfig(
    name="Whisper-ASR",
    model_id="Systran/faster-whisper-small",
    vram_estimate_gb=1.5,
    priority=1,
    quantize="int8",
    fallback="Systran/faster-whisper-tiny",
)

# 环节 2：文案仿写（LLM）
# 根据显存量自动选择模型尺寸：
#  >= 8GB → Qwen2.5-3B-Instruct (int4, ~3.5GB)
#  >= 6GB → Qwen2.5-1.5B-Instruct (int4, ~1.8GB)
#  < 6GB  → Qwen2.5-0.5B-Instruct (int4, ~0.8GB) 或 CPU 模式
LLM_CONFIG_TIERS = {
    "high": ModelConfig(
        name="Qwen2.5-3B-Instruct",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        vram_estimate_gb=3.5,
        priority=2,
        quantize="int4",
    ),
    "medium": ModelConfig(
        name="Qwen2.5-1.5B-Instruct",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        vram_estimate_gb=1.8,
        priority=2,
        quantize="int4",
    ),
    "low": ModelConfig(
        name="Qwen2.5-0.5B-Instruct",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        vram_estimate_gb=0.8,
        priority=2,
        quantize="int4",
    ),
    "cpu": ModelConfig(
        name="Qwen2.5-1.5B-Instruct-CPU",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        vram_estimate_gb=0,
        priority=2,
        quantize="cpu",
    ),
}

# 环节 3：语音合成（TTS / CosyVoice）
TTS_CONFIG = ModelConfig(
    name="CosyVoice-300M",
    model_id="FunAudioLLM/CosyVoice-300M",
    vram_estimate_gb=2.0,
    priority=3,
    quantize=None,   # CosyVoice-300M 本身已很轻量
)

# 环节 4：口型合成（LipSync）
# ┌───────────────┬──────────┬──────────┬─────────────┐
# │ 方案           │ 显存占用  │ 画质     │ 适用场景      │
# ├───────────────┼──────────┼──────────┼─────────────┤
# │ MuseTalk       │ ~4 GB    │ ⭐⭐⭐⭐⭐ │ 实时/高画质   │
# │ Wav2Lip        │ ~2.5 GB  │ ⭐⭐⭐⭐  │ 兼容性最好    │
# │ Wav2Lip (low)  │ ~1.5 GB  │ ⭐⭐⭐   │ 最低配置兜底  │
# └───────────────┴──────────┴──────────┴─────────────┘

LIPSYNC_CONFIG_TIERS = {
    "high": ModelConfig(
        name="MuseTalk",
        model_id="TMElyralab/MuseTalk",
        vram_estimate_gb=4.0,
        priority=4,
        quantize="fp16",
    ),
    "medium": ModelConfig(
        name="Wav2Lip",
        model_id="hunyuan/Wav2Lip",    # 社区维护版
        vram_estimate_gb=2.5,
        priority=4,
        quantize="fp16",
    ),
    "low": ModelConfig(
        name="Wav2Lip-Lite",
        model_id="hunyuan/Wav2Lip",
        vram_estimate_gb=1.5,
        priority=4,
        quantize="int8",
    ),
}


# ---------------------------------------------------------------------------
# 自动分级决策
# ---------------------------------------------------------------------------

def get_llm_config(vram_gb: Optional[int] = None) -> ModelConfig:
    """根据显存量自动选择合适的 LLM 模型"""
    if vram_gb is None:
        vram_gb = detect_vram_gb()
    if vram_gb >= 8:
        return LLM_CONFIG_TIERS["high"]
    elif vram_gb >= 6:
        return LLM_CONFIG_TIERS["medium"]
    elif vram_gb >= 4:
        return LLM_CONFIG_TIERS["low"]
    else:
        return LLM_CONFIG_TIERS["cpu"]


def get_lipsync_config(vram_gb: Optional[int] = None) -> ModelConfig:
    """根据显存量自动选择口型合成方案"""
    if vram_gb is None:
        vram_gb = detect_vram_gb()
    if vram_gb >= 8:
        return LIPSYNC_CONFIG_TIERS["high"]     # MuseTalk 优先
    elif vram_gb >= 6:
        return LIPSYNC_CONFIG_TIERS["medium"]   # Wav2Lip 标准
    else:
        return LIPSYNC_CONFIG_TIERS["low"]      # Wav2Lip 轻量


def get_all_configs(vram_gb: Optional[int] = None) -> dict:
    """
    返回所有环节的模型配置。
    Returns:
        {
            "asr": ModelConfig,
            "llm": ModelConfig,
            "tts": ModelConfig,
            "lipsync": ModelConfig,
        }
    """
    if vram_gb is None:
        vram_gb = detect_vram_gb()

    return {
        "asr": ASR_CONFIG,
        "llm": get_llm_config(vram_gb),
        "tts": TTS_CONFIG,
        "lipsync": get_lipsync_config(vram_gb),
    }


def print_model_plan(vram_gb: Optional[int] = None):
    """打印当前硬件下的模型选择方案"""
    if vram_gb is None:
        vram_gb = detect_vram_gb()

    configs = get_all_configs(vram_gb)
    budget = VRAMBudget(total_gb=vram_gb)

    print(f"\n{'='*60}")
    print(f"🎯 本地模型适配方案 — GPU 显存：{vram_gb} GB")
    print(f"{'='*60}")
    print(f"{'环节':<10} {'模型':<28} {'显存(G)':<8} {'量化':<8}")
    print("-" * 60)

    total_vram = 0
    for key, cfg in configs.items():
        tag_map = {"asr": "语音识别", "llm": "文案仿写", "tts": "语音合成", "lipsync": "口型合成"}
        tag = tag_map.get(key, key)
        quant = cfg.quantize or "fp16"
        print(f"{tag:<10} {cfg.name:<28} {cfg.vram_estimate_gb:<8.1f} {quant:<8}")
        total_vram += cfg.vram_estimate_gb

    print("-" * 60)
    print(f"{'合计（串行加载）':<10} {'峰值约':<28} {total_vram:<8.1f}")
    print(f"⚠️  所有模型分时加载，实际峰值 ≈ {max(c.vram_estimate_gb for c in configs.values()):.1f} GB")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# 便捷测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_model_plan()
