
"""
本地语音合成引擎（CosyVoice AutoModel + 本地模型）
"""
import gc
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple, Generator, Dict

import numpy as np
import torch
import soundfile as sf

# 添加 CosyVoice 仓库到 Python 路径（项目根目录下的 CosyVoice/）
_ROOOT = Path(__file__).resolve().parent.parent  # local_models/ 的上一级 = IP_Agent/
_COSYVOICE_DIR = _ROOOT / "CosyVoice"
if str(_COSYVOICE_DIR) not in sys.path:
    sys.path.insert(0, str(_COSYVOICE_DIR))
sys.path.insert(0, str(_COSYVOICE_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel

from local_models.config import TTS_CONFIG

logger = logging.getLogger(__name__)


class CosyVoiceEngine:
    """CosyVoice-300M-SFT TTS 引擎 — 使用 CosyVoice 官方 AutoModel 加载"""

    def __init__(self, local_path: str = None):
        self.model_path = local_path or TTS_CONFIG["local_path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[AutoModel] = None

    # ---- 模型加载 ----

    def load_model(self) -> bool:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                logger.warning("⚠️ 未检测到 CUDA，CosyVoice 将在 CPU 上运行（较慢）")

            logger.info(f"加载 CosyVoice AutoModel 从: {self.model_path}")
            use_fp16 = (self.device == "cuda")
            self.model = AutoModel(model_dir=self.model_path, fp16=use_fp16)
            self.model.sample_rate  # 触发实际加载

            dtype_info = "fp16" if use_fp16 else "fp32"
            gpu_info = f"GPU={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
            logger.info(f"CosyVoice 加载完成 ({gpu_info}, dtype={dtype_info})")

            # 列出可用音色
            try:
                spks = self.model.list_available_spks()
                logger.info(f"可用音色: {spks}")
            except Exception:
                pass

            return True
        except Exception as e:
            logger.error(f"CosyVoice 加载失败: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    @property
    def sample_rate(self) -> int:
        return getattr(self.model, "sample_rate", 22050) if self.model else 22050

    # ---- 推理：单段文本 → 原始音频 ----

    def synthesize_raw(
        self, text: str, speaker: str = "default", speed: float = 1.0,
    ) -> Optional[np.ndarray]:
        """合成单段文本，返回原始音频 numpy 数组 [samples, channels]。
        调用方负责拼接和写出，避免频繁文件 I/O。
        """
        if self.model is None and not self.load_model():
            return None
        speaker = self._resolve_speaker(speaker)

        result = list(self.model.inference_sft(
            text, speaker, stream=False, speed=speed,
        ))
        if not result:
            return None

        wav_list = []
        for seg in result:
            wav_seg = seg["tts_speech"].cpu()
            if wav_seg.ndim == 2:
                wav_seg = wav_seg.T
            else:
                wav_seg = wav_seg.unsqueeze(-1)
            wav_list.append(wav_seg)
        return np.concatenate(wav_list, axis=0)

    # ---- 推理：逐段合成 + 静音间隔 + 时间戳 ----

    def synthesize_segments(
        self, segments: List[str], speaker: str = "default",
        speed: float = 1.0, gap_ms: int = 500,
        output_path: Optional[str] = None,
    ) -> Optional[Tuple[str, List[dict]]]:
        """逐段合成文本并插入静音间隔，返回 (音频路径, 时间轴)。

        每个 segment 独立推理，避免长文本 OOM；段落间插入 gap_ms 毫秒静音。
        时间轴中每项: {"index": int, "text": str, "start_s": float, "end_s": float}
        时间严格对齐实际音频样本数，不依赖估算。
        """
        if self.model is None and not self.load_model():
            return None

        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        speaker = self._resolve_speaker(speaker)
        sr = self.sample_rate
        gap_samples = int(sr * gap_ms / 1000.0)
        gap_chunk = np.zeros((gap_samples, 1), dtype=np.float32)

        audio_parts: List[np.ndarray] = []
        timeline: List[dict] = []
        cursor_samples = 0  # 当前累计采样点（用于计算时间轴）

        for i, seg_text in enumerate(segments):
            seg_text = seg_text.strip()
            if not seg_text:
                continue

            logger.info(f"[{i+1}/{len(segments)}] 合成: {seg_text[:50]}...")
            wav = self.synthesize_raw(seg_text, speaker, speed)
            if wav is None:
                logger.error(f"段 [{i}] 合成失败，跳过")
                continue

            # 记录时间轴（基于实际音频样本数）
            dur_s = wav.shape[0] / sr
            start_s = cursor_samples / sr
            end_s = start_s + dur_s
            timeline.append({
                "index": i + 1,
                "text": seg_text,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "dur_s": round(dur_s, 3),
                "sample_start": cursor_samples,
                "sample_end": cursor_samples + wav.shape[0],
            })

            audio_parts.append(wav)
            cursor_samples += wav.shape[0]

            # 段落间插入静音（最后一段不加）
            if i < len(segments) - 1 and gap_ms > 0:
                audio_parts.append(gap_chunk)
                cursor_samples += gap_samples
                logger.debug(f"  插入静音间隔 {gap_ms}ms")

        if not audio_parts:
            logger.error("所有段合成均失败")
            return None

        full_audio = np.concatenate(audio_parts, axis=0)
        sf.write(output_path, full_audio, sr)
        total_s = full_audio.shape[0] / sr
        logger.info(f"语音合成完成: {output_path} ({total_s:.2f}s, {len(timeline)}段, 间隔{gap_ms}ms)")
        # 重新计算 timeline 的 end_s（因为静音段的加入，实际位置不变因为我们按样本位记录）
        # 修正：时间轴中每个段落使用的是绝对样本偏移，不受静音影响
        return output_path, timeline

    # ---- 兼容旧接口 ----

    def synthesize(
        self, text: str, speaker: str = "default", speed: float = 1.0,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """兼容旧接口：整段文本合成（内部复用 synthesize_segments）。"""
        # 按空行分段，无间隔，以保持与原行为一致
        segments = [s.strip() for s in text.split("\n\n") if s.strip()]
        if not segments:
            segments = [text.strip()]
        result = self.synthesize_segments(
            segments, speaker, speed, gap_ms=0, output_path=output_path,
        )
        if result:
            return result[0]
        return None

    def _resolve_speaker(self, speaker: str) -> str:
        """将用户输入映射为 CosyVoice 能识别的发音人。"""
        if self.model is None:
            return speaker

        available = self.model.list_available_spks() if self.model else []

        # 增强映射：为常见输入补充备选名
        alias = {
            "default": ["中文女", "默认女声"],
            "中文女":  ["中文女"],
            "中文男":  ["中文男"],
            "默认女声": ["中文女"],
            "默认男声": ["中文男"],
            "男声":    ["中文男"],
            "女声":    ["中文女"],
        }

        if isinstance(available, list):
            mapped = alias.get(speaker, [speaker])
            for m in mapped:
                if m in available:
                    return m

        return speaker

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
