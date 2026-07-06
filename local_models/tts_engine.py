

"""
本地语音合成引擎（CosyVoice AutoModel + 本地模型）
"""
import gc
import logging
import os
import re
import sys
import tempfile
import time
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


# ---------------------------------------------------------------------------
# 文本清洗：过滤 emoji 和特殊字符，避免 TTS 合成出怪声 / 卡住
# ---------------------------------------------------------------------------

# 匹配所有 emoji 区段（包括组合序列）和常见装饰符号
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # 符号 & 表情
    "\U0001F600-\U0001F64F"  # 表情符号
    "\U0001F680-\U0001F6FF"  # 交通 & 地图
    "\U0001F700-\U0001F77F"  # 炼金术符号
    "\U0001F780-\U0001F7FF"  # 几何形状扩展
    "\U0001F800-\U0001F8FF"  # 补充箭头
    "\U0001F900-\U0001F9FF"  # 补充符号 & 表情
    "\U0001FA00-\U0001FAFF"  # 扩展符号 & 表情
    "\U00002600-\U000027BF"  # 杂项符号 / 装饰
    "\U00002B00-\U00002BFF"  # 杂项符号 & 箭头
    "\U0001F1E6-\U0001F1FF"  # 国旗
    "\U0001F000-\U0001F02F"  # 麻将 / 扑克
    "\U0001F0A0-\U0001F0FF"  # 扑克牌
    "\U0000FE00-\U0000FE0F"  # 变体选择符（emoji 组合）
    "\U0000200D"             # 零宽连接符
    "\U00002702-\U000027B0"  # 装饰符号
    "]+",
    flags=re.UNICODE,
)

# 其它常见特殊符号：星号、方头括号、箭头、圈数字等
_SPECIAL_SYMBOLS_PATTERN = re.compile(
    "["
    ""
    "◆◇■□●○"
    "▲△▼▽"
    "→←↑↓⇒⇐"
    "①②③④⑤⑥⑦⑧⑨⑩"
    "⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽"
    ""
    "─━│┃"
    "…"
    "·•・"
    "【】〖〗"
    "⟨⟩"
    "©®™"
    ""
    ""
    "]+",
    flags=re.UNICODE,
)


def clean_text_for_tts(text: str) -> str:
    """
    清洗送给 TTS 的文本：去除 emoji 和特殊装饰符号。

    保留：
      - 中文字符（CJK）
      - 英文字母 / 数字
      - 标准中文标点（，。！？；：""''【】()、《》——…）
      - 标准英文标点（,.!?:;'"()-）
      - 空白字符
    """
    if not text:
        return text

    # 1) 先去掉 emoji
    text = _EMOJI_PATTERN.sub("", text)
    # 2) 再去掉装饰符号
    text = _SPECIAL_SYMBOLS_PATTERN.sub("", text)
    # 3) 折叠连续空白
    text = re.sub(r"[ \t]+", " ", text)
    # 4) 折叠连续换行
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


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
                logger.warning("未检测到 CUDA，CosyVoice 将在 CPU 上运行（较慢）")

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

    def synthesize_with_prompt(
        self, text: str, prompt_audio_path: str, prompt_text: str = "",
        speed: float = 1.0,
    ) -> Optional[np.ndarray]:
        """使用参考音频做 zero-shot 音色克隆（CosyVoice-300M-SFT 不支持时回退到 SFT）。

        Args:
            text: 待合成文本
            prompt_audio_path: 参考音频 wav 路径（用户上传 / 录音）
            prompt_text: 参考音频对应的文本（zero-shot 需要）
            speed: 语速

        Returns:
            原始音频 numpy 数组 [samples, 1]
        """
        if self.model is None and not self.load_model():
            return None

        if not prompt_audio_path or not os.path.exists(prompt_audio_path):
            logger.warning("未提供参考音频，回退到默认 SFT 音色")
            return self.synthesize_raw(text, "default", speed)

        # 把音频重采样到模型采样率，避免 sr 不匹配
        # 优先用 torchaudio 加载 wav；若失败（如 mp3/m4a 无后端）用 ffmpeg 转码
        try:
            import torchaudio
            wav, sr = torchaudio.load(prompt_audio_path)
            target_sr = self.sample_rate
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                wav = resampler(wav)
                # 写回临时文件
                tmp_prompt = os.path.join(
                    tempfile.gettempdir(),
                    f"_prompt_resampled_{int(time.time() * 1000)}.wav",
                )
                torchaudio.save(tmp_prompt, wav, target_sr)
                prompt_audio_path = tmp_prompt
        except ImportError:
            pass  # torchaudio 不在则用 ffmpeg 兜底
        except Exception as e:
            logger.warning(f"torchaudio 加载参考音频失败，尝试 ffmpeg 兜底: {e}")
            try:
                import subprocess
                target_sr = self.sample_rate
                tmp_prompt = os.path.join(
                    tempfile.gettempdir(),
                    f"_prompt_ffmpeg_{int(time.time() * 1000)}.wav",
                )
                # ffmpeg 转码为 wav 单声道 target_sr
                subprocess.run(
                    ["ffmpeg", "-y", "-i", prompt_audio_path,
                     "-ac", "1", "-ar", str(target_sr), tmp_prompt],
                    check=True, capture_output=True, timeout=30,
                )
                prompt_audio_path = tmp_prompt
                logger.info(f"ffmpeg 转码成功: {tmp_prompt}")
            except Exception as fe:
                logger.error(f"ffmpeg 兜底也失败: {fe}")
                # 不抛出，让 zero-shot 自己用原路径尝试（CosyVoice 内部也可能处理）

        # 默认 prompt 文本（用户没填时）
        if not prompt_text or not prompt_text.strip():
            prompt_text = "这是一段示例音频，用于音色克隆。"

        # 尝试 zero-shot
        try:
            if hasattr(self.model, "inference_zero_shot"):
                logger.info("[ZeroShot] 使用参考音频做音色克隆")
                result = list(self.model.inference_zero_shot(
                    text, prompt_text, prompt_audio_path,
                    zero_shot_spk_id="user_uploaded",
                    stream=False, speed=speed,
                ))
                if result:
                    wav_list = []
                    for seg in result:
                        w = seg["tts_speech"].cpu()
                        if w.ndim == 2:
                            w = w.T
                        else:
                            w = w.unsqueeze(-1)
                        wav_list.append(w)
                    return np.concatenate(wav_list, axis=0)
        except Exception as e:
            logger.warning(f"Zero-shot 失败，回退 SFT: {e}")

        # 回退
        logger.info("[SFT] Zero-shot 不可用，使用默认中文女声")
        return self.synthesize_raw(text, "default", speed)


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
        """整段文本合成（按中文句号分句，逐句独立推理避免长文本 OOM）。"""
        # 按句号分句，保留句号在每句末尾
        segments = [s.strip() for s in re.split(r'(?<=。)', text) if s.strip()]
        if not segments:
            segments = [text.strip()]
        result = self.synthesize_segments(
            segments, speaker, speed, gap_ms=300, output_path=output_path,
        )
        if result:
            return result[0]
        return None

    def _save_with_prompt(
        self, text: str, prompt_audio_path: str, prompt_text: str = "",
        speed: float = 1.0,
    ) -> Optional[str]:
        """Zero-shot 音色克隆 + 写出 wav 文件，返回文件路径"""
        wav = self.synthesize_with_prompt(
            text=text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            speed=speed,
        )
        if wav is None:
            return None
        out_path = os.path.join(
            tempfile.gettempdir(),
            f"tts_cloned_{int(time.time() * 1000)}.wav",
        )
        try:
            import soundfile as sf
            sf.write(out_path, wav, self.sample_rate)
            logger.info(f"[TTS] 写出音频: {out_path}")
            return out_path
        except Exception as e:
            logger.error(f"写 wav 失败: {e}")
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
        """卸载模型并释放显存。

        Windows 上 del torch 模型对象时，ProactorEventLoop 会主动关闭底层 socket，
        这会触发 ConnectionResetError (10054)。这里的 gc.collect() 会立刻触发该清理，
        导致看到 "主机强迫关闭了一个现有的连接" 的告警。属于无害的 Windows 平台噪音。
        """
        if self.model is not None:
            try:
                del self.model
            except Exception as e:
                logger.debug(f"卸载 CosyVoice 模型时小警告: {e}")
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
        return audio_path, "语音合成完成"
    return None, "语音合成失败"


def generate_audio_only_local(text: str, api_key: Optional[str] = None) -> Tuple[Optional[str], str]:
    return handle_audio_creation_local(text, 0, 1.0)
