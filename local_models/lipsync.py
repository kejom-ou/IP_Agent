"""
============================================================
本地口型合成引擎（local_models/lipsync.py）
============================================================
提供统一的本地口型合成接口，自动根据显存选择最优方案：
  - 8GB+  → MuseTalk（实时高画质）
  - 6GB+  → Wav2Lip（兼容性好）
  - <6GB  → Wav2Lip-Lite（最低配兜底）

设计目标：
  1. 资源占用小：fp16/int8 推理，用完即释放显存
  2. 效果好：优先 MuseTalk，降级 Wav2Lip
  3. 接口统一：与原有 generate_tuilionnx_video 兼容
============================================================
"""

import os
import gc
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Literal

import torch
import numpy as np

from local_models.config import (
    get_lipsync_config,
    detect_vram_gb,
    LIPSYNC_CONFIG_TIERS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 显存管理工具
# ---------------------------------------------------------------------------

class VRAMManager:
    """显存管理器：在模型间切换时确保前一个模型释放显存"""

    @staticmethod
    def clear():
        """强制释放所有未使用的 GPU 显存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def get_usage_gb() -> float:
        """获取当前已用显存（GB）"""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 3)

    @staticmethod
    def get_free_gb() -> float:
        """获取当前可用显存（GB）"""
        if not torch.cuda.is_available():
            return 0.0
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return total - VRAMManager.get_usage_gb()


# ---------------------------------------------------------------------------
# 方案 1：MuseTalk 封装（高画质，需 ~4GB）
# ---------------------------------------------------------------------------

class MuseTalkEngine:
    """
    MuseTalk 本地推理引擎封装。
    基于 TMElyralab/MuseTalk，在潜在空间中生成唇形同步。
    官方最低要求：RTX 3050 Ti 4GB (fp16)，8GB 绰绰有余。
    """

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self._loaded = False

    def load(self) -> bool:
        """加载 MuseTalk 模型到显存"""
        try:
            VRAMManager.clear()
            logger.info(f"🔄 正在加载 MuseTalk 模型... 当前可用显存: {VRAMManager.get_free_gb():.1f} GB")

            # 使用 ModelScope 镜像（国内加速）
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            self.pipeline = pipeline(
                Tasks.talking_head_lip_sync,
                model="TMElyralab/MuseTalk",
                device=self.device,
            )
            self._loaded = True
            logger.info(f"✅ MuseTalk 加载完成，显存占用: {VRAMManager.get_usage_gb():.1f} GB")
            return True

        except ImportError:
            logger.warning(
                "⚠️ modelscope 未安装，MuseTalk 不可用。\n"
                "   安装方式: pip install modelscope\n"
                "   降级为 Wav2Lip..."
            )
            return False
        except Exception as e:
            logger.error(f"❌ MuseTalk 加载失败: {e}")
            return False

    def generate(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        fps: int = 25,
    ) -> Optional[str]:
        """
        生成口型同步视频。

        Args:
            video_path:  输入视频路径（含人脸）
            audio_path:  输入音频路径（驱动音频）
            output_path: 输出视频路径
            fps:         输出帧率

        Returns:
            输出视频路径，失败返回 None
        """
        if not self._loaded:
            logger.error("MuseTalk 模型未加载")
            return None

        try:
            logger.info(f"🎬 MuseTalk 生成中... 视频: {video_path}, 音频: {audio_path}")
            result = self.pipeline(
                dict(
                    video_path=video_path,
                    audio_path=audio_path,
                    output_video_path=output_path,
                    fps=fps,
                )
            )
            logger.info(f"✅ MuseTalk 生成完成: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ MuseTalk 生成失败: {e}")
            return None

    def unload(self):
        """从显存卸载模型"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        self._loaded = False
        VRAMManager.clear()
        logger.info("🗑️  MuseTalk 已从显存卸载")


# ---------------------------------------------------------------------------
# 方案 2：Wav2Lip 封装（标准画质，需 ~2.5GB）
# ---------------------------------------------------------------------------

class Wav2LipEngine:
    """
    Wav2Lip 本地推理引擎封装。
    经典唇形同步方案，兼容性最好，显存需求低。
    最低 6GB 显存可流畅运行，int8 量化后仅需 ~1.5GB。
    """

    def __init__(self, device: str = "cuda", use_int8: bool = False):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_int8 = use_int8
        self.model = None
        self.face_detector = None
        self._loaded = False

    def load(self) -> bool:
        """加载 Wav2Lip 模型"""
        try:
            VRAMManager.clear()
            logger.info(f"🔄 正在加载 Wav2Lip 模型... 当前可用显存: {VRAMManager.get_free_gb():.1f} GB")

            # 导入 Wav2Lip 核心模块
            import sys
            wav2lip_dir = os.path.join(os.path.dirname(__file__), "..", "Wav2Lip")
            if os.path.exists(wav2lip_dir):
                sys.path.insert(0, wav2lip_dir)

            # 尝试从 ModelScope 加载预训练权重
            from modelscope.hub.file_download import model_file_download

            model_path = model_file_download(
                model_id="hunyuan/Wav2Lip",
                file_path="wav2lip_gan.pth",
            )
            face_det_path = model_file_download(
                model_id="hunyuan/Wav2Lip",
                file_path="s3fd.pth",
            )

            # 加载模型
            from Wav2Lip.models import Wav2Lip as Wav2LipModel
            self.model = Wav2LipModel()
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()

            if self.use_int8 and self.device == "cuda":
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                )

            self._loaded = True
            logger.info(f"✅ Wav2Lip 加载完成，显存占用: {VRAMManager.get_usage_gb():.1f} GB")
            return True

        except ImportError as e:
            logger.warning(
                f"⚠️ Wav2Lip 依赖未安装: {e}\n"
                "   安装方式: git clone https://github.com/Rudrabha/Wav2Lip.git"
            )
            return False
        except Exception as e:
            logger.error(f"❌ Wav2Lip 加载失败: {e}")
            return False

    def generate(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        fps: int = 25,
        pads: tuple = (0, 10, 0, 0),
        resize_factor: int = 1,
    ) -> Optional[str]:
        """
        生成口型同步视频。
        """
        if not self._loaded:
            logger.error("Wav2Lip 模型未加载")
            return None

        try:
            import subprocess
            cmd = [
                "python", "Wav2Lip/inference.py",
                "--checkpoint_path", "Wav2Lip/checkpoints/wav2lip_gan.pth",
                "--face", video_path,
                "--audio", audio_path,
                "--outfile", output_path,
                "--fps", str(fps),
                "--pads", *[str(p) for p in pads],
                "--resize_factor", str(resize_factor),
            ]
            logger.info(f"🎬 Wav2Lip 生成: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=os.path.dirname(__file__))
            logger.info(f"✅ Wav2Lip 生成完成: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ Wav2Lip 生成失败: {e}")
            return None

    def unload(self):
        """从显存卸载模型"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.face_detector is not None:
            del self.face_detector
            self.face_detector = None
        self._loaded = False
        VRAMManager.clear()
        logger.info("🗑️  Wav2Lip 已从显存卸载")


# ---------------------------------------------------------------------------
# 统一接口
# ---------------------------------------------------------------------------

class LipSyncEngine:
    """
    口型合成统一入口，自动选择最优方案。

    使用示例:
        engine = LipSyncEngine()
        engine.generate("input.mp4", "audio.wav", "output.mp4")
    """

    def __init__(self, force_engine: Optional[Literal["musetalk", "wav2lip"]] = None):
        """
        Args:
            force_engine: 强制指定引擎（None = 自动选择）
        """
        self.force_engine = force_engine
        self.config = get_lipsync_config()
        self.engine = None
        self._initialized = False

    def _select_engine(self) -> str:
        """根据显存和配置选择最优引擎"""
        if self.force_engine:
            return self.force_engine

        vram = detect_vram_gb()
        if vram >= 8:
            preferred = "musetalk"
        elif vram >= 6:
            preferred = "wav2lip"
        else:
            preferred = "wav2lip_lite"

        logger.info(f"🔍 自动选择口型引擎: {preferred} (显存: {vram} GB)")
        return preferred

    def init(self) -> bool:
        """初始化引擎"""
        engine_type = self._select_engine()

        if engine_type == "musetalk":
            self.engine = MuseTalkEngine()
            if self.engine.load():
                self._initialized = True
                return True
            # MuseTalk 失败，降级到 Wav2Lip
            logger.warning("⚠️ MuseTalk 初始化失败，降级到 Wav2Lip...")
            VRAMManager.clear()
            engine_type = "wav2lip"

        if engine_type in ("wav2lip", "wav2lip_lite"):
            use_int8 = (engine_type == "wav2lip_lite")
            self.engine = Wav2LipEngine(use_int8=use_int8)
            if self.engine.load():
                self._initialized = True
                return True

        logger.error("❌ 所有口型引擎均初始化失败")
        return False

    def generate(
        self,
        video_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
        fps: int = 25,
    ) -> Optional[str]:
        """
        生成口型同步视频（统一接口，与原 TuiliONNX 接口兼容）。

        Args:
            video_path:  输入视频路径
            audio_path:  驱动音频路径
            output_path: 输出路径（None 则自动生成）
            fps:         帧率

        Returns:
            输出视频路径
        """
        if not self._initialized:
            if not self.init():
                return None

        if output_path is None:
            suffix = Path(video_path).suffix or ".mp4"
            output_path = str(
                Path(video_path).parent
                / f"{Path(video_path).stem}_lipsync{suffix}"
            )

        result = self.engine.generate(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path,
            fps=fps,
        )

        # 生成完成后释放显存（后续步骤不需要口型模型）
        self.unload()
        return result

    def unload(self):
        """释放引擎显存"""
        if self.engine is not None:
            self.engine.unload()
            self.engine = None
        self._initialized = False

    @property
    def engine_name(self) -> str:
        return type(self.engine).__name__ if self.engine else "未初始化"


# ---------------------------------------------------------------------------
# 便捷函数：兼容原 generate_tuilionnx_video 接口
# ---------------------------------------------------------------------------

def generate_lipsync_video(
    face_model: str,       # 原接口保留，本地版忽略（用 MuseTalk/Wav2Lip 替换）
    video_path: str,
    audio_path: str,
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
    与原 TuiliONNX 接口兼容的口型合成函数。
    将 TuiliONNX 替换为 MuseTalk/Wav2Lip 本地引擎。

    Returns:
        (video_path, generation_time, download_file, share_url)
    """
    import time

    start = time.time()
    engine = LipSyncEngine()

    output_path = str(Path(audio_path).parent / f"lipsync_output_{int(start)}.mp4")

    result = engine.generate(
        video_path=video_path,
        audio_path=audio_path,
        output_path=output_path,
    )

    elapsed = time.time() - start
    time_str = f"生成耗时: {elapsed:.1f}s | 引擎: {engine.engine_name}"

    if result:
        logger.info(f"✅ 口型合成完成 ({time_str})")
        return result, time_str, result, None
    else:
        logger.error(f"❌ 口型合成失败 ({time_str})")
        return None, time_str, None, None
