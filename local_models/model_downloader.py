"""
============================================================
模型下载器（local_models/model_downloader.py）
============================================================
一键下载所有本地模型，支持断点续传和进度显示。

使用方法:
    python local_models/model_downloader.py          # 交互式选择
    python local_models/model_downloader.py --all    # 下载全部
    python local_models/model_downloader.py --check  # 仅检查状态
============================================================
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "pretrained_models"


@dataclass
class ModelInfo:
    """模型下载信息"""
    name: str                    # 模型名称
    description: str             # 用途描述
    size_gb: float               # 下载大小（GB）
    vram_gb: float               # 运行时显存占用
    required: bool               # 是否必需
    download_method: str         # "modelscope" / "huggingface" / "ollama"
    model_id: str                # ModelScope/HuggingFace ID 或 ollama model name


# ---------------------------------------------------------------------------
# 模型清单
# ---------------------------------------------------------------------------

ALL_MODELS = [
    ModelInfo(
        name="faster-whisper-small",
        description="语音识别（ASR）— 将视频音频转为口播文案",
        size_gb=0.5,
        vram_gb=1.5,
        required=True,
        download_method="auto",     # faster-whisper 自动下载
        model_id="small",
    ),
    ModelInfo(
        name="CosyVoice-300M-SFT",
        description="语音合成（TTS）— 文字转语音 + 声音克隆",
        size_gb=0.6,
        vram_gb=2.0,
        required=True,
        download_method="modelscope",
        model_id="iic/CosyVoice-300M-SFT",
    ),
    ModelInfo(
        name="MuseTalk",
        description="口型合成（高画质）— 8GB+ 显存推荐",
        size_gb=1.5,
        vram_gb=4.0,
        required=False,             # 非必需，可降级为 Wav2Lip
        download_method="modelscope",
        model_id="TMElyralab/MuseTalk",
    ),
    ModelInfo(
        name="Wav2Lip",
        description="口型合成（标准）— 6GB 显存可用",
        size_gb=0.5,
        vram_gb=2.5,
        required=False,             # 非必需，MuseTalk 的降级方案
        download_method="modelscope",
        model_id="hunyuan/Wav2Lip",
    ),
    ModelInfo(
        name="Qwen2.5-3B-Instruct (Ollama)",
        description="文案仿写（LLM）— 推荐 Ollama 方式",
        size_gb=2.0,
        vram_gb=3.5,
        required=True,
        download_method="ollama",
        model_id="qwen2.5:3b",
    ),
    ModelInfo(
        name="Qwen2.5-1.5B-Instruct (Ollama)",
        description="文案仿写（LLM）— 低配显卡备用",
        size_gb=1.0,
        vram_gb=1.8,
        required=False,
        download_method="ollama",
        model_id="qwen2.5:1.5b",
    ),
]


# ---------------------------------------------------------------------------
# 下载方法
# ---------------------------------------------------------------------------

def download_from_modelscope(model_id: str, save_dir: str) -> bool:
    """从 ModelScope 下载模型"""
    try:
        from modelscope import snapshot_download
        logger.info(f"⬇️  ModelScope 下载: {model_id}")
        snapshot_download(model_id, cache_dir=save_dir)
        logger.info(f"✅ 下载完成: {model_id}")
        return True
    except ImportError:
        logger.error("❌ modelscope 未安装。pip install modelscope")
        return False
    except Exception as e:
        logger.error(f"❌ 下载失败: {e}")
        return False


def download_from_ollama(model_name: str) -> bool:
    """通过 Ollama 拉取模型"""
    import subprocess
    try:
        logger.info(f"⬇️  Ollama 拉取: {model_name}")
        subprocess.run(["ollama", "pull", model_name], check=True)
        logger.info(f"✅ 拉取完成: {model_name}")
        return True
    except FileNotFoundError:
        logger.error(
            "❌ Ollama 未安装。\n"
            "   安装: curl -fsSL https://ollama.com/install.sh | sh"
        )
        return False
    except Exception as e:
        logger.error(f"❌ Ollama 拉取失败: {e}")
        return False


def download_whisper_model(model_size: str = "small") -> bool:
    """触发 faster-whisper 自动下载模型（首次使用时自动缓存）"""
    try:
        from faster_whisper import WhisperModel
        logger.info(f"⬇️  触发 Whisper 模型下载: {model_size}")
        # 临时加载触发下载，然后立即释放
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        del model
        logger.info(f"✅ Whisper 模型就绪: {model_size}")
        return True
    except ImportError:
        logger.error("❌ faster-whisper 未安装。pip install faster-whisper")
        return False
    except Exception as e:
        logger.error(f"❌ Whisper 下载失败: {e}")
        return False


# ---------------------------------------------------------------------------
# 状态检查
# ---------------------------------------------------------------------------

def check_model_status(model: ModelInfo) -> str:
    """检查模型下载状态"""
    if model.download_method == "ollama":
        import subprocess
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True,
            )
            if model.model_id.split(":")[0] in result.stdout:
                return "✅ 已安装"
            return "⬜ 未安装"
        except FileNotFoundError:
            return "⚠️  Ollama 未安装"

    elif model.download_method == "modelscope":
        # 检查 ModelScope 缓存
        cache_dir = os.path.expanduser(f"~/.cache/modelscope/hub/{model.model_id}")
        if os.path.exists(cache_dir):
            return "✅ 已下载"
        # 也检查本地 pretrained_models
        local_dir = MODELS_DIR / model.model_id.split("/")[-1]
        if local_dir.exists():
            return "✅ 已下载"
        return "⬜ 未下载"

    elif model.download_method == "auto":
        # faster-whisper 检查缓存
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        if os.path.exists(cache_dir):
            return "✅ 已缓存"
        return "⬜ 未缓存"

    return "❓ 未知"


# ---------------------------------------------------------------------------
# 交互式 / 批量下载
# ---------------------------------------------------------------------------

def print_status():
    """打印所有模型的状态"""
    vram = 0
    if torch.cuda.is_available():
        vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))

    print(f"\n{'='*70}")
    print(f"📦 本地模型状态 — GPU 显存: {vram} GB")
    print(f"{'='*70}")
    print(f"{'模型':<30} {'大小':<8} {'显存':<8} {'状态':<12} {'必需':<6}")
    print("-" * 70)

    for model in ALL_MODELS:
        status = check_model_status(model)
        req = "✅" if model.required else "⚪"
        print(f"{model.name:<30} {model.size_gb:<8.1f} {model.vram_gb:<8.1f} {status:<12} {req:<6}")

    print(f"{'='*70}\n")


def download_all(interactive: bool = True):
    """下载所有模型"""
    print_status()

    for model in ALL_MODELS:
        if not model.required:
            if interactive:
                answer = input(f"下载 {model.name} ({model.description})? [y/N]: ")
                if answer.lower() != "y":
                    logger.info(f"⏭️  跳过: {model.name}")
                    continue

        logger.info(f"\n{'='*50}")
        logger.info(f"📥 下载: {model.name} ({model.size_gb} GB)")
        logger.info(f"   用途: {model.description}")

        if model.download_method == "modelscope":
            success = download_from_modelscope(model.model_id, str(MODELS_DIR))
        elif model.download_method == "ollama":
            success = download_from_ollama(model.model_id)
        elif model.download_method == "auto":
            success = download_whisper_model(model.model_id)
        else:
            success = False

        if success:
            logger.info(f"✅ {model.name} 下载成功")
        else:
            logger.error(f"❌ {model.name} 下载失败")


def download_essentials():
    """仅下载必需模型"""
    logger.info("📥 下载必需模型...")
    for model in ALL_MODELS:
        if not model.required:
            continue
        logger.info(f"   下载: {model.name}")

        if model.download_method == "modelscope":
            download_from_modelscope(model.model_id, str(MODELS_DIR))
        elif model.download_method == "ollama":
            download_from_ollama(model.model_id)
        elif model.download_method == "auto":
            download_whisper_model(model.model_id)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="旗博士 AI 本地模型下载器")
    parser.add_argument("--all", action="store_true", help="下载全部模型")
    parser.add_argument("--essentials", action="store_true", help="仅下载必需模型")
    parser.add_argument("--check", action="store_true", help="仅检查模型状态")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过交互确认")
    args = parser.parse_args()

    if args.check:
        print_status()
    elif args.essentials:
        download_essentials()
    elif args.all:
        download_all(interactive=not args.yes)
    else:
        print_status()
        print("\n用法:")
        print("  python model_downloader.py --check      查看状态")
        print("  python model_downloader.py --essentials 下载必需模型")
        print("  python model_downloader.py --all        下载全部模型")
        print("  python model_downloader.py --all -y     静默下载全部")


if __name__ == "__main__":
    main()
