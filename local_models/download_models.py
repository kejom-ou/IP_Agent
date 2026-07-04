"""
一键下载所有模型到 pretrained_models/

用法:
    python local_models/download_models.py          # 下载全部
    python local_models/download_models.py --skip-llm  # 跳过 LLM
"""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("download")

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "pretrained_models"

# ModelScope 模型 ID → 本地子目录
MODELS = [
    ("ASR: SenseVoiceSmall",            "iic/SenseVoiceSmall",                           "SenseVoiceSmall"),
    ("LLM: Qwen2.5-0.5B-Instruct",      "Qwen/Qwen2.5-0.5B-Instruct",                   "Qwen2.5-0.5B-Instruct"),
    ("TTS: CosyVoice-300M-SFT (Lite)",  "iic/CosyVoice-300M-SFT",                       "CosyVoice-300M-SFT"),
]

# Wav2Lip 模型需手动下载（不在 ModelScope 上）
#   wav2lip_gan.pth → pretrained_models/Wav2Lip/wav2lip_gan.pth
#   下载地址: https://drive.google.com/file/d/15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk/view


def download_model(name: str, model_id: str, subdir: str):
    """使用 ModelScope snapshot_download 下载模型"""
    target = MODELS_DIR / subdir
    if target.exists() and any(target.iterdir()):
        logger.info(f"⏭️  {name} 已存在，跳过")
        return True

    logger.info(f"⬇️  下载 {name} → {target}")
    try:
        from modelscope import snapshot_download
        snapshot_download(
            model_id=model_id,
            local_dir=str(target),
            revision="master",
        )
        logger.info(f"✅ {name} 下载完成")
        return True
    except Exception as e:
        logger.error(f"❌ {name} 下载失败: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="一键下载所有模型")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 LLM 下载")
    parser.add_argument("--skip-tts", action="store_true", help="跳过 TTS 下载")
    parser.add_argument("--skip-asr", action="store_true", help="跳过 ASR 下载")
    parser.add_argument("--skip-lipsync", action="store_true", help="跳过 LipSync 下载")
    args = parser.parse_args()

    skip_map = {
        "ASR": args.skip_asr,
        "LLM": args.skip_llm,
        "TTS": args.skip_tts,
        "LipSync": args.skip_lipsync,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"模型目录: {MODELS_DIR}\n")

    results = []
    for name, model_id, subdir in MODELS:
        tag = name.split(":")[0].strip()
        if skip_map.get(tag, False):
            logger.info(f"⏭️  {name} 跳过（--skip-{tag.lower()}）")
            results.append(True)
            continue
        results.append(download_model(name, model_id, subdir))

    ok = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"  下载完成: {ok}/{total}")
    if ok == total:
        print(f"  ✅ 全部就绪，可运行 python local_models/test_downloader.py 检查")
    else:
        print(f"  ⚠️  部分失败，请检查网络后重试")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
