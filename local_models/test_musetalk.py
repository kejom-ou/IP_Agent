"""
MuseTalk 口型合成 - 独立测试脚本
================================
用法:
    python test_musetalk.py                           # 使用内置测试数据
    python test_musetalk.py --video my_video.mp4      # 指定视频
    python test_musetalk.py --video v.mp4 --audio a.wav  # 指定视频+音频
    python test_musetalk.py --video v.mp4 --audio a.wav -o output.mp4 --bbox 5

依赖:
    - conda activate ip_agent_gpu (需要 torch + CUDA)
    - MuseTalk_repo/ 已 clone
    - pretrained_models/MuseTalk/ 模型已下载
    - ffmpeg 在 PATH 中
"""
import os
import sys
import time
import argparse
import logging
import shutil
import subprocess
from pathlib import Path

# ── 路径设置 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 日志 ──
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("MuseTalk-Test")


# ============================================================
# 环境检查
# ============================================================
def check_environment():
    """检查运行环境是否就绪"""
    errors = []
    warnings = []

    # 1. Python
    logger.info(f"Python: {sys.version}")

    # 2. CUDA / torch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({vram_total:.1f} GB)")
        else:
            errors.append("CUDA 不可用，MuseTalk 需要 GPU 运行")
    except ImportError:
        errors.append("torch 未安装")

    # 3. MuseTalk 源码
    musetalk_root = PROJECT_ROOT / "MuseTalk_repo"
    if not musetalk_root.exists() or not list(musetalk_root.glob("musetalk/models/*.py")):
        errors.append(
            f"MuseTalk 源码缺失: {musetalk_root}\n"
            f"  请执行: cd {PROJECT_ROOT} && git clone https://github.com/TMElyralab/MuseTalk.git MuseTalk_repo"
        )
    else:
        logger.info(f"MuseTalk 源码: {musetalk_root}")

    # 4. 模型权重
    models_dir = PROJECT_ROOT / "pretrained_models" / "MuseTalk"
    required_models = {
        "UNet v1.0": models_dir / "musetalk" / "pytorch_model.bin",
        "UNet v1.5 (可选)": models_dir / "musetalkV15" / "unet.pth",
        "VAE": models_dir / "sd-vae-ft-mse" / "diffusion_pytorch_model.bin",
        "FaceParsing": models_dir / "face-parse-bisent" / "79999_iter.pth",
    }

    model_ok = False
    for name, path in required_models.items():
        if path.exists():
            size_gb = path.stat().st_size / 1024**3
            logger.info(f"  ✅ {name}: {size_gb:.1f} GB")
            model_ok = True
        else:
            if "(可选)" not in name:
                errors.append(f"模型缺失: {name} ({path})")

    if not model_ok:
        errors.append(f"未找到任何 MuseTalk 模型权重: {models_dir}")

    # 5. ffmpeg
    if shutil.which("ffmpeg") is None:
        errors.append("ffmpeg 未找到，请添加到 PATH")
    else:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        ver_line = result.stdout.split('\n')[0] if result.stdout else "unknown"
        logger.info(f"ffmpeg: {ver_line}")

    # 6. CV2
    try:
        import cv2
        logger.info(f"OpenCV: {cv2.__version__}")
    except ImportError:
        errors.append("opencv-python 未安装")

    # 7. 检查测试数据
    test_video = models_dir / "video.mp4"
    test_audio = models_dir / "ash.wav"
    if test_video.exists():
        logger.info(f"测试视频: {test_video} ({test_video.stat().st_size / 1024**2:.1f} MB)")
    else:
        warnings.append(f"内置测试视频缺失: {test_video}")
    if test_audio.exists():
        logger.info(f"测试音频: {test_audio} ({test_audio.stat().st_size / 1024**2:.1f} MB)")
    else:
        warnings.append(f"内置测试音频缺失: {test_audio}")

    # 汇总
    if errors:
        logger.error("=" * 60)
        logger.error("环境检查失败！")
        for e in errors:
            logger.error(f"  ❌ {e}")
        logger.error("=" * 60)
        return False

    if warnings:
        logger.warning("=" * 60)
        for w in warnings:
            logger.warning(f"  ⚠️  {w}")
        logger.warning("=" * 60)

    logger.info("✅ 环境检查通过")
    return True


# ============================================================
# 主测试逻辑
# ============================================================
def create_test_video_from_image(image_path: str, audio_path: str, output_path: str) -> str:
    """
    如果是图片格式，先用 ffmpeg 转成动态视频。
    读取音频时长，生成匹配长度的视频。
    """
    import subprocess

    # 1. 获取音频时长
    cmd_dur = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        audio_path,
    ]
    result = subprocess.run(cmd_dur, capture_output=True, text=True)
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        logger.warning(f"无法获取音频时长，使用默认 10s")
        duration = 10.0

    logger.info(f"音频时长: {duration:.1f}s，生成对应长度视频...")

    # 2. 图片 → 动态视频 (zoompan 缓动效果)
    total_frames = max(int(duration * 25), 1)

    cmd = [
        "ffmpeg", "-y", "-v", "warning",
        "-loop", "1",
        "-i", image_path,
        "-vf",
        f"scale=720:1280:force_original_aspect_ratio=decrease,"
        f"pad=720:1280:(ow-iw)/2:(oh-ih)/2,"
        f"zoompan=z='1.0+0.01*sin(on*0.05)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={total_frames}:s=720x1280",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", "25",
        "-t", str(duration),
        output_path,
    ]
    subprocess.run(cmd, check=True)

    logger.info(f"图片转视频完成: {output_path}")
    return output_path


def run_musetalk(
    video_path: str,
    audio_path: str,
    output_path: str = None,
    bbox_shift: int = 0,
):
    """
    运行 MuseTalk 并打印详细统计
    """
    from local_models.musetalk_engine import MuseTalkEngine

    # ── 处理输入（图片自动转视频） ──
    ext = Path(video_path).suffix.lower()
    if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
        logger.info(f"检测到图片输入 ({ext})，自动转为视频...")
        temp_vid = str(Path(video_path).parent / f"_musetalk_temp_{Path(video_path).stem}.mp4")
        video_path = create_test_video_from_image(video_path, audio_path, temp_vid)

    # ── 输出路径 ──
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_musetalk_test.mp4")

    logger.info("=" * 60)
    logger.info("MuseTalk 口型合成测试")
    logger.info(f"  输入视频: {video_path}")
    logger.info(f"  输入音频: {audio_path}")
    logger.info(f"  输出路径: {output_path}")
    logger.info(f"  bbox_shift: {bbox_shift}")
    logger.info("=" * 60)

    # ── 加载模型 ──
    logger.info("[1/3] 加载 MuseTalk 模型...")
    t_load = time.time()

    engine = MuseTalkEngine()
    if not engine.load():
        logger.error("模型加载失败！")
        return None

    load_time = time.time() - t_load
    vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    logger.info(f"  ✅ 加载完成 ({load_time:.1f}s, 显存: {vram:.2f} GB)")

    # ── 推理 ──
    logger.info("[2/3] 口型合成推理...")
    t_infer = time.time()

    result = engine.generate(
        video_path=video_path,
        audio_path=audio_path,
        output_path=output_path,
        bbox_shift=bbox_shift,
    )

    infer_time = time.time() - t_infer

    if result is None:
        logger.error("推理失败！")
        engine.unload()
        return None

    size_mb = os.path.getsize(result) / 1024**2

    # ── 释放模型 ──
    logger.info("[3/3] 释放 GPU 资源...")
    engine.unload()

    # 强制清理 CUDA 显存
    try:
        import gc
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
    except Exception:
        pass

    # ── 汇总 ──
    logger.info("=" * 60)
    logger.info("✅ MuseTalk 测试完成！")
    logger.info(f"  输出文件: {result}")
    logger.info(f"  文件大小: {size_mb:.1f} MB")
    logger.info(f"  模型加载: {load_time:.1f}s")
    logger.info(f"  推理耗时: {infer_time:.1f}s")
    logger.info(f"  总耗时:   {load_time + infer_time:.1f}s")
    logger.info("=" * 60)

    return result


# ============================================================
# CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="MuseTalk 口型合成 - 独立测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python test_musetalk.py
    python test_musetalk.py --video my_avatar.mp4 --audio my_tts.wav
    python test_musetalk.py --video avatar.png --audio speech.wav -o result.mp4
        """,
    )
    parser.add_argument(
        "--video", "-v",
        default=None,
        help="输入视频/图片路径（默认使用 pretrained_models/MuseTalk/video.mp4）",
    )
    parser.add_argument(
        "--audio", "-a",
        default=None,
        help="输入音频路径（默认使用 pretrained_models/MuseTalk/ash.wav）",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出视频路径（默认: {input_stem}_musetalk_test.mp4）",
    )
    parser.add_argument(
        "--bbox", "-b",
        type=int,
        default=0,
        help="bbox_shift (正数=增大张嘴幅度, 默认 0)",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="跳过环境检查",
    )

    args = parser.parse_args()

    # ── 默认测试数据 ──
    models_dir = PROJECT_ROOT / "pretrained_models" / "MuseTalk"
    default_video = models_dir / "video.mp4"
    default_audio = models_dir / "ash.wav"

    video = args.video or (str(default_video) if default_video.exists() else None)
    audio = args.audio or (str(default_audio) if default_audio.exists() else None)

    if not video:
        logger.error("未指定输入视频，且默认测试视频不存在")
        logger.error(f"默认路径: {default_video}")
        sys.exit(1)
    if not audio:
        logger.error("未指定输入音频，且默认测试音频不存在")
        logger.error(f"默认路径: {default_audio}")
        sys.exit(1)

    if not os.path.exists(video):
        logger.error(f"视频文件不存在: {video}")
        sys.exit(1)
    if not os.path.exists(audio):
        logger.error(f"音频文件不存在: {audio}")
        sys.exit(1)

    # ── 环境检查 ──
    if not args.skip_check:
        if not check_environment():
            sys.exit(1)

    # ── 运行 ──
    import torch  # noqa: E402 (import after sys.path setup)
    result = run_musetalk(
        video_path=video,
        audio_path=audio,
        output_path=args.output,
        bbox_shift=args.bbox,
    )

    if result:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
