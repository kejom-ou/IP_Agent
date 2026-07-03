"""
============================================================
端到端 Pipeline 测试（ASR → LLM → TTS → LipSync）— Windows
============================================================
用法:
    python local_models/test_pipeline.py                              # 环境检查
    python local_models/test_pipeline.py --full                       # 全流程
    python local_models/test_pipeline.py --text "文案" --video in.mp4  # 跳过 ASR
============================================================
"""

import os
import sys
import time
import argparse
import logging
import subprocess
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_pipeline")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# 环境检查
# ═══════════════════════════════════════════════════════════

def check_environment() -> dict:
    """检查全链路依赖"""
    env = {}

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3)
            env["GPU"] = f"✅ {torch.cuda.get_device_name(0)} ({vram} GB)"
        else:
            env["GPU"] = "⚠️  CPU 模式"
    except ImportError:
        env["GPU"] = "❌ PyTorch 未安装"

    # ASR (SenseVoiceSmall)
    try:
        import funasr
        env["ASR"] = f"✅ FunASR SenseVoiceSmall {funasr.__version__}"
    except ImportError:
        env["ASR"] = "❌ funasr 未安装 → pip install funasr"

    # LLM
    try:
        import transformers
        env["LLM"] = f"✅ Transformers {transformers.__version__}"
    except ImportError:
        env["LLM"] = "❌ Transformers 未安装 → pip install transformers"

    # TTS
    from local_models.config import TTS_CONFIG
    tts_path = TTS_CONFIG["local_path"]
    env["TTS"] = "✅ CosyVoice 本地模型" if os.path.isdir(tts_path) else f"❌ 模型缺失 → {tts_path}"

    # LipSync
    try:
        import modelscope
        env["LipSync"] = "✅ ModelScope"
    except ImportError:
        env["LipSync"] = "❌ ModelScope 未安装 → pip install modelscope"

    # FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        env["FFmpeg"] = "✅ FFmpeg"
    except Exception:
        env["FFmpeg"] = "❌ FFmpeg 未安装"

    return env


# ═══════════════════════════════════════════════════════════
# Pipeline 步骤
# ═══════════════════════════════════════════════════════════

def step_asr(video_path: str) -> str:
    """ASR：从视频提取音频并转写"""
    from local_models.asr_engine import ASREngine

    audio_path = tempfile.mktemp(suffix=".wav")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_path, "-y", "-loglevel", "error",
    ], check=True, timeout=30)

    asr = ASREngine()
    asr.load()
    text = asr.transcribe(audio_path)
    asr.unload()

    if not text:
        text = "欢迎收看今天的节目，记得点赞关注哦。"
        logger.warning("   (无声视频，使用默认文案)")
    return text


def step_llm(text: str) -> str:
    """LLM：文案仿写（INT4 量化）"""
    from local_models.llm_engine import LocalLLMEngine

    engine = LocalLLMEngine()
    if not engine.init():
        raise RuntimeError("LLM 模型加载失败")

    result = engine.rewrite(text, mode="AI自动仿写")
    # 立即卸载，释放显存给 TTS
    engine.unload()
    return result if result != text else text


def step_tts(text: str) -> str:
    """TTS：语音合成，返回音频文件路径"""
    from local_models.tts_engine import CosyVoiceEngine

    engine = CosyVoiceEngine()
    engine.load_model()
    output = tempfile.mktemp(suffix=".wav")
    result = engine.synthesize(text=text, speaker="default", speed=1.0, output_path=output)
    # 立即卸载，释放显存给 LipSync
    engine.unload()
    if not result:
        raise RuntimeError("CosyVoice 合成失败")
    return output


def step_lipsync(video_path: str, audio_path: str) -> str:
    """LipSync：口型合成，返回输出视频路径"""
    from local_models.lipsync import MuseTalkEngine

    engine = MuseTalkEngine()
    if not engine.load():
        raise RuntimeError("MuseTalk 加载失败")

    output = tempfile.mktemp(suffix=".mp4")
    engine.generate(video_path=video_path, audio_path=audio_path, output_path=output, fps=25)
    engine.unload()
    return output


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def run_pipeline(video_path: str = None, text: str = None,
                 skip_asr=False, skip_llm=False, skip_tts=False, skip_lipsync=False):
    """运行全流程并打印每步耗时"""
    pipeline_start = time.time()
    steps = []

    # ── 生成默认视频 ──
    if not video_path and not skip_lipsync:
        logger.info("📹 生成测试视频...")
        video_path = os.path.join(tempfile.mkdtemp(), "test_input.mp4")
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "color=c=gray:s=512x512:d=3:r=25",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            video_path, "-y", "-loglevel", "error",
        ], check=True, timeout=30)

    # ── Step 1: ASR ──
    t0 = time.time()
    if not skip_asr and not text:
        logger.info("[1/4] 语音识别 (ASR)...")
        text = step_asr(video_path)
        steps.append(("ASR", time.time() - t0, True))
        logger.info(f"  ✅ ASR ({steps[-1][1]:.1f}s): {text[:80]}...")
    else:
        steps.append(("ASR", 0, True))
        logger.info("[1/4] ASR 跳过（使用用户文案）")

    # ── Step 2: LLM ──
    t0 = time.time()
    if not skip_llm:
        logger.info("[2/4] 文案仿写 (LLM)...")
        try:
            text = step_llm(text)
            steps.append(("LLM", time.time() - t0, True))
            logger.info(f"  ✅ LLM ({steps[-1][1]:.1f}s): {text[:80]}...")
        except Exception as e:
            steps.append(("LLM", time.time() - t0, False))
            logger.error(f"  ❌ LLM 失败: {e}")
    else:
        steps.append(("LLM", 0, True))
        logger.info("[2/4] LLM 跳过")

    # ── Step 3: TTS ──
    audio_path = None
    t0 = time.time()
    if not skip_tts:
        logger.info("[3/4] 语音合成 (TTS)...")
        try:
            audio_path = step_tts(text)
            steps.append(("TTS", time.time() - t0, True))
            logger.info(f"  ✅ TTS ({steps[-1][1]:.1f}s)")
        except Exception as e:
            steps.append(("TTS", time.time() - t0, False))
            logger.error(f"  ❌ TTS 失败: {e}")
    else:
        steps.append(("TTS", 0, True))
        logger.info("[3/4] TTS 跳过")

    # ── Step 4: LipSync ──
    t0 = time.time()
    if not skip_lipsync and video_path and audio_path:
        logger.info("[4/4] 口型合成 (LipSync)...")
        try:
            result = step_lipsync(video_path, audio_path)
            steps.append(("LipSync", time.time() - t0, True))
            logger.info(f"  ✅ LipSync ({steps[-1][1]:.1f}s): {result}")
        except Exception as e:
            steps.append(("LipSync", time.time() - t0, False))
            logger.error(f"  ❌ LipSync 失败: {e}")
    else:
        steps.append(("LipSync", 0, True))
        logger.info("[4/4] LipSync 跳过")

    # ── 汇总 ──
    total = time.time() - pipeline_start
    all_ok = all(s[2] for s in steps)

    print(f"\n{'='*50}")
    print(f"  {'步骤':<12} {'耗时':<12} {'状态'}")
    print(f"  {'-'*38}")
    for name, dur, ok in steps:
        icon = "✅" if ok else "❌"
        print(f"  {name:<12} {dur:.1f}s{'':>6} {icon}")
    print(f"  {'-'*38}")
    print(f"  总耗时:    {total:.1f}s")
    print(f"  结果:      {'✅ 全部通过' if all_ok else '⚠️  部分失败'}")
    print(f"{'='*50}\n")


# ═══════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="端到端 Pipeline 测试 (Windows)")
    parser.add_argument("--video", help="输入视频路径")
    parser.add_argument("--text", help="自定义文案（跳过 ASR）")
    parser.add_argument("--full", action="store_true", help="运行全流程")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-lipsync", action="store_true")
    args = parser.parse_args()

    # 环境检查
    env = check_environment()
    print(f"\n{'='*50}")
    print(f"  Pipeline 环境检查")
    print(f"{'='*50}")
    passed = 0
    for k, v in env.items():
        print(f"  {v}")
        if v.startswith("✅"):
            passed += 1
    print(f"{'='*50}")
    print(f"  就绪: {passed}/{len(env)}")
    print(f"{'='*50}\n")

    if args.full or args.video or args.text:
        run_pipeline(
            video_path=args.video,
            text=args.text,
            skip_asr=args.skip_asr or bool(args.text),
            skip_llm=args.skip_llm,
            skip_tts=args.skip_tts,
            skip_lipsync=args.skip_lipsync,
        )
    else:
        print("💡 使用 --full 运行全流程:")
        print("   python test_pipeline.py --full")
        print("   python test_pipeline.py --text '文案' --video in.mp4")


if __name__ == "__main__":
    main()
