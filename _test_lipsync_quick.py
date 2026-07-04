"""Wav2Lip 快速测试：2秒视频 + TTS前段音频"""
import sys, os, logging, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lipsync_test")

import torch
import librosa
import soundfile as sf
import subprocess

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

OUT_DIR = r"e:\OwenSpace\IP_Agent\local_models\test_downloads"
VIDEO_IN = os.path.join(OUT_DIR, "clip_2s.mp4")
AUDIO_IN = os.path.join(OUT_DIR, "tts_clip.wav")
AUDIO_16K = os.path.join(OUT_DIR, "tts_clip_16k.wav")
OUTPUT = os.path.join(OUT_DIR, "lipsync_test.mp4")

# Step 1: 转换为 16kHz
logger.info("[1/3] 转换音频到 16kHz...")
wav, sr = librosa.load(AUDIO_IN, sr=16000)
sf.write(AUDIO_16K, wav, 16000)
logger.info(f"  音频: {sr}Hz → 16000Hz, {len(wav)/16000:.2f}s")

# Step 2: 加载模型
logger.info("[2/3] 加载 Wav2Lip 模型...")
t0 = time.time()
from local_models.lipsync import Wav2LipEngine
engine = Wav2LipEngine()
if not engine.load():
    print("[FAIL] 模型加载失败")
    sys.exit(1)
vram = torch.cuda.memory_allocated() / 1024**3
logger.info(f"  加载完成，显存: {vram:.2f} GB，耗时 {time.time()-t0:.1f}s")

# Step 3: 推理
logger.info("[3/3] Wav2Lip 推理中...")
t1 = time.time()
result = engine.generate(
    video_path=VIDEO_IN,
    audio_path=AUDIO_16K,
    output_path=OUTPUT,
    fps=25,
)
engine.unload()

if result and os.path.exists(result):
    size_mb = os.path.getsize(result) / 1024**2
    logger.info(f"\n{'='*50}")
    logger.info(f"✅ LipSync 完成！")
    logger.info(f"   输出: {result}")
    logger.info(f"   大小: {size_mb:.1f} MB")
    logger.info(f"   耗时: {time.time()-t1:.1f}s")
    logger.info(f"{'='*50}")
else:
    logger.error("[FAIL] 生成失败")
