"""
完整 Wav2Lip 推理：用户图片动态视频 + TTS音频 → 口型同步视频
"""
import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import torch
import librosa
import soundfile as sf

OUT_DIR = r"e:\OwenSpace\IP_Agent\local_models\test_downloads"
VIDEO_IN = os.path.join(OUT_DIR, "img_animated.mp4")
AUDIO_IN = os.path.join(OUT_DIR, "tts_output.wav")
AUDIO_16K = os.path.join(OUT_DIR, "tts_16k.wav")
OUTPUT = os.path.join(OUT_DIR, "lipsync_full.mp4")

# Step 1: 转换音频 16kHz
print("[1/3] 转换音频到 16kHz...")
wav, sr = librosa.load(AUDIO_IN, sr=16000)
sf.write(AUDIO_16K, wav, 16000)
print(f"  TTS: {len(wav)/16000:.1f}s")

# Step 2: 加载 Wav2Lip
print("[2/3] 加载 Wav2Lip 模型...")
t0 = time.time()
from local_models.lipsync import Wav2LipEngine
engine = Wav2LipEngine()
if not engine.load():
    print("[FAIL] 模型加载失败")
    sys.exit(1)
vram = torch.cuda.memory_allocated() / 1024**3
print(f"  加载完成，显存: {vram:.2f} GB，耗时 {time.time()-t0:.1f}s")

# Step 3: 推理
print("[3/3] Wav2Lip 推理（3197 帧 2 分 7 秒视频）...")
print(f"  Video: {VIDEO_IN}")
print(f"  Audio: {AUDIO_16K}")
print(f"  Output: {OUTPUT}")
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
    print()
    print("="*50)
    print(f"  LipSync 完整视频生成完成")
    print(f"  Output: {result}")
    print(f"  Size:   {size_mb:.1f} MB")
    print(f"  推理耗时: {time.time()-t1:.1f}s")
    print("="*50)
else:
    print("[FAIL] 生成失败")
