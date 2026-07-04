"""Wav2Lip 完整推理测试"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}, PyTorch: {torch.__version__}")

from local_models.lipsync import Wav2LipEngine

# 1. 转换音频到 16kHz（Wav2Lip 要求）
import librosa
import soundfile as sf

audio_in = r"e:\OwenSpace\IP_Agent\local_models\test_output.wav"
audio_16k = r"e:\OwenSpace\IP_Agent\local_models\test_output_16k.wav"

wav, sr = librosa.load(audio_in, sr=16000)
sf.write(audio_16k, wav, 16000)
print(f"Audio converted: {sr}Hz -> 16000Hz, {len(wav)/16000:.1f}s")

# 2. 加载 Wav2Lip
engine = Wav2LipEngine()
if not engine.load():
    print("[FAIL] Model load failed")
    sys.exit(1)

vram = torch.cuda.memory_allocated() / 1024**3
print(f"[OK] Model loaded, VRAM: {vram:.2f} GB")

# 3. 生成口型同步视频
video_path = r"e:\OwenSpace\IP_Agent\pretrained_models\MuseTalk\video.mp4"
output_path = r"e:\OwenSpace\IP_Agent\local_models\test_lipsync_output.mp4"

print(f"\nGenerating lip-sync...")
print(f"  Video: {video_path}")
print(f"  Audio: {audio_16k}")
print(f"  Output: {output_path}")

result = engine.generate(video_path, audio_16k, output_path)

# 4. 检查结果
engine.unload()

if result and os.path.exists(result):
    size_mb = os.path.getsize(result) / 1024**2
    print(f"\n[OK] Lip-sync video generated: {result}")
    print(f"[OK] Size: {size_mb:.1f} MB")
else:
    print(f"\n[FAIL] Generation failed")
