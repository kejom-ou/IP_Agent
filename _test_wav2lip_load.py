import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from local_models.lipsync import Wav2LipEngine

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

engine = Wav2LipEngine()
if engine.load():
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"\n[OK] Wav2Lip loaded! VRAM: {vram:.2f} GB")
    engine.unload()
    vram_after = torch.cuda.memory_allocated() / 1024**3
    print(f"[OK] After unload VRAM: {vram_after:.2f} GB")
else:
    print("\n[FAIL] Load failed")
