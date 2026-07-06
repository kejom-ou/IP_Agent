# -*- mode: python ; coding: utf-8 -*-
# ============================================================
# PyInstaller 打包配置 - 口播智能体桌面版
# 使用方式: pyinstaller desktop_app.spec
# 输出: dist/口播智能体/ 文件夹（含 口播智能体.exe）
# ============================================================

import sys
import os
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent

block_cipher = None

a = Analysis(
    ['desktop_app.py'],
    pathex=[str(ROOT), str(ROOT / 'local_models')],
    binaries=[],
    datas=[
        # 项目模块（必须打包）
        (str(ROOT / 'local_models' / 'pipeline_gradio.py'), 'local_models'),
        (str(ROOT / 'local_models' / 'modules.py'), 'local_models'),
        # 静态资源（如果有的话）
    ],
    hiddenimports=[
        # === PySide6 ===
        'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets',
        'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets',
        # === torch + CUDA ===
        'torch', 'torch.cuda', 'torch.amp', 'torchvision', 'torchaudio',
        'torch.backends.cudnn', 'torch._C',
        # === transformers / LLM ===
        'transformers', 'transformers.models.qwen2',
        'bitsandbytes', 'accelerate', 'sentencepiece', 'tiktoken',
        # === ASR ===
        'funasr', 'modelscope',
        # === TTS (CosyVoice) ===
        'soundfile', 'librosa', 'HyperPyYAML', 'inflect', 'conformer',
        'onnxruntime', 'pyworld', 'omegaconf', 'diffusers',
        # === 图像/视频 ===
        'cv2', 'face_alignment', 'pydub', 'skimage',
        'PIL', 'matplotlib',
        # === 其他 ===
        'numpy', 'requests', 'psutil',
        'yt_dlp', 'playwright.async_api',
        'uvicorn', 'fastapi',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除不需要的 torch 部分以减小体积
        'torch.distributed', 'torch.jit',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='口播智能体',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # 不显示控制台窗口（桌面版）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Windows 特定：设置 exe 图标
    icon=str(ROOT / 'local_models' / 'app_icon.ico') if (ROOT / 'local_models' / 'app_icon.ico').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='口播智能体',
)
