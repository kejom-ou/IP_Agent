# 本地模型改造模块（local_models）

## 概述

本目录包含旗博士 AI 数字人项目的 **全本地化改造方案**，将原有依赖云端 API 的模型全部替换为可在本地运行的开源模型。

## 改造目标

| 原方案 | 问题 | 改造后 |
|--------|------|--------|
| DeepSeek 671B API | 需联网、需付费 API Key | Qwen2.5-3B 本地运行 |
| CosyVoice（云服务） | 依赖外部服务 | CosyVoice-300M 本地 |
| Whisper（云 ASR） | 需联网 | faster-whisper 本地 |
| TuiliONNX | 闭源、资源占用大 | MuseTalk/Wav2Lip 开源 |

## 目录结构

```
local_models/
├── __init__.py           # 模块入口
├── config.py             # 配置中心（显存检测、模型分级、自动降级）
├── adapter.py            # 统一适配器（桥接原 app.py 接口）
├── asr_engine.py         # 语音识别引擎（faster-whisper）
├── llm_engine.py         # 文案仿写引擎（Qwen2.5 + Ollama）
├── tts_engine.py         # 语音合成引擎（CosyVoice-300M）
├── lipsync.py            # 口型合成引擎（MuseTalk/Wav2Lip）
├── model_downloader.py   # 一键模型下载器
├── requirements.txt      # Python 依赖
└── README.md             # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r local_models/requirements.txt
```

### 2. 安装 Ollama（推荐）

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# 拉取 Qwen 模型
ollama pull qwen2.5:3b
```

### 3. 下载模型

```bash
# 查看模型状态
python local_models/model_downloader.py --check

# 下载必需模型
python local_models/model_downloader.py --essentials

# 下载全部模型
python local_models/model_downloader.py --all
```

### 4. 检查环境

```bash
python local_models/adapter.py
```

输出示例：
```
============================================================
🎯 本地模型适配方案 — GPU 显存：8 GB
============================================================
环节       模型                          显存(G)  量化
------------------------------------------------------------
语音识别    Whisper-ASR                   1.5      int8
文案仿写    Qwen2.5-3B-Instruct           3.5      int4
语音合成    CosyVoice-300M                2.0      fp16
口型合成    MuseTalk                      4.0      fp16
------------------------------------------------------------
合计（串行加载） 峰值约                    11.0
⚠️  所有模型分时加载，实际峰值 ≈ 4.0 GB
============================================================
```

## 显存分级策略

| 显卡 | ASR | LLM | TTS | 口型合成 |
|------|-----|-----|-----|---------|
| RTX 5060 8GB | Whisper small | Qwen2.5-3B (int4) | CosyVoice-300M | **MuseTalk** ⭐ |
| RTX 3060 6GB | Whisper small | Qwen2.5-1.5B (int4) | CosyVoice-300M | Wav2Lip |
| GTX 1050 4GB | Whisper tiny | Qwen2.5-0.5B (int4) | CosyVoice-300M | Wav2Lip Lite |
| CPU only | Whisper tiny | Qwen2.5-1.5B (GGUF) | CosyVoice-300M | ❌ 不可用 |

> **核心设计**：所有模型 **分时加载**，用完后立即释放显存。实际峰值显存 = 单个最大模型的显存（约 4GB）。

## 口型合成方案对比

| 方案 | 显存 | 画质 | 速度 | 适用 |
|------|------|------|------|------|
| **MuseTalk** ⭐ | ~4 GB | ⭐⭐⭐⭐⭐ | 30fps+ | 8GB+ 显卡 |
| Wav2Lip | ~2.5 GB | ⭐⭐⭐⭐ | ~15fps | 6GB+ 显卡 |
| Wav2Lip Lite | ~1.5 GB | ⭐⭐⭐ | ~10fps | 4GB+ 显卡 |

**选择 MuseTalk 的理由**：
- 腾讯音乐 Lyra Lab 开源，社区活跃
- 在潜在空间中生成，口型精度更高
- 官方实测 RTX 3050 Ti 4GB 可运行
- 支持中文/英文/日语多语言
- 实时推理（30fps+），画质优异

## 在 app.py 中启用

只需修改 import 语句即可切换：

```python
# 原版（云端 API）
# from utils.video_processor import download_and_extract_text
# from ai_processing.text_rewriter import execute_rewrite, AI_write_descriptions
# from video_tools.generate_video import generate_tuilionnx_video
# from utils.voice_processor import handle_audio_creation, get_pt_files, ...

# 本地版
from local_models.adapter import (
    download_and_extract_text,
    execute_rewrite,
    AI_write_descriptions,
    generate_tuilionnx_video,
    handle_audio_creation,
    get_pt_files,
    generate_audio_only,
    run_GPTvoice_command,
)
```

接口参数完全兼容，无需修改任何业务逻辑代码。

## 各模型说明

### 语音识别 — faster-whisper

- **模型**: Systran/faster-whisper-small
- **显存**: ~1.5 GB (int8)
- **语言**: 支持 99 种语言，中文准确率 > 95%
- **速度**: 比原版 Whisper 快 4 倍，显存省 38%
- **功能**: 语音转文字 + 带时间戳的字幕生成

### 文案仿写 — Qwen2.5

- **模型**: Qwen2.5-3B-Instruct（阿里通义千问）
- **显存**: ~3.5 GB (int4 量化)
- **推理方式**: Ollama / llama.cpp
- **功能**: AI 自动仿写、按指令仿写、视频描述生成

### 语音合成 — CosyVoice-300M

- **模型**: 阿里通义实验室 CosyVoice-300M-SFT
- **显存**: ~2.0 GB
- **功能**: 文字转语音 + 3 秒零样本声音克隆
- **语言**: 中/英/日/粤/韩
- **特色**: 情感控制、流式合成

### 口型合成 — MuseTalk

- **模型**: 腾讯音乐 Lyra Lab MuseTalk
- **显存**: ~4.0 GB (fp16)
- **功能**: 音频驱动的口型同步
- **速度**: 实时推理 (30fps+)
- **特点**: 潜在空间修复技术，高保真画质
