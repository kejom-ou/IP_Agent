# IP Agent — 本地化 AI 数字人口播视频生成

全本地推理的 AI 数字人管线：**ASR（语音识别）→ LLM（文案仿写）→ TTS（语音合成）→ LipSync（口型合成）**。ASR 使用 FunASR AutoModel，LLM 使用 Transformers 原生推理（INT4 量化），TTS/LipSync 使用 ModelScope pipeline，所有模型从本地加载，无需联网。**8GB 显存友好**。

## 目录结构

```
IP_Agent/
├── pretrained_models/              # 模型存放目录（需自行准备）
│   ├── SenseVoiceSmall/             # ASR: SenseVoiceSmall
│   ├── Qwen2.5-0.5B-Instruct/      # LLM: Qwen2.5-0.5B (INT4)
│   ├── CosyVoice-300M-SFT/         # TTS: CosyVoice Lite (~1-2GB 显存)
│   └── MuseTalk/                   # LipSync: MuseTalk
└── local_models/
    ├── config.py                   # 模型路径 & 设备配置
    ├── asr_engine.py               # 语音识别（SenseVoiceSmall）
    ├── llm_engine.py               # 文案仿写（INT4 量化）
    ├── tts_engine.py               # 语音合成
    ├── lipsync.py                  # 口型合成
    ├── pipeline_gradio.py          # Gradio Web 界面（串行加载）
    ├── publisher.py                # 抖音自动发布
    └── test_*.py                   # 单模块测试
```

## 准备模型

### 一键下载（推荐）

```bash
pip install funasr modelscope
python local_models/download_models.py
```

自动将 ASR、LLM、TTS、LipSync 全部下载到 `pretrained_models/`。

### 手动准备

将模型放到 `pretrained_models/` 对应子目录：

| 模型 | 目录 | 设备 |
|------|------|------|
| SenseVoiceSmall | `pretrained_models/SenseVoiceSmall/` | CPU |
| Qwen2.5-0.5B-Instruct | `pretrained_models/Qwen2.5-0.5B-Instruct/` | GPU |
| CosyVoice-300M-SFT (Lite) | `pretrained_models/CosyVoice-300M-SFT/` | GPU |
| MuseTalk | `pretrained_models/MuseTalk/` | GPU |

### 资源占用

| 模型 | 磁盘 | 参数量 | CPU 内存 | GPU 显存 | 推理速度 |
|------|------|--------|----------|----------|----------|
| SenseVoiceSmall | ~0.9 GB | 234M | ~2 GB | — | 170x 实时（10s 音频 → 70ms） |
| Qwen2.5-0.5B-Instruct (INT4) | ~1.0 GB | 494M | — | ~0.5 GB | 取决于生成长度 |
| CosyVoice-300M-SFT (Lite) | ~0.3 GB | 300M | — | ~1-2 GB | 接近实时 |
| MuseTalk | ~6.0 GB | — | — | ~6 GB（单张人脸） | 30+ fps (V100) |
| **总计（串行）** | **~8.2 GB** | **~1B** | **~2 GB** | **~6 GB 峰值** ✅ | — |

> **8GB 显存适配策略**：串行加载/卸载 — ASR(CPU) → LLM INT4(~0.5GB) → 卸载 → TTS Lite(~1-2GB) → 卸载 → LipSync(~6GB)，峰值仅 ~6GB。

检查模型就绪状态：

```bash
python local_models/test_downloader.py
```

## 各模块用法

所有命令在项目根目录下执行。

### 1. ASR — 语音识别

```python
from local_models.asr_engine import ASREngine

asr = ASREngine()
asr.load()
text = asr.transcribe("audio.wav")
print(text)
asr.unload()
```

测试：

```bash
python local_models/test_asr.py
# 输入音频文件路径
```

### 2. LLM — 文案仿写

```python
from local_models.llm_engine import LocalLLMEngine

engine = LocalLLMEngine()
engine.init()
result = engine.rewrite("原始文案", mode="AI自动仿写")
print(result)
engine.unload()  # 释放 ~0.5GB 显存
```

测试：

```bash
python local_models/test_llm.py
# 输入文案，交互式仿写
```

### 3. TTS — 语音合成

```python
from local_models.tts_engine import CosyVoiceEngine

engine = CosyVoiceEngine()
engine.load_model()
result = engine.synthesize(
    text="你好，欢迎收看今天的节目",
    speaker="中文女",
    speed=1.0,
    output_path="output.wav",
)
engine.unload()  # 释放 ~4GB 显存
```

测试：

```bash
python local_models/test_tts.py
# 输入合成文本，生成 test_output.wav
```

### 4. LipSync — 口型合成

```python
from local_models.lipsync import MuseTalkEngine

engine = MuseTalkEngine()
engine.load()
engine.generate(
    video_path="input.mp4",   # 数字人视频
    audio_path="audio.wav",   # 音频
    output_path="output.mp4",
)
engine.unload()
```

测试：

```bash
python local_models/test_lipsync.py
# 输入口播视频路径和音频路径
```

## Pipeline 全流程

### Gradio Web 界面（推荐）

```bash
python local_models/pipeline_gradio.py
# 访问 http://localhost:7861
```

提供四个步骤的独立操作 + 一键全流程（**自动串行管理显存**）：
1. **ASR** — 上传视频，提取音频并转写
2. **LLM** — 输入文案，AI 仿写优化（INT4, ~0.5GB）
3. **TTS** — 文案合成语音（Lite 版 ~1-2GB，LLM 已卸载）
4. **LipSync** — 视频 + 音频合成数字人口播（~6GB，TTS 已卸载）

### 命令行全流程测试

```bash
# 环境检查
python local_models/test_pipeline.py

# 全流程（自动生成测试视频）
python local_models/test_pipeline.py --full

# 指定视频 + 自定义文案
python local_models/test_pipeline.py --video input.mp4 --text "今天给大家分享一个..."

# 跳过特定步骤
python local_models/test_pipeline.py --full --skip-llm --skip-lipsync
```

## 抖音自动发布

需先以调试模式启动 Chrome：

```bash
# macOS
open -a "Google Chrome" --args --remote-debugging-port=9222
```

```bash
pip install playwright
playwright install chromium
```

```python
from local_models.publisher import auto_publishing_videos_DY

result = auto_publishing_videos_DY(
    video_path="/path/to/video.mp4",
    title="视频标题",
    pulish_with_cover=False,
)
```

## 关键设计

- **LLM INT4 量化** — Qwen2.5-0.5B 仅需 ~0.5GB 显存（bitsandbytes 4-bit NF4）
- **串行加载/卸载** — ASR(CPU) → LLM → 卸载 → TTS → 卸载 → LipSync，峰值 ~6GB，适配 8GB 显卡
- **全本地推理** — 所有模型从 `pretrained_models/` 加载，不联网下载
- **模块解耦** — 每个模块可独立使用、独立测试
