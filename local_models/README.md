# 本地模型模块（local_models）

全本地化 AI 数字人管线：**ASR → LLM → TTS → LipSync**。TTS/LipSync 使用 ModelScope pipeline 加载，ASR 使用 FunASR AutoModel (SenseVoiceSmall)，LLM 使用 Transformers 原生推理（INT4 量化），所有模型从本地 `pretrained_models/` 目录加载，无需联网。**串行加载 8GB 显存适配**。

## 目录结构

```
IP_Agent/
├── pretrained_models/              # 模型存放目录（需自行准备）
│   ├── SenseVoiceSmall/             # ASR: SenseVoiceSmall（CPU）
│   ├── Qwen2.5-0.5B-Instruct/      # LLM: Qwen2.5-0.5B（GPU, INT4）
│   ├── CosyVoice-300M-SFT/         # TTS: CosyVoice Lite (~1-2GB 显存)
│   └── MuseTalk/                   # LipSync: MuseTalk（GPU）
└── local_models/
    ├── config.py                   # 模型配置（本地路径 + 设备分配）
    ├── asr_engine.py               # 语音识别（SenseVoiceSmall, CPU）
    ├── llm_engine.py               # 文案仿写（Qwen2.5-0.5B + Transformers, GPU）
    ├── tts_engine.py               # 语音合成（CosyVoice 纯本地推理, GPU）
    ├── lipsync.py                  # 口型合成（MuseTalk, GPU）
    ├── adapter.py                  # 与原 app.py 兼容的适配器
    ├── model_downloader.py         # 本地模型状态检查
    ├── pipeline_gradio.py          # Gradio Web 界面
    ├── publisher.py                # 抖音自动发布（Playwright + CDP）
    └── test_*.py                   # 单模块测试脚本
```

## 设备分配

| 环节 | 模型 | 设备 | 原因 |
|------|------|------|------|
| ASR | SenseVoiceSmall | **CPU** | 与 GPU 模型无冲突，CPU 足够处理 |
| LLM | Qwen2.5-0.5B-Instruct (INT4) | **GPU** | INT4 量化仅 ~0.5GB，用后卸载 |
| TTS | CosyVoice-300M-SFT (Lite) | **GPU** | ~1-2GB，LLM 卸载后加载 |
| LipSync | MuseTalk | **GPU** | ~6GB，TTS 卸载后加载 |
| **峰值** | — | **GPU** | **~6 GB**（串行，适配 8GB 显卡） |

### 资源占用

| 模型 | 磁盘 | 参数量 | CPU 内存 | GPU 显存 | 推理速度 |
|------|------|--------|----------|----------|----------|
| SenseVoiceSmall | ~0.9 GB | 234M | ~2 GB | — | 170x 实时（10s 音频 → 70ms） |
| Qwen2.5-0.5B-Instruct (INT4) | ~1.0 GB | 494M | — | ~0.5 GB | 取决于生成长度 |
| CosyVoice-300M-SFT (Lite) | ~5.5 GB | 300M | — | ~1-2 GB (fp16) | 接近实时 |
| MuseTalk | ~6.0 GB | — | — | ~6 GB（单张人脸） | 30+ fps (V100) |
| **总计（串行）** | **~13.4 GB** | **~1B** | **~2 GB** | **~6 GB 峰值** ✅ | — |

> **8GB 显存策略**：串行加载/卸载 — ASR(CPU) → LLM INT4(~0.5GB) → 卸载 → TTS Lite(~1-2GB) → 卸载 → LipSync(~6GB)。

## 模型说明

| 环节 | 模型 | 本地路径 | 设备 | 必需 |
|------|------|----------|------|------|
| ASR | SenseVoiceSmall | `pretrained_models/SenseVoiceSmall/` | CPU | ✅ |
| LLM | Qwen2.5-0.5B-Instruct | `pretrained_models/Qwen2.5-0.5B-Instruct/` | GPU | ✅ |
| TTS | CosyVoice-300M-SFT (Lite) | `pretrained_models/CosyVoice-300M-SFT/` | GPU | ✅ |
| LipSync | MuseTalk | `pretrained_models/MuseTalk/` | GPU | 可选 |

> ASR 使用 FunASR AutoModel (SenseVoiceSmall) 加载；TTS/LipSync 使用 ModelScope pipeline 加载；LLM 使用 Transformers 原生推理。

## 快速开始

### 1. 准备本地模型

将所有模型文件放到 `pretrained_models/` 对应子目录下（参考上方目录结构）。

### 2. 检查模型状态

```bash
python local_models/test_downloader.py
```

### 3. 启动 Gradio 界面

```bash
python local_models/pipeline_gradio.py
# 访问 http://localhost:7861
```

## 单模块测试

```bash
python local_models/test_asr.py      # 语音识别测试
python local_models/test_llm.py      # LLM 仿写测试
python local_models/test_tts.py      # 语音合成测试（纯本地推理）
python local_models/test_lipsync.py  # 口型合成测试
python local_models/test_pipeline.py --full  # 全流程测试
```

## 抖音自动发布

`local_models/publisher.py` — 基于 Playwright + Chrome DevTools Protocol 实现。

### 前置条件

```bash
pip install playwright
playwright install chromium
```

### 启动 Chrome（调试模式）

```bash
# macOS
open -a "Google Chrome" --args --remote-debugging-port=9222

# Windows
chrome.exe --remote-debugging-port=9222
```

### 使用

```python
from local_models.publisher import auto_publishing_videos_DY

result = auto_publishing_videos_DY(
    video_path="/path/to/video.mp4",
    title="视频标题",
    pulish_with_cover=False,
)
print(result)  # "✅ 抖音发布成功"
```

## 关键设计

- **LLM INT4 量化**（bitsandbytes NF4）— Qwen2.5-0.5B 仅 ~0.5GB 显存
- **串行加载/卸载** — ASR(CPU) → LLM → 卸载 → TTS → 卸载 → LipSync，峰值 ~6GB，适配 8GB 显卡
- **TTS 使用 fp16 dtype** — CosyVoice Lite 以 `torch_dtype=torch.float16` 加载，显存降至 ~1-2GB
- **所有模型从本地 `pretrained_models/` 加载**，`config.py` 统一管理路径
- **`local_files_only=True`** — LLM/ASR 拒绝联网下载
- **ASR 固定在 CPU**，GPU 模型串行使用
