# 本地模型模块（local_models）

全本地化 AI 数字人管线：**ASR → LLM → TTS → LipSync**，所有模型从本地 `pretrained_models/` 目录加载，无需联网。

## 目录结构

```
IP_Agent/
├── pretrained_models/              # 模型存放目录（需自行准备）
│   ├── faster-whisper-small/       # ASR: faster-whisper-small（CPU）
│   ├── Qwen2.5-0.5B-Instruct/      # LLM: Qwen2.5-0.5B（GPU）
│   ├── CosyVoice-300M/             # TTS: CosyVoice-300M（GPU）
│   └── MuseTalk/                   # LipSync: MuseTalk（GPU）
└── local_models/
    ├── config.py                   # 模型配置（本地路径 + 设备分配）
    ├── asr_engine.py               # 语音识别（faster-whisper, CPU）
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
| ASR | faster-whisper-small | **CPU** | 避免与 LLM/TTS 抢显存，CPU 足够处理 |
| LLM | Qwen2.5-0.5B-Instruct | **GPU** | 推理需要 GPU 加速 |
| TTS | CosyVoice-300M | **GPU** | 语音合成需 GPU |
| LipSync | MuseTalk | **GPU** | 口型合成是显存/算力密集型 |

## 模型说明

| 环节 | 模型 | 本地路径 | 设备 | 必需 |
|------|------|----------|------|------|
| ASR | faster-whisper-small | `pretrained_models/faster-whisper-small/` | CPU | ✅ |
| LLM | Qwen2.5-0.5B-Instruct | `pretrained_models/Qwen2.5-0.5B-Instruct/` | GPU | ✅ |
| TTS | CosyVoice-300M | `pretrained_models/CosyVoice-300M/` | GPU | ✅ |
| LipSync | MuseTalk | `pretrained_models/MuseTalk/` | GPU | 可选 |

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
python local_models/test_tts.py      # 语音合成测试（SDK 模式）
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

- **所有模型从本地 `pretrained_models/` 加载**，`config.py` 统一管理路径
- **`local_files_only=True`** — LLM/ASR 拒绝联网下载
- **LLM 固定使用 Qwen2.5-0.5B**（最小参数量）
- **ASR 固定在 CPU**，LLM/TTS/LipSync 运行在 GPU
- **TTS 纯本地推理** — 直接从本地 CosyVoice 目录加载，无需 API 服务
- **抖音发布使用 Playwright + CDP** — 控制已登录 Chrome 自动发布
