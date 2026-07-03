"""
============================================================
本地化数字人一键管线（pipeline_gradio.py）
============================================================
提供 Gradio Web 界面，串联完整的 AI 数字人工作流：

  ① ASR（语音识别）→ ② LLM（文案仿写）→ ③ TTS（语音合成）→ ④ LipSync（口型合成）

使用方法:
    python local_models/pipeline_gradio.py

8GB 显存适配策略（串行加载）：
  - ASR 固定在 CPU（~2GB 内存）
  - LLM INT4 量化（~0.5GB 显存），用后卸载
  - TTS Lite（~1-2GB 显存），用后卸载
  - LipSync（~6GB 显存），最后加载
  → 峰值显存 ~6GB ✅
============================================================
"""

import os
import sys
import time
import tempfile
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

# 确保项目根目录在 path 中
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("pipeline")


# ===========================================================================
# 单例引擎管理（8GB 显存：串行加载/卸载）
# ===========================================================================

_asr_engine = None
_llm_engine = None
_tts_engine = None
_lipsync_engine = None


def _get_asr():
    global _asr_engine
    if _asr_engine is None:
        from local_models.asr_engine import ASREngine
        _asr_engine = ASREngine()
        _asr_engine.load()
    return _asr_engine


def _get_llm():
    global _llm_engine
    if _llm_engine is None:
        from local_models.llm_engine import LocalLLMEngine
        _llm_engine = LocalLLMEngine()
        _llm_engine.init()
    return _llm_engine


def _unload_llm():
    """卸载 LLM，释放 ~0.5GB 显存给 TTS"""
    global _llm_engine
    if _llm_engine:
        _llm_engine.unload()
        _llm_engine = None
        logger.info("[显存管理] LLM 已卸载")


def _get_tts():
    global _tts_engine
    if _tts_engine is None:
        from local_models.tts_engine import CosyVoiceEngine
        _tts_engine = CosyVoiceEngine()
        _tts_engine.load_model()
    return _tts_engine


def _unload_tts():
    """卸载 TTS，释放 ~1-2GB 显存给 LipSync"""
    global _tts_engine
    if _tts_engine:
        _tts_engine.unload()
        _tts_engine = None
        logger.info("[显存管理] TTS 已卸载")


def _get_lipsync():
    global _lipsync_engine
    if _lipsync_engine is None:
        from local_models.lipsync import MuseTalkEngine
        _lipsync_engine = MuseTalkEngine()
        _lipsync_engine.load()
    return _lipsync_engine


def _unload_lipsync():
    """卸载 LipSync"""
    global _lipsync_engine
    if _lipsync_engine:
        _lipsync_engine.unload()
        _lipsync_engine = None
        logger.info("[显存管理] LipSync 已卸载")


# ===========================================================================
# 步骤 1：语音识别（ASR）
# ===========================================================================

def step1_extract_text(file_obj) -> Tuple[str, str]:
    """
    从上传的视频中提取文案。
    支持两种输入：
      - 视频文件（.mp4/.mov 等）
      - 音频文件（.wav/.mp3/.m4a 等）
    """
    if file_obj is None:
        return "", "❌ 请先上传视频或音频文件"

    file_path = file_obj if isinstance(file_obj, str) else file_obj.name
    logger.info(f"[ASR] 输入文件: {file_path}")

    # 判断是否为纯音频（无视频轨道）
    is_audio = file_path.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg"))

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")

        if is_audio:
            # 直接转成 16kHz WAV
            try:
                subprocess.run([
                    "ffmpeg", "-i", file_path,
                    "-ar", "16000", "-ac", "1",
                    audio_path, "-y",
                ], check=True, capture_output=True, timeout=30)
            except subprocess.CalledProcessError as e:
                err = e.stderr.decode() if e.stderr else str(e)
                return "", f"❌ 音频转换失败: {err}"
            except FileNotFoundError:
                return "", "❌ FFmpeg 未安装 → https://ffmpeg.org/download.html"
        else:
            # 分离音频
            try:
                subprocess.run([
                    "ffmpeg", "-i", file_path,
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1",
                    audio_path, "-y",
                ], check=True, capture_output=True, timeout=30)
            except subprocess.CalledProcessError as e:
                err = e.stderr.decode() if e.stderr else str(e)
                return "", f"❌ 音频提取失败: {err}"
            except FileNotFoundError:
                return "", "❌ FFmpeg 未安装 → https://ffmpeg.org/download.html"

        # ASR 转写
        try:
            asr = _get_asr()
            text = asr.transcribe(audio_path)
            if text:
                logger.info(f"[ASR] 提取完成，{len(text)} 字符")
                return text, f"✅ 识别完成，共 {len(text)} 字符"
            else:
                return "", "❌ 未识别到语音内容"
        except Exception as e:
            logger.error(f"[ASR] 识别失败: {e}")
            return "", f"❌ 识别失败: {e}"


# ===========================================================================
# 步骤 2：文案仿写（LLM）
# ===========================================================================

def step2_rewrite(text: str, ai_mode: str, custom_prompt: str) -> Tuple[str, str]:
    """
    使用本地 Qwen2.5 模型对文案进行 AI 仿写（INT4 量化，~0.5GB 显存）。
    """
    if not text or not text.strip():
        return "", "❌ 请先提取文案"

    logger.info(f"[LLM] 开始仿写，模式: {ai_mode}")

    try:
        engine = _get_llm()
        prompt = custom_prompt if ai_mode == "根据指令仿写" else None
        result = engine.rewrite(
            original_text=text,
            mode=ai_mode,
            custom_prompt=prompt,
        )
        # 用后立即卸载，释放显存给 TTS
        _unload_llm()

        if result and result != text:
            return result, f"✅ 仿写完成，{len(result)} 字符"
        else:
            return text, "⚠️ 仿写未生效，使用原文"
    except Exception as e:
        _unload_llm()
        logger.error(f"[LLM] 仿写失败: {e}")
        return text, f"❌ 仿写失败: {e}"


# ===========================================================================
# 步骤 3：语音合成（TTS）
# ===========================================================================

def step3_generate_audio(text: str, speed: float) -> Tuple[Optional[str], str]:
    """
    使用 CosyVoice Lite 将文案合成为语音（~1-2GB 显存）。
    合成完成后自动卸载，释放显存给 LipSync。
    """
    if not text or not text.strip():
        return None, "❌ 文案为空"

    # 确保 LLM 已卸载，释放显存
    _unload_llm()

    logger.info(f"[TTS] 合成语音，语速={speed}")

    try:
        engine = _get_tts()
        audio_path = engine.synthesize(
            text=text,
            speaker="default",
            speed=speed,
        )
        # 用后立即卸载，释放显存给 LipSync
        _unload_tts()

        if audio_path and os.path.exists(audio_path):
            duration = _get_audio_duration(audio_path)
            return audio_path, f"✅ 语音合成完成（{duration:.1f}s）"
        else:
            return None, "❌ 语音合成失败"
    except Exception as e:
        _unload_tts()
        logger.error(f"[TTS] 合成失败: {e}")
        return None, f"❌ 合成失败: {e}"


def _get_audio_duration(path: str) -> float:
    """获取音频时长（秒）"""
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries",
             "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except:
        return 0.0


# ===========================================================================
# 步骤 4：口型合成（LipSync）
# ===========================================================================

def step4_lipsync(video_path, audio_path) -> Tuple[Optional[str], str]:
    """
    使用 MuseTalk 将音频驱动的口型同步到视频上（~6GB 显存）。
    """
    if video_path is None:
        return None, "❌ 缺少视频输入"
    if audio_path is None:
        return None, "❌ 缺少音频输入"

    # 确保 TTS 已卸载，释放显存
    _unload_tts()

    vp = video_path if isinstance(video_path, str) else video_path.name
    ap = audio_path if isinstance(audio_path, str) else audio_path.name

    logger.info(f"[LipSync] 视频={vp}, 音频={ap}")

    try:
        engine = _get_lipsync()

        output_path = os.path.join(
            tempfile.gettempdir(),
            f"lipsync_output_{int(time.time())}.mp4",
        )

        result = engine.generate(
            video_path=vp,
            audio_path=ap,
            output_path=output_path,
        )

        if result and os.path.exists(result):
            return result, f"✅ 口型合成完成"
        else:
            return None, "❌ 口型合成失败"
    except Exception as e:
        logger.error(f"[LipSync] 合成失败: {e}")
        return None, f"❌ 合成失败: {e}"


# ===========================================================================
# 一键全流程管线
# ===========================================================================

def run_full_pipeline(
    input_video,
    ai_mode: str,
    custom_prompt: str,
    speed: float,
    progress=gr.Progress(),
):
    """
    一键执行完整管线（串行加载/卸载，8GB 显存适配）：
      ASR(CPU) → LLMINT4(0.5GB)→卸载 → TTS(4GB)→卸载 → LipSync(6GB)
    """
    if input_video is None:
        return None, "", "", None, "❌ 请先上传视频"

    file_path = input_video if isinstance(input_video, str) else input_video.name

    progress(0.05, desc="步骤 1/4: 提取视频文案...")
    text, status1 = step1_extract_text(input_video)
    yield None, text, status1, None, ""
    if not text:
        return None, text, "❌ 管线中断：文案提取失败", None, status1

    progress(0.30, desc="步骤 2/4: AI 仿写文案 (INT4)...")
    rewritten, status2 = step2_rewrite(text, ai_mode, custom_prompt)
    yield None, rewritten, f"{status1}\n{status2}", None, ""
    if not rewritten:
        return None, text, "❌ 管线中断：仿写失败", None, f"{status1}\n{status2}"

    progress(0.55, desc="步骤 3/4: 合成语音（释放 LLM 显存）...")
    audio_path, status3 = step3_generate_audio(rewritten, speed)
    yield None, rewritten, f"{status1}\n{status2}\n{status3}", audio_path, ""
    if not audio_path:
        return None, rewritten, "❌ 管线中断：语音合成失败", None, f"{status1}\n{status2}\n{status3}"

    progress(0.75, desc="步骤 4/4: 生成口型同步视频（释放 TTS 显存）...")
    video_path, status4 = step4_lipsync(file_path, audio_path)
    yield video_path, rewritten, f"{status1}\n{status2}\n{status3}\n{status4}", audio_path, ""

    progress(1.0, desc="完成！")
    yield video_path, rewritten, f"{status1}\n{status2}\n{status3}\n{status4}", audio_path, "✅ 全流程完成！"


# ===========================================================================
# 环境检查
# ===========================================================================

def check_environment() -> str:
    """检查本地运行环境"""
    lines = ["### 🖥️ 环境检查"]

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
            lines.append(f"- GPU: ✅ {torch.cuda.get_device_name(0)} ({vram} GB)")
        else:
            lines.append("- GPU: ⚠️ 未检测到 CUDA GPU，将使用 CPU")
    except ImportError:
        lines.append("- GPU: ❌ PyTorch 未安装")

    # ASR (SenseVoiceSmall)
    try:
        import funasr
        from local_models.config import ASR_CONFIG
        asr_ok = Path(ASR_CONFIG["local_path"]).is_dir()
        lines.append(f"- ASR (SenseVoiceSmall): {'✅ 模型就绪' if asr_ok else '❌ 缺失 → ' + ASR_CONFIG['local_path']}")
    except ImportError:
        lines.append("- ASR (SenseVoiceSmall): ❌ funasr 未安装 (pip install funasr)")

    # LLM
    try:
        import transformers
        from local_models.config import get_llm_model
        llm_cfg = get_llm_model()
        llm_ok = Path(llm_cfg["local_path"]).is_dir()
        lines.append(f"- LLM ({llm_cfg['name']}): {'✅ 模型就绪' if llm_ok else '❌ 缺失 → ' + llm_cfg['local_path']}")
    except ImportError:
        lines.append("- LLM (Transformers): ❌ 未安装 → pip install transformers")

    # TTS
    from local_models.config import TTS_CONFIG
    tts_path = Path(TTS_CONFIG["local_path"])
    if tts_path.is_dir():
        lines.append(f"- TTS (CosyVoice): ✅ 本地模型就绪 ({tts_path})")
    else:
        lines.append(f"- TTS (CosyVoice): ❌ 未找到 → {tts_path}")

    # LipSync
    try:
        import modelscope
        from local_models.config import LIPSYNC_CONFIG
        lip_ok = Path(LIPSYNC_CONFIG["local_path"]).is_dir()
        lines.append(f"- LipSync (MuseTalk): {'✅ 模型就绪' if lip_ok else '❌ 缺失 → ' + LIPSYNC_CONFIG['local_path']}")
    except ImportError:
        lines.append("- LipSync (MuseTalk): ❌ 未安装 (pip install modelscope)")

    # FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        lines.append("- FFmpeg: ✅")
    except:
        lines.append("- FFmpeg: ❌ 未安装 → https://ffmpeg.org/download.html")

    return "\n".join(lines)


def refresh_env():
    """刷新环境检查"""
    return check_environment()


# ===========================================================================
# Gradio UI
# ===========================================================================

def create_pipeline_ui():
    """创建 Gradio 管线界面"""

    CSS = """
    .step-header { font-size: 1.1em; font-weight: bold; margin-bottom: 8px; }
    .pipeline-status { font-size: 0.9em; }
    #full-run-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    """

    with gr.Blocks(title="旗博士 AI 追爆管线", css=CSS, theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # 🎬 旗博士 AI 追爆管线
        **全本地化方案** | ASR → LLM(INT4) → TTS → LipSync | 8GB 显存友好
        """)

        # ---- 环境状态 ----
        with gr.Accordion("🖥️ 环境状态", open=False):
            env_output = gr.Markdown(check_environment())
            refresh_env_btn = gr.Button("重新检查", size="sm")
            refresh_env_btn.click(fn=refresh_env, outputs=[env_output])

        gr.Markdown("---")

        # ---- 输入区 ----
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown('<div class="step-header">📹 步骤一：上传视频</div>')
                input_video = gr.Video(label="上传口播视频", sources=["upload"])

            with gr.Column(scale=1):
                gr.Markdown('<div class="step-header">📝 步骤二：文案处理</div>')

                # 步骤 1: 提取文案
                extract_btn = gr.Button("1️⃣ 提取视频文案", variant="secondary")
                original_text = gr.Textbox(
                    label="原始文案（可手动编辑）",
                    lines=6,
                    placeholder="自动识别后显示...",
                )
                asr_status = gr.Textbox(label="识别状态", interactive=False)

                # 步骤 2: AI 仿写
                gr.Markdown('<div class="step-header" style="margin-top:12px;">✍️ 步骤三：AI 仿写</div>')
                with gr.Row():
                    ai_mode = gr.Radio(
                        choices=["AI自动仿写", "根据指令仿写"],
                        value="AI自动仿写",
                        label="仿写模式",
                    )
                custom_prompt = gr.Textbox(
                    label="自定义仿写指令",
                    placeholder="例如：用幽默口吻改写，加入网络热梗",
                    lines=2,
                    visible=False,
                )
                rewrite_btn = gr.Button("2️⃣ 执行 AI 仿写", variant="secondary")
                rewritten_text = gr.Textbox(
                    label="仿写后文案",
                    lines=6,
                    placeholder="仿写结果...",
                )

                def toggle_prompt(mode):
                    return gr.update(visible=(mode == "根据指令仿写"))

                ai_mode.change(fn=toggle_prompt, inputs=[ai_mode], outputs=[custom_prompt])

        # ---- 输出区 ----
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown('<div class="step-header">🔊 步骤四：语音合成</div>')
                with gr.Row():
                    speed_slider = gr.Slider(
                        value=1.0, minimum=0.5, maximum=2.0, step=0.1,
                        label="语速调节",
                    )
                tts_btn = gr.Button("3️⃣ 生成语音", variant="secondary")
                audio_output = gr.Audio(label="合成音频", type="filepath")
                tts_status = gr.Textbox(label="合成状态", interactive=False)

            with gr.Column(scale=1):
                gr.Markdown('<div class="step-header">🎭 步骤五：口型合成</div>')
                lipsync_btn = gr.Button("4️⃣ 生成口型视频", variant="secondary")
                video_output = gr.Video(label="最终数字人视频")
                step_status = gr.Textbox(
                    label="状态日志", lines=3, interactive=False,
                    elem_classes=["pipeline-status"],
                )

        # ---- 一键全流程 ----
        gr.Markdown("---")
        with gr.Row():
            full_run_btn = gr.Button(
                "🚀 一键全流程执行",
                variant="primary",
                elem_id="full-run-btn",
                size="lg",
            )
        full_status = gr.Textbox(label="全流程状态", lines=3, interactive=False)

        # ===== 事件绑定 =====

        # 步骤 1
        extract_btn.click(
            fn=step1_extract_text,
            inputs=[input_video],
            outputs=[original_text, asr_status],
        )

        # 步骤 2
        rewrite_btn.click(
            fn=step2_rewrite,
            inputs=[original_text, ai_mode, custom_prompt],
            outputs=[rewritten_text, step_status],
        )

        # 步骤 3
        tts_btn.click(
            fn=step3_generate_audio,
            inputs=[rewritten_text, speed_slider],
            outputs=[audio_output, tts_status],
        )

        # 步骤 4
        lipsync_btn.click(
            fn=step4_lipsync,
            inputs=[input_video, audio_output],
            outputs=[video_output, step_status],
        )

        # 一键全流程
        full_run_btn.click(
            fn=run_full_pipeline,
            inputs=[input_video, ai_mode, custom_prompt, speed_slider],
            outputs=[video_output, rewritten_text, full_status, audio_output, step_status],
        )

    return demo


# ===========================================================================
# 启动入口
# ===========================================================================

if __name__ == "__main__":
    demo = create_pipeline_ui()
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
    )
