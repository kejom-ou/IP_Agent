"""
============================================================
本地化数字人一键管线（pipeline_gradio.py）
============================================================
提供 Gradio Web 界面，串联完整的 AI 数字人工作流：

  ① ASR（语音识别）→ ② LLM（文案仿写）→ ③ TTS（语音合成）→ ④ LipSync（口型合成）

使用方法:
    python local_models/pipeline_gradio.py

8GB 显存适配策略（串行加载 + MuseTalk 前激进清理）：
  - ASR → CPU/GPU（~1-2GB），用后保持
  - LLM FP16（~1GB 显存），用后卸载
  - TTS Lite（~1-2GB 显存），用后卸载
  - LipSync MuseTalk（~6-8GB 显存），加载前卸载 ASR/LLM/TTS
  → 峰值显存 ~8GB ✅
============================================================
"""

import os
import re
import sys
import time
import uuid
import tempfile
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple

# ------------------------------------------------------------
# Windows: 切换到 SelectorEventLoop，抑制 Proactor socket 告警
# ------------------------------------------------------------
# Windows 默认 asyncio 事件循环是 ProactorEventLoop。
# 在 del PyTorch 模型 + gc.collect() 时，Proactor 会主动关闭底层 socket，
# 触发 ConnectionResetError (10054) 告警：
#   "Exception in callback _ProactorBasePipeTransport._call_connection_lost"
# 该错误发生在 Python 内部事件循环的回调中，用户代码无法捕获。
# 解法：在任何 asyncio / gradio 导入前，切换到 SelectorEventLoop。
# 实际功能完全正常（socket 已经关闭了），只是日志污染。
# ------------------------------------------------------------
if sys.platform == "win32":
    import asyncio as _asyncio
    try:
        _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

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


def _unload_asr():
    """卸载 ASR，释放 GPU 显存（ASR 可能占用 ~1-2GB）"""
    global _asr_engine
    if _asr_engine:
        _asr_engine.unload()
        _asr_engine = None
        logger.info("[显存管理] ASR 已卸载")
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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


def _unload_all_for_lipsync():
    """激进清理：加载 MuseTalk 前，强制卸载所有前序模型 + 清空 GPU 缓存
    
    MuseTalk 需要 ~6-8GB 独享显存。此函数会：
    1. 卸载 ASR / LLM / TTS（如果还在 GPU 上）
    2. 调用 gc.collect() 清理 Python 对象
    3. 调用 torch.cuda.empty_cache() 释放 PyTorch 缓存
    """
    import gc
    import torch
    logger.info("[显存管理] 🔄 加载 MuseTalk 前执行显存清理...")
    _unload_llm()
    _unload_tts()
    _unload_asr()
    gc.collect()
    if torch.cuda.is_available():
        before = torch.cuda.memory_allocated() / 1024**3
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"[显存管理] CUDA 缓存已清空，当前占用: {after:.2f} GB (清理前: {before:.2f} GB)")
    else:
        logger.info("[显存管理] 无 CUDA 设备，跳过 GPU 清理")


def _get_lipsync():
    """懒加载 LipSync 引擎（仅 MuseTalk，高质量 ~6-8GB 显存）
    
    加载前会先卸载所有其他 GPU 模型（ASR/LLM/TTS），确保 MuseTalk 有充足显存。
    """
    global _lipsync_engine
    if _lipsync_engine is None:
        # 激进释放所有 GPU 显存
        _unload_all_for_lipsync()
        from local_models.musetalk_engine import MuseTalkEngine
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

def step2_rewrite(text: str, ai_mode: str, custom_prompt: str) -> Tuple[str, str, str, str]:
    """
    使用本地 Qwen2.5 模型对文案进行 AI 仿写（INT4 量化，~0.5GB 显存）。
    返回: (正文, 标题, 标签, 状态)
    """
    if not text or not text.strip():
        return "", "", "", "❌ 请先提取文案"

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

        rewritten = result.get("text", text)
        title = result.get("title", "")
        tags = result.get("tags", "")

        if rewritten and rewritten != text:
            return rewritten, title, tags, f"✅ 仿写完成，{len(rewritten)} 字符"
        else:
            return text, "", "", "⚠️ 仿写未生效，使用原文"
    except Exception as e:
        _unload_llm()
        logger.error(f"[LLM] 仿写失败: {e}")
        return text, "", "", f"❌ 仿写失败: {e}"


# ===========================================================================
# 步骤 3：语音合成（TTS）
# ===========================================================================

def step3_generate_audio(
    text: str,
    speed: float,
    voice_mode: str,
    speaker: str,
    custom_audio: Optional[str],
    custom_audio_text: str,
) -> Tuple[Optional[str], str]:
    """
    使用 CosyVoice 将文案合成为语音（~1-2GB 显存）。

    Args:
        text: 待合成文案
        speed: 语速倍率
        voice_mode: 音色来源（"内置音色" / "上传音频" / "录制音频"）
        speaker: 内置音色名（voice_mode=内置音色 时使用）
        custom_audio: 用户上传/录制的参考音频路径
        custom_audio_text: 参考音频对应的文本（zero-shot 必需）
    """
    if not text or not text.strip():
        return None, "❌ 文案为空"

    # 确保 LLM 已卸载，释放显存
    _unload_llm()

    logger.info(f"[TTS] 合成语音，语速={speed}, 模式={voice_mode}")
    if voice_mode != "内置音色":
        logger.info(f"[TTS] 参考音频: {custom_audio}, 文本: {custom_audio_text[:30] if custom_audio_text else '(空)'}...")

    try:
        engine = _get_tts()

        if voice_mode == "内置音色":
            audio_path = engine.synthesize(
                text=text,
                speaker=speaker or "default",
                speed=speed,
            )
        else:
            # 上传/录制音频 → zero-shot 音色克隆
            if not custom_audio or not os.path.exists(custom_audio):
                return None, "❌ 请先上传或录制参考音频"
            # 自定义音色时强制卸载 TTS 模型（_save_with_prompt 内部若失败回退到 SFT，
            # 必须先有干净显存，否则会触发 CUDA 断言异常）
            _unload_tts()
            engine = _get_tts()
            # 兜底：如果用户没填参考音频文本，给一段默认示例，避免 zero-shot 抛异常
            prompt_text = (custom_audio_text or "").strip() or "这是一段示例音频，用于音色克隆。"
            try:
                audio_path = engine._save_with_prompt(
                    text=text,
                    prompt_audio_path=custom_audio,
                    prompt_text=prompt_text,
                    speed=speed,
                )
            except Exception as inner_e:
                _unload_tts()
                logger.error(f"[TTS] zero-shot 失败: {inner_e}", exc_info=True)
                return None, f"❌ 音色克隆失败：{inner_e}。请检查音频格式（WAV 16k 单声道最佳）或填写正确的参考文本。"

        # 用后立即卸载，释放显存给 LipSync
        _unload_tts()

        if audio_path and os.path.exists(audio_path):
            duration = _get_audio_duration(audio_path)
            return audio_path, f"✅ 语音合成完成（{duration:.1f}s）"
        else:
            return None, "❌ 语音合成失败"
    except Exception as e:
        _unload_tts()
        logger.error(f"[TTS] 合成失败: {e}", exc_info=True)
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

def step4_lipsync(avatar_video, audio_path, fallback_video=None) -> Tuple[Optional[str], str]:
    """
    使用 MuseTalk 将音频驱动的口型同步到指定形象视频上（~6-8GB 显存）。

    优先级：avatar_video（用户在步骤四上传的形象视频） > fallback_video（步骤一的输入视频）

    加载 MuseTalk 前会通过 _unload_all_for_lipsync() 卸载所有前序模型
    （ASR/LLM/TTS），确保 MuseTalk 有充足的 GPU 显存可用。
    """
    video_path = avatar_video or fallback_video
    if video_path is None:
        return None, "❌ 请上传数字人形象视频（或先在步骤一上传参考视频）"
    if audio_path is None:
        return None, "❌ 缺少音频输入"

    # 确保所有前序模型已卸载（TTS/LLM/ASR 全部移出 GPU）
    _unload_all_for_lipsync()

    vp = video_path if isinstance(video_path, str) else video_path.name
    ap = audio_path if isinstance(audio_path, str) else audio_path.name

    logger.info(f"[LipSync] 视频={vp}, 音频={ap}")

    # 图片 → 视频转换：如果用户上传的是静态图片，先用 FFmpeg 生成
    # 与音频等长的动态视频（zoompan 缓动），再送入 MuseTalk
    img_ext = vp.rsplit(".", 1)[-1].lower() if "." in vp else ""
    is_image = img_ext in ("png", "jpg", "jpeg", "bmp", "webp")
    img_video_temp = None  # 如果有临时视频，最后需要清理

    if is_image:
        logger.info(f"[LipSync] 检测到图片格式 ({img_ext})，先转为动态视频")
        duration = _get_audio_duration(ap)
        if duration <= 0:
            return None, "❌ 无法获取音频时长，图片转视频失败"
        fps = 25
        total_frames = max(int(duration * fps), fps)  # 至少 1 秒
        duration_s = total_frames / fps

        img_video_temp = os.path.join(
            tempfile.gettempdir(),
            f"img2vid_{uuid.uuid4().hex}.mp4",
        )
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", vp,
            "-vf",
            f"scale=2160:-1,"
            f"zoompan=z='min(zoom+0.0005,1.3)':"
            f"d={total_frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"s=720x1280:fps={fps}",
            "-t", str(duration_s),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            img_video_temp,
        ]
        logger.info(f"[LipSync] FFmpeg: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0 or not os.path.exists(img_video_temp):
            logger.error(f"[LipSync] 图片转视频失败: {proc.stderr[:500]}")
            return None, f"❌ 图片转视频失败：{proc.stderr[:200]}"
        logger.info(f"[LipSync] 图片→视频完成: {img_video_temp}")
        vp = img_video_temp

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
    finally:
        # 清理图片转视频的中间产物
        if img_video_temp and os.path.exists(img_video_temp):
            try:
                os.remove(img_video_temp)
            except Exception:
                pass


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
    一键执行完整管线（串行加载/卸载，8-10GB 显存适配）：
      ASR → LLM(INT4, 0.5GB) → 卸载 → TTS(1-2GB) → 卸载 → MuseTalk(6-8GB)
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
    rewritten, ai_title, ai_tags, status2 = step2_rewrite(text, ai_mode, custom_prompt)
    yield None, rewritten, f"{status1}\n{status2}", None, ""
    if not rewritten:
        return None, text, "❌ 管线中断：仿写失败", None, f"{status1}\n{status2}"

    progress(0.55, desc="步骤 3/4: 合成语音（释放 LLM 显存）...")
    audio_path, status3 = step3_generate_audio(
        text=rewritten, speed=speed,
        voice_mode="内置音色", speaker="中文女",
        custom_audio=None, custom_audio_text="",
    )
    yield None, rewritten, f"{status1}\n{status2}\n{status3}", audio_path, ""
    if not audio_path:
        return None, rewritten, "❌ 管线中断：语音合成失败", None, f"{status1}\n{status2}\n{status3}"

    progress(0.75, desc="步骤 4/4: 生成口型同步视频（释放 TTS 显存）...")
    video_path, status4 = step4_lipsync(None, audio_path, fallback_video=file_path)
    yield video_path, rewritten, f"{status1}\n{status2}\n{status3}\n{status4}", audio_path, ""

    progress(1.0, desc="完成！")
    yield video_path, rewritten, f"{status1}\n{status2}\n{status3}\n{status4}", audio_path, "✅ 全流程完成！"

    # 步骤 5: 自动发布到抖音（后台线程）
    publish_path, publish_msg = step5_publish(video_path, rewritten, ai_title, ai_tags)
    yield video_path, rewritten, f"{status1}\n{status2}\n{status3}\n{status4}\n{publish_msg}", audio_path, "✅ 全流程完成，已启动发布！"


# ===========================================================================
# 步骤 0：抖音链接下载
# ===========================================================================

def step0_download_douyin(douyin_url: str) -> Tuple[Optional[str], str]:
    """下载抖音视频，返回视频路径

    支持直接粘贴抖音分享文本（含中文标题与短链混合的内容），
    自动从中提取真实 https 链接。
    """
    if not douyin_url or not douyin_url.strip():
        return None, "请输入抖音链接"

    raw = douyin_url.strip()
    logger.info(f"[Download] 原始输入: {raw[:80]}...")

    try:
        from local_models.douyin_crawler import (
            DouyinCrawler,
            download_douyin_video,
        )

        # 从分享文本中提取真实链接（兼容 "3.00 :2pm y@G.ic ...  https://v.douyin.com/xxx/"）
        real_url = DouyinCrawler.extract_share_url(raw)
        if not real_url:
            # 兜底：用户可能直接粘贴了纯链接，做一次宽松校验
            if re.search(r"https?://", raw):
                real_url = re.search(r"https?://\S+", raw).group().rstrip(".,;!?，。；！？'\")]")
            else:
                return None, "❌ 输入内容中未找到抖音链接，请粘贴形如 https://v.douyin.com/xxxxx/ 的链接"

        logger.info(f"[Download] 解析出链接: {real_url}")

        output_dir = os.path.join(tempfile.gettempdir(), "ip_agent_downloads")
        os.makedirs(output_dir, exist_ok=True)

        # 每次都用 UUID 目录，避免被覆盖
        output_dir = os.path.join(output_dir, uuid.uuid4().hex)
        os.makedirs(output_dir, exist_ok=True)

        video_path = download_douyin_video(real_url, output_dir=output_dir)
        if video_path and os.path.exists(video_path):
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            logger.info(f"[Download] 完成: {video_path}")
            return video_path, f"✅ 下载完成 ({size_mb:.1f}MB): {os.path.basename(video_path)}"
        else:
            return None, "❌ 下载失败，请检查链接是否有效或网络是否通畅"
    except Exception as e:
        logger.error(f"[Download] 失败: {e}", exc_info=True)
        return None, f"❌ 下载失败: {e}"


# ===========================================================================
# 步骤 5：发布到抖音
# ===========================================================================

def step5_publish(video_path, rewritten_text: str, ai_title: str = "", ai_tags: str = "") -> Tuple[str, str]:
    """将最终视频发布到抖音创作者平台（后台线程，不阻塞 Gradio）。

    使用 publish_simple.py 的半自动模式：自动上传视频 + 填写标题/文案，
    用户在浏览器中手动点击「发布」按钮。
    """
    if video_path is None:
        return "", "❌ 请先生成口型视频（步骤四）"
    if not os.path.exists(video_path):
        return "", f"❌ 视频文件不存在: {video_path}"

    # 优先使用 AI 生成的标题和标签，其次从文案中提取
    title = ai_title or (rewritten_text.strip().split("。")[0][:50] if rewritten_text else "精彩视频")
    tags = ai_tags or ""
    if not tags and rewritten_text:
        tag_matches = re.findall(r'#[\w\u4e00-\u9fff]+', rewritten_text)
        tags = " ".join(tag_matches[:5]) if tag_matches else ""
    desc = rewritten_text.strip()[:200] if rewritten_text else ""

    logger.info(f"[Publish] 视频={video_path}, 标题={title}, 标签={tags}")

    # 在后台线程中运行 Playwright（不阻塞 Gradio UI）
    import threading

    def _do_publish():
        try:
            sys.path.insert(0, str(ROOT_DIR))
            import publish_simple
            # 确保 profile 目录在项目根下
            profile_dir = os.path.join(str(ROOT_DIR), ".douyin_browser_profile")
            publish_simple.PROFILE_DIR = profile_dir
            ok = publish_simple.run(
                video_path=os.path.abspath(video_path),
                title=title,
                tags=tags,
                desc=desc,
            )
            logger.info(f"[Publish] 发布流程结束, success={ok}")
        except Exception as e:
            logger.error(f"[Publish] 后台发布异常: {e}", exc_info=True)

    t = threading.Thread(target=_do_publish, daemon=True)
    t.start()

    return video_path, (
        f"✅ 已启动发布流程！\n"
        f"浏览器窗口将自动打开，请扫码登录抖音创作者平台\n"
        f"视频将自动上传，标题/文案会自动填写\n"
        f"请在浏览器中检查信息并手动点击「发布」按钮"
    )


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

    # LipSync (MuseTalk)
    try:
        import modelscope
        from local_models.config import LIPSYNC_CONFIG
        lip_ok = Path(LIPSYNC_CONFIG["local_path"]).is_dir()
        lines.append(f"- LipSync (MuseTalk v1.5): {'✅ 模型就绪' if lip_ok else '❌ 缺失 → ' + LIPSYNC_CONFIG['local_path']}")
        if lip_ok:
            lines.append("  ⚠️ MuseTalk 需要 ~6-8GB 独享显存，加载前会自动卸载其他模型")
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

# 全局 CSS（Gradio 6.x 要求在 launch() 中传入）
PIPELINE_CSS = """
.step-header { font-size: 1.15em; font-weight: bold; margin-bottom: 6px; padding: 6px 10px; border-radius: 6px; color: #1f2937; }
.s0 { background: #fef3c7; color: #1f2937; } .s1 { background: #dbeafe; color: #1f2937; } .s2 { background: #ede9fe; color: #1f2937; } .s3 { background: #d1fae5; color: #1f2937; } .s4 { background: #fce7f3; color: #1f2937; }
.pipeline-status { font-size: 0.9em; }
#full-run-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.douyin-row { background: linear-gradient(135deg, #f0f4ff 0%, #f5f3ff 100%); border: 1px solid #c7d2fe; border-radius: 12px; padding: 16px 20px; margin-bottom: 16px; color: #1f2937; box-shadow: 0 1px 3px rgba(99, 102, 241, 0.08); }
"""

def create_pipeline_ui():
    """创建 Gradio 管线界面 — 横向四步布局"""

    with gr.Blocks(title="口播智能体") as demo:

        gr.Markdown("# 🎬 口播智能体")

        # ================================================================
        # 🔗 抖音链接下载区
        # ================================================================
        with gr.Row(elem_classes=["douyin-row"]):
            with gr.Column(scale=4):
                douyin_url = gr.Textbox(
                    label="🔗 抖音视频链接",
                    placeholder="粘贴抖音分享链接，例如: https://v.douyin.com/xxxxx/",
                    lines=1,
                    show_label=True,
                )
            with gr.Column(scale=1, min_width=120):
                download_btn = gr.Button("📥 下载视频", variant="secondary", size="lg")
            with gr.Column(scale=2):
                download_status = gr.Textbox(label="下载状态", interactive=False, show_label=True)

        gr.Markdown("---")

        # ================================================================
        # 步骤一：视频输入 & 文案提取（横向：上传 | 操作 | 结果）
        # ================================================================
        gr.Markdown('<div class="step-header s1">📹 步骤一：视频输入 &amp; 文案提取</div>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=200):
                input_video = gr.Video(label="口播视频", sources=["upload"], height=260)
            with gr.Column(scale=1, min_width=140):
                gr.Markdown("")  # spacer
                extract_btn = gr.Button("1️⃣ 提取视频文案", variant="secondary", size="lg")
                asr_status = gr.Textbox(label="状态", interactive=False, show_label=True)
            with gr.Column(scale=2):
                original_text = gr.Textbox(
                    label="原始文案（可手动编辑）",
                    lines=8,
                    placeholder="上传视频后点击提取，或手动粘贴文案...",
                    show_label=True,
                )

        gr.Markdown("---")

        # ================================================================
        # 步骤二：AI 仿写文案（横向：设置 | 操作 | 结果）
        # ================================================================
        gr.Markdown('<div class="step-header s2">✍️ 步骤二：AI 仿写文案</div>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=200):
                ai_mode = gr.Radio(
                    choices=["AI自动仿写", "根据指令仿写"],
                    value="AI自动仿写",
                    label="仿写模式",
                )
                custom_prompt = gr.Textbox(
                    label="自定义仿写指令",
                    placeholder="例如：用幽默口吻改写，加入网络热梗",
                    lines=3,
                    visible=False,
                )
            with gr.Column(scale=1, min_width=140):
                gr.Markdown("")  # spacer
                rewrite_btn = gr.Button("2️⃣ 执行 AI 仿写", variant="secondary", size="lg")
                step_status = gr.Textbox(label="状态", lines=2, interactive=False, show_label=True)
            with gr.Column(scale=2):
                rewritten_text = gr.Textbox(
                    label="仿写后文案",
                    lines=5,
                    placeholder="点击执行仿写后显示结果...",
                    show_label=True,
                )
                with gr.Row():
                    title_output = gr.Textbox(
                        label="📌 视频标题", lines=1,
                        placeholder="AI 自动生成的标题",
                    )
                    tags_output = gr.Textbox(
                        label="🏷️ 标签", lines=1,
                        placeholder="AI 自动生成的标签",
                    )

        def toggle_prompt(mode):
            return gr.update(visible=(mode == "根据指令仿写"))

        ai_mode.change(fn=toggle_prompt, inputs=[ai_mode], outputs=[custom_prompt])

        gr.Markdown("---")

        # ================================================================
        # 步骤三：语音合成（横向：音色设置 | 操作 | 结果）
        # ================================================================
        gr.Markdown('<div class="step-header s3">🔊 步骤三：语音合成（TTS）</div>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=260):
                # 音色来源
                voice_mode = gr.Radio(
                    choices=["内置音色", "上传音频", "录制音频"],
                    value="内置音色",
                    label="音色来源",
                )
                # 内置音色下拉
                with gr.Group(visible=True) as builtin_group:
                    speaker_dd = gr.Dropdown(
                        choices=["中文女", "中文男", "默认女声", "默认男声"],
                        value="中文女",
                        label="内置音色",
                    )
                # 上传 / 录音（动态显示）
                with gr.Group(visible=False) as upload_group:
                    custom_audio_upload = gr.Audio(
                        label="上传参考音频（WAV/MP3，5-15 秒最佳）",
                        sources=["upload"],
                        type="filepath",
                    )
                with gr.Group(visible=False) as record_group:
                    custom_audio_record = gr.Audio(
                        label="录制参考音频（建议 5-15 秒清晰人声）",
                        sources=["microphone"],
                        type="filepath",
                    )
                # 参考音频对应文本：三种模式都可能用到 zero-shot，统一展示
                custom_audio_text = gr.Textbox(
                    label="参考音频对应的文本（不填时使用默认示例文本）",
                    placeholder="例如：大家好，欢迎来到我的频道。",
                    lines=2,
                )
                # 语速
                speed_slider = gr.Slider(
                    value=1.0, minimum=0.5, maximum=2.0, step=0.1,
                    label="语速调节",
                )

            with gr.Column(scale=1, min_width=140):
                gr.Markdown("")  # spacer
                tts_btn = gr.Button("3️⃣ 生成语音", variant="secondary", size="lg")
                tts_status = gr.Textbox(label="状态", interactive=False, show_label=True)
            with gr.Column(scale=2):
                audio_output = gr.Audio(label="合成音频", type="filepath")

        # 动态切换音色来源的可见性
        def switch_voice_mode(mode):
            return (
                gr.update(visible=(mode == "内置音色")),
                gr.update(visible=(mode == "上传音频")),
                gr.update(visible=(mode == "录制音频")),
            )

        voice_mode.change(
            fn=switch_voice_mode,
            inputs=[voice_mode],
            outputs=[builtin_group, upload_group, record_group],
        )

        gr.Markdown("---")

        # ================================================================
        # 步骤四：口型合成（横向：设置 | 操作 | 结果，对齐其他步骤布局）
        # ================================================================
        gr.Markdown('<div class="step-header s4">🎭 步骤四：口型合成（MuseTalk）</div>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=260):
                # 形象视频上传
                avatar_video = gr.Video(
                    label="🎭 数字人形象视频/图片",
                    sources=["upload"],
                    height=200,
                )
                gr.Markdown(
                    value="上传你的数字人形象视频或图片，  \n"
                           "音频将驱动该形象进行口型同步。  \n"
                           "（不填则使用步骤一的参考视频）"
                )
            with gr.Column(scale=1, min_width=140):
                gr.Markdown("")  # spacer
                lipsync_btn = gr.Button("4️⃣ 生成口型视频", variant="secondary", size="lg")
                step_status = gr.Textbox(label="状态", interactive=False, show_label=True)
            with gr.Column(scale=2):
                video_output = gr.Video(label="最终数字人视频", height=360)

        # ================================================================
        # 步骤五：发布到抖音
        # ================================================================
        gr.Markdown("---")
        gr.Markdown('<div class="step-header s0">📤 步骤五：发布到抖音</div>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=260):
                publish_info = gr.Markdown(
                    value="**发布模式：半自动**  \n"
                           "自动上传视频 + 填写标题和文案，  \n"
                           "用户在浏览器中手动点击「发布」。  \n"
                           "⚠️ 首次使用需扫码登录抖音创作者平台。"
                )
            with gr.Column(scale=1, min_width=140):
                gr.Markdown("")
                publish_btn = gr.Button("📤 发布到抖音", variant="secondary", size="lg")
                publish_status = gr.Textbox(label="状态", interactive=False, show_label=True)
            with gr.Column(scale=2):
                publish_video = gr.Video(label="发布预览", height=360)

        # ================================================================
        # 一键全流程
        # ================================================================
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                full_run_btn = gr.Button(
                    "🚀 一键全流程执行",
                    variant="primary",
                    elem_id="full-run-btn",
                    size="lg",
                )
            with gr.Column(scale=3):
                full_status = gr.Textbox(label="全流程状态", lines=2, interactive=False, show_label=True)

        # ===== 事件绑定 =====

        # 抖音下载
        download_btn.click(
            fn=step0_download_douyin,
            inputs=[douyin_url],
            outputs=[input_video, download_status],
        )

        # 步骤 1: 提取文案
        extract_btn.click(
            fn=step1_extract_text,
            inputs=[input_video],
            outputs=[original_text, asr_status],
        )

        # 步骤 2: AI 仿写
        rewrite_btn.click(
            fn=step2_rewrite,
            inputs=[original_text, ai_mode, custom_prompt],
            outputs=[rewritten_text, title_output, tags_output, step_status],
        )

        # 步骤 3: 语音合成
        def _tts_dispatch(
            text, speed, voice_mode, speaker,
            custom_audio_upload, custom_audio_record, custom_audio_text,
        ):
            """根据 voice_mode 选择正确的参考音频源"""
            if voice_mode == "上传音频":
                custom_audio = custom_audio_upload
            elif voice_mode == "录制音频":
                custom_audio = custom_audio_record
            else:
                custom_audio = None
            return step3_generate_audio(
                text=text,
                speed=speed,
                voice_mode=voice_mode,
                speaker=speaker,
                custom_audio=custom_audio,
                custom_audio_text=custom_audio_text,
            )

        tts_btn.click(
            fn=_tts_dispatch,
            inputs=[
                rewritten_text, speed_slider, voice_mode, speaker_dd,
                custom_audio_upload, custom_audio_record, custom_audio_text,
            ],
            outputs=[audio_output, tts_status],
        )

        # 步骤 4: 口型合成（优先使用用户上传的形象视频，回退到步骤一视频）
        lipsync_btn.click(
            fn=step4_lipsync,
            inputs=[avatar_video, audio_output, input_video],
            outputs=[video_output, step_status],
        )

        # 步骤 5: 发布到抖音
        publish_btn.click(
            fn=step5_publish,
            inputs=[video_output, rewritten_text, title_output, tags_output],
            outputs=[publish_video, publish_status],
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
        css=PIPELINE_CSS,
        theme=gr.themes.Default(),
    )
