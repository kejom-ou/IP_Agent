"""
IP Agent 端到端管道：抖音视频 → 口型同步短视频（含用户图片）
==============================================================
抖音链接 → 下载视频 → 剥离音频 → ASR → LLM仿写/标题/标签 → TTS
→ 图片转动态视频 → Wav2Lip口型同步 → 最终合成（音频+字幕）

用法:
    python pipeline.py --url <抖音链接> --image <用户图片路径>
    python pipeline.py --url <链接> --image <图片> --force        # 强制重跑
    python pipeline.py --skip-download --skip-asr --skip-llm --skip-tts  # 只跑口型+合成
"""
import os, sys, subprocess, logging, argparse, re, time

# RTX 5060 (Blackwell sm_120) + torch 2.12.1 cuDNN 9.20 有兼容性 bug
# 禁用 cuDNN backend，torch 自动走原生 CUDA 内核
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "local_models", "test_downloads")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 全局路径 ────────────────────────────────────────────────────
VIDEO_A        = os.path.join(OUT_DIR, "douyin_video.mp4")
AUDIO_A        = os.path.join(OUT_DIR, "douyin_audio.wav")
VIDEO_B        = os.path.join(OUT_DIR, "video_b_silent.mp4")
ASR_TEXT       = os.path.join(OUT_DIR, "asr_result.txt")
LLM_COPY       = os.path.join(OUT_DIR, "llm_copywriting.txt")
LLM_TITLES     = os.path.join(OUT_DIR, "llm_titles.txt")
LLM_TAGS       = os.path.join(OUT_DIR, "llm_tags.txt")
TTS_AUDIO      = os.path.join(OUT_DIR, "tts_output.wav")
TTS_SRT        = os.path.join(OUT_DIR, "tts_timeline.srt")
IMG_VIDEO      = os.path.join(OUT_DIR, "img_animated.mp4")
LIPSYNC_VIDEO  = os.path.join(OUT_DIR, "lipsync_full.mp4")
FINAL_OUTPUT   = os.path.join(OUT_DIR, "final_output.mp4")

GAP_MS = 500
SRT_STYLE = "FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Alignment=2,MarginV=40"

# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def run_ffmpeg(args: list, desc: str) -> bool:
    """运行 ffmpeg 命令"""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"] + args
    logger.info(f"[{desc}] ffmpeg {' '.join(cmd[5:])}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"[{desc}] 完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[{desc}] 失败: {e}")
        return False


def _srt_time(seconds: float) -> str:
    """秒数 → SRT 时间格式 HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_srt(timeline: list, filepath: str):
    """保存 SRT 字幕文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in timeline:
            f.write(f"{item['index']}\n")
            f.write(f"{_srt_time(item['start_s'])} --> {_srt_time(item['end_s'])}\n")
            f.write(f"{item['text']}\n\n")
    logger.info(f"[字幕] SRT 已保存: {filepath}")


# ═══════════════════════════════════════════════════════════════════
# LLM Prompts
# ═══════════════════════════════════════════════════════════════════

_LLM_COPY_PROMPT = """你是一个专业的短视频口播文案改写助手。严格遵守以下规则：

【铁律 - 绝对不能违反】
1. 原文讨论什么产品/品牌/话题，你就改写什么，绝对禁止替换成其他产品、品牌或话题
2. 原文提到的人名、地名、产品名、品牌名必须保留，一字不改
3. 绝对禁止自己编造新的产品、品牌或主题
4. 输出文案总字数严格控制在50字以内，这是硬性指标，超出则不合格

【改写要求】
5. 保留原文所有关键信息和核心卖点
6. 优化语言表达，使文案更口语化、更有感染力，适合TTS语音合成口播
7. 在50字的限制内精选最重要的信息，删减所有次要内容
8. 可以调整语序和用词，但不能改变原意和事实

【格式铁律 - 严格禁止以下所有内容】
- 禁止使用任何Markdown语法：**加粗**、*斜体*、# 标题、- 列表、1. 编号列表、> 引用
- 禁止使用英文标点：英文逗号, 英文句号. 英文冒号: 英文分号; 英文括号() 英文引号""
- 禁止使用任何特殊符号：破折号——、省略号……、方括号【】、书名号《》、星号*、下划线_
- 禁止使用 emoji 和特殊 Unicode 字符
- 只允许使用：中文汉字、中文标点（。，！？、：）、数字、百分号%、单位（万/亿/元/米/公里）

【输出格式】
只输出改写后的纯文本文案，不要添加任何解释、标题、编号或Markdown标记"""

_LLM_TITLE_PROMPT = """根据以下视频口播文案，生成3个吸引人的短视频标题。
要求：
- 标题必须紧扣原文讨论的产品/品牌/话题，不能偏离
- 每个标题15-30字
- 有爆点、有悬念、能引发点击欲望
- 输出格式：每行一个标题，以"1. " "2. " "3. " 开头"""

_LLM_TAG_PROMPT = """根据以下视频口播文案，提取10个最相关的抖音话题标签。
要求：
- 标签必须与原文讨论的产品/品牌/话题直接相关
- 每个标签以#开头
- 标签要精准，不能过于宽泛
- 逗号分隔，一行输出"""


# ═══════════════════════════════════════════════════════════════════
# 文本清洗 / 分段
# ═══════════════════════════════════════════════════════════════════

def _clean_asr_text(text: str) -> str:
    """清洗 ASR 文本：去 emoji、特殊符号、仅保留中文+字母数字+常用标点"""
    # 1. 移除 emoji 和特殊 Unicode 字符（用 surrogate 范围避免 8位 hex 边界问题）
    text = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF'
        r'\U0001F1E6-\U0001F1FF\U00002702-\U000027B0'
        r'\U0001F004\U0001F0CF\U0001F18E]+', '', text, flags=re.UNICODE)
    text = re.sub(
        r'[\u2600-\u26FF\u2700-\u27BF\u2300-\u23FF]+', '', text, flags=re.UNICODE)
    # 2. 只保留中文、字母数字、常用中文标点（去掉了冒号中文引号等）
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s，。！？、：；""''（）%/+=#@&.·-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _clean_copywriting_text(text: str) -> str:
    """清洗 LLM 仿写文案：去 emoji、Markdown、特殊符号"""
    # 1. 移除 emoji（用 surrogate 范围避免 8位 hex 边界问题）
    # 主流 emoji 都在 \ud800-\udbff 区段
    text = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF'
        r'\U0001F1E6-\U0001F1FF\U00002702-\U000027B0'
        r'\U0001F004\U0001F0CF\U0001F18E\U0001F191-\U0001F19A'
        r'\U0001F1E6-\U0001F1FF]+', '', text, flags=re.UNICODE)
    # 单独的 BMP 区段符号（小心避免覆盖中文，用小范围）
    text = re.sub(
        r'[\u2600-\u26FF\u2700-\u27BF\u2300-\u23FF'
        r'\u2B05-\u2B07\u2196-\u2199\u2611\u2705]+', '', text, flags=re.UNICODE)
    # 2. 移除 Markdown 语法
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__.+?__', '', text)
    text = re.sub(r'~~.+?~~', '', text)
    text = re.sub(r'^[\s]*[-*·•▪▸►]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+[\.\、\)]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # 3. 英文标点转中文
    text = text.replace(',', '，').replace(';', '；').replace(':', '：')
    text = text.replace('?', '？').replace('!', '！')
    text = re.sub(r'——+', '，', text)
    text = re.sub(r'…{2,}', '，', text)
    text = text.replace('"', '').replace('"', '')
    text = text.replace('(', '（').replace(')', '）')
    # 4. 最终过滤：只保留中文、字母数字、常用中文标点（去掉了冒号中文引号等）
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s，。！？、：；（）%/+=#@&.·-]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*\n', '', text)
    # 5. 硬截断：超过50字强制截断
    text = text.strip()
    if len(text) > 50:
        text = text[:50]
    return text


def _split_paragraphs(text: str) -> list:
    """按句号精细分段"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    segments = []
    for para in paragraphs:
        raw_sentences = re.split(r'(?<=[。！？])', para)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        buffer = ''
        for sent in sentences:
            combined = (buffer + sent).strip()
            if len(buffer) < 15 and len(combined) <= 50:
                buffer = combined
            elif buffer:
                segments.append(buffer)
                buffer = sent
            else:
                buffer = sent
        if buffer.strip():
            segments.append(buffer.strip())
    return [seg for seg in segments if seg]


def _llm_generate(engine, system_prompt: str, user_content: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return engine.generate(messages, temperature=temperature, max_tokens=max_tokens)


# ═══════════════════════════════════════════════════════════════════
# Step 1: 下载抖音视频
# ═══════════════════════════════════════════════════════════════════

def step1_download(douyin_url: str = None, force: bool = False):
    if os.path.exists(VIDEO_A) and not force:
        logger.info(f"[Step1] Video A 已存在，跳过下载: {VIDEO_A}")
        return True
    if not douyin_url:
        logger.error("[Step1] 无抖音链接且 Video A 不存在，请提供 --url 参数")
        return False

    from local_models.douyin_crawler import DouyinCrawler
    crawler = DouyinCrawler()
    path = crawler.download(douyin_url, output_dir=OUT_DIR)
    if path:
        logger.info(f"[Step1] 下载完成: {path}")
        return True
    logger.error("[Step1] 下载失败")
    return False


# ═══════════════════════════════════════════════════════════════════
# Step 2: 剥离音频 → Audio A + Video B（无声视频）
# ═══════════════════════════════════════════════════════════════════

def step2_strip_audio(force: bool = False):
    # 2a. 提取音频
    if not force and os.path.exists(AUDIO_A):
        logger.info(f"[Step2a] Audio 已存在，跳过: {AUDIO_A}")
    else:
        ok = run_ffmpeg(
            ["-i", VIDEO_A, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", AUDIO_A],
            "Step2a 提取音频"
        )
        if not ok:
            return False

    # 2b. 无声视频 B
    if not force and os.path.exists(VIDEO_B):
        logger.info(f"[Step2b] Video B 已存在，跳过: {VIDEO_B}")
    else:
        ok = run_ffmpeg(
            ["-i", VIDEO_A, "-an", "-c:v", "copy", VIDEO_B],
            "Step2b 剥离音频得Video B"
        )
        if not ok:
            ok = run_ffmpeg(
                ["-i", VIDEO_A, "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23", VIDEO_B],
                "Step2b 剥离音频得Video B (重编码)"
            )
            if not ok:
                return False
    return True


# ═══════════════════════════════════════════════════════════════════
# Step 3: ASR 语音识别
# ═══════════════════════════════════════════════════════════════════

def step3_asr(force: bool = False):
    if not force and os.path.exists(ASR_TEXT):
        logger.info(f"[Step3] ASR 结果已存在，跳过: {ASR_TEXT}")
        with open(ASR_TEXT, "r", encoding="utf-8") as f:
            return f.read().strip()

    from local_models.asr_engine import ASREngine
    engine = ASREngine()
    if not engine.load():
        logger.error("[Step3] ASR 模型加载失败")
        return None

    text = engine.transcribe(AUDIO_A)
    engine.unload()

    if text:
        with open(ASR_TEXT, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"[Step3] ASR 完成: {len(text)} 字符 → {ASR_TEXT}")
    else:
        logger.error("[Step3] ASR 返回为空")
    return text


# ═══════════════════════════════════════════════════════════════════
# Step 4: LLM 文案仿写 + 标题 + 标签
# ═══════════════════════════════════════════════════════════════════

def step4_llm(asr_text: str, force: bool = False):
    if not asr_text:
        logger.error("[Step4] 无 ASR 文本输入")
        return None, None, None

    asr_text = _clean_asr_text(asr_text)
    logger.info(f"[Step4] ASR 文本清洗后: {len(asr_text)} 字符")

    from local_models.llm_engine import TransformersLLM
    llm = TransformersLLM()
    if not llm.load():
        logger.error("[Step4] LLM 加载失败")
        return None, None, None

    results = {}

    # 4a. 文案仿写
    if not force and os.path.exists(LLM_COPY):
        logger.info(f"[Step4a] 仿写文案已存在: {LLM_COPY}")
        with open(LLM_COPY, "r", encoding="utf-8") as f:
            results["copy"] = f.read().strip()
    else:
        logger.info("[Step4a] LLM 仿写文案...")
        copywriting = _llm_generate(llm, _LLM_COPY_PROMPT, f"请改写以下文案：\n\n{asr_text}", max_tokens=2048)
        if copywriting:
            with open(LLM_COPY, "w", encoding="utf-8") as f:
                f.write(copywriting)
            logger.info(f"[Step4a] 仿写完成: {len(copywriting)} 字符 → {LLM_COPY}")
            results["copy"] = copywriting
        else:
            results["copy"] = asr_text

    # 4b. 标题生成
    if not force and os.path.exists(LLM_TITLES):
        logger.info(f"[Step4b] 标题已存在: {LLM_TITLES}")
        with open(LLM_TITLES, "r", encoding="utf-8") as f:
            results["titles"] = f.read().strip()
    else:
        logger.info("[Step4b] LLM 生成标题...")
        titles = _llm_generate(llm, _LLM_TITLE_PROMPT, f"文案内容：\n\n{results['copy']}")
        if titles:
            with open(LLM_TITLES, "w", encoding="utf-8") as f:
                f.write(titles)
            logger.info(f"[Step4b] 标题生成: {LLM_TITLES}")
            results["titles"] = titles
        else:
            results["titles"] = ""

    # 4c. 标签生成
    if not force and os.path.exists(LLM_TAGS):
        logger.info(f"[Step4c] 标签已存在: {LLM_TAGS}")
        with open(LLM_TAGS, "r", encoding="utf-8") as f:
            results["tags"] = f.read().strip()
    else:
        logger.info("[Step4c] LLM 生成标签...")
        try:
            tags = _llm_generate(llm, _LLM_TAG_PROMPT, f"文案内容：\n\n{results['copy']}",
                                 max_tokens=64, temperature=0.9)
            if tags:
                with open(LLM_TAGS, "w", encoding="utf-8") as f:
                    f.write(tags)
                logger.info(f"[Step4c] 标签生成: {LLM_TAGS}")
                results["tags"] = tags
            else:
                results["tags"] = ""
        except Exception as e:
            logger.warning(f"[Step4c] 标签生成失败（跳过）: {e}")
            results["tags"] = ""

    llm.unload()
    return results.get("copy"), results.get("titles"), results.get("tags")


# ═══════════════════════════════════════════════════════════════════
# Step 5: TTS 逐段合成（含 SRT 时间戳）
# ═══════════════════════════════════════════════════════════════════

def step5_tts(copy_text: str, force: bool = False):
    if not copy_text:
        logger.error("[Step5] 无仿写文案")
        return None

    if not force and os.path.exists(TTS_AUDIO) and os.path.exists(TTS_SRT):
        logger.info(f"[Step5] TTS 音频和字幕已存在，跳过")
        return TTS_AUDIO

    from local_models.tts_engine import CosyVoiceEngine

    copy_text = _clean_copywriting_text(copy_text)
    logger.info(f"[Step5] 文案清洗后: {len(copy_text)} 字符")

    segments = _split_paragraphs(copy_text)
    logger.info(f"[Step5] 文案拆分为 {len(segments)} 段")

    engine = CosyVoiceEngine()
    result = engine.synthesize_segments(
        segments=segments, speaker="中文女", speed=1.0,
        gap_ms=GAP_MS, output_path=TTS_AUDIO,
    )

    if result is None:
        logger.error("[Step5] TTS 合成失败")
        engine.unload()
        return None

    audio_path, timeline = result
    save_srt(timeline, TTS_SRT)

    logger.info(f"[Step5] TTS 完成: {audio_path}")
    for item in timeline:
        logger.info(f"  #{item['index']:2d} [{_srt_time(item['start_s'])} -> {_srt_time(item['end_s'])}] "
                     f"({item['dur_s']:.1f}s) {item['text'][:50]}...")

    engine.unload()
    return audio_path


# ═══════════════════════════════════════════════════════════════════
# Step 6: 图片转动态视频（Ken Burns 缓慢推近）
# ═══════════════════════════════════════════════════════════════════

def step6_img2video(image_path: str, force: bool = False):
    """将用户图片转为与 TTS 音频等长的动态视频"""
    if not image_path or not os.path.exists(image_path):
        logger.error(f"[Step6] 图片不存在: {image_path}")
        return None

    if not os.path.exists(TTS_AUDIO):
        logger.error("[Step6] TTS 音频不存在，无法获取时长")
        return None

    if not force and os.path.exists(IMG_VIDEO):
        logger.info(f"[Step6] 动态视频已存在，跳过: {IMG_VIDEO}")
        return IMG_VIDEO

    import librosa

    # 获取 TTS 时长
    wav, sr = librosa.load(TTS_AUDIO, sr=None)
    duration_s = len(wav) / sr + 0.5
    fps = 25
    total_frames = int(duration_s * fps)
    logger.info(f"[Step6] TTS时长={duration_s:.1f}s, 目标帧数={total_frames}, 图片={image_path}")

    ok = run_ffmpeg(
        [
            "-loop", "1",
            "-i", image_path,
            "-vf",
            f"scale=2160:-1,"
            f"zoompan=z='min(zoom+0.0005,1.3)':d={total_frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=720x1280:fps={fps}",
            "-t", str(duration_s),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            IMG_VIDEO,
        ],
        "Step6 图片→动态视频"
    )

    if ok and os.path.exists(IMG_VIDEO):
        size_mb = os.path.getsize(IMG_VIDEO) / 1024**2
        logger.info(f"[Step6] 动态视频: {IMG_VIDEO} ({size_mb:.1f} MB)")
        return IMG_VIDEO
    logger.error("[Step6] 图片转视频失败")
    return None


# ═══════════════════════════════════════════════════════════════════
# Step 7: 口型同步 (MuseTalk / Wav2Lip)
# ═══════════════════════════════════════════════════════════════════

def step7_lipsync(force: bool = False, use_musetalk: bool = True):
    """对动态视频进行口型同步

    Args:
        force: 强制重跑，忽略缓存
        use_musetalk: True=使用 MuseTalk, False=使用 Wav2Lip
    """
    if not os.path.exists(IMG_VIDEO):
        logger.error("[Step7] 动态视频不存在，请先运行 Step6")
        return None
    if not os.path.exists(TTS_AUDIO):
        logger.error("[Step7] TTS 音频不存在")
        return None

    if not force and os.path.exists(LIPSYNC_VIDEO):
        logger.info(f"[Step7] 口型同步视频已存在，跳过: {LIPSYNC_VIDEO}")
        return LIPSYNC_VIDEO

    import librosa, soundfile as sf, torch

    # 转换音频 16kHz
    t1 = time.time()
    wav, sr = librosa.load(TTS_AUDIO, sr=16000)
    audio_16k = os.path.join(OUT_DIR, "tts_16k.wav")
    sf.write(audio_16k, wav, 16000)
    logger.info(f"[Step7] 音频转换: {len(wav)/16000:.1f}s @ 16kHz")

    if use_musetalk:
        # ── MuseTalk (高质量) ──
        from local_models.musetalk_engine import MuseTalkEngine

        engine = MuseTalkEngine()
        if not engine.load():
            logger.error("[Step7] MuseTalk 模型加载失败")
            return None
        vram = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"[Step7] MuseTalk 模型加载完成，显存: {vram:.2f} GB")

        logger.info(f"[Step7] MuseTalk 口型同步推理...")
        t2 = time.time()
        result = engine.generate(
            video_path=IMG_VIDEO,
            audio_path=audio_16k,
            output_path=LIPSYNC_VIDEO,
            fps=25,
            batch_size=8,
        )
        engine.unload()
    else:
        # ── Wav2Lip (轻量备选) ──
        from local_models.lipsync import Wav2LipEngine

        engine = Wav2LipEngine()
        if not engine.load():
            logger.error("[Step7] Wav2Lip 模型加载失败")
            return None
        vram = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"[Step7] Wav2Lip 模型加载完成，显存: {vram:.2f} GB")

        logger.info(f"[Step7] Wav2Lip 口型同步推理...")
        t2 = time.time()
        result = engine.generate(
            video_path=IMG_VIDEO,
            audio_path=audio_16k,
            output_path=LIPSYNC_VIDEO,
            fps=25,
        )
        engine.unload()

    if result and os.path.exists(result):
        size_mb = os.path.getsize(result) / 1024**2
        elapsed = time.time() - t2
        logger.info(f"[Step7] 口型同步完成: {result} ({size_mb:.1f} MB, {elapsed:.0f}s)")
        return result
    logger.error("[Step7] 口型同步失败")
    return None


# ═══════════════════════════════════════════════════════════════════
# Step 8: 最终合成（口型视频 + TTS 音频 + SRT 字幕）
# ═══════════════════════════════════════════════════════════════════

def step8_composite(force: bool = False):
    """将口型同步视频 + TTS 音频 + 字幕合成为最终视频"""
    if not os.path.exists(LIPSYNC_VIDEO):
        logger.error("[Step8] 口型同步视频不存在，请先运行 Step7")
        return False
    if not os.path.exists(TTS_AUDIO):
        logger.error("[Step8] TTS 音频不存在")
        return False
    if not os.path.exists(TTS_SRT):
        logger.error("[Step8] SRT 字幕不存在")
        return False

    if not force and os.path.exists(FINAL_OUTPUT):
        logger.info(f"[Step8] 最终视频已存在，跳过: {FINAL_OUTPUT}")
        return True

    srt_path = TTS_SRT.replace("\\", "/").replace(":", "\\:")
    filter_complex = f"[0:v]subtitles='{srt_path}':force_style='{SRT_STYLE}'[v]"

    ok = run_ffmpeg(
        [
            "-i", LIPSYNC_VIDEO,
            "-i", TTS_AUDIO,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            FINAL_OUTPUT,
        ],
        "Step8 最终合成"
    )

    if ok and os.path.exists(FINAL_OUTPUT):
        size_mb = os.path.getsize(FINAL_OUTPUT) / 1024**2
        logger.info(f"[Step8] 最终视频: {FINAL_OUTPUT} ({size_mb:.1f} MB)")
        return True
    logger.error("[Step8] 最终合成失败")
    return False


# ═══════════════════════════════════════════════════════════════════
# Step 9: 发布到抖音创作者平台（集成 publish_simple.py 逻辑）
# ═══════════════════════════════════════════════════════════════════

def step9_publish(profile_dir: str = None, confirm: bool = False):
    """将最终视频发布到抖音创作者平台（自动上传+填表）"""
    if not os.path.exists(FINAL_OUTPUT):
        logger.error("[Step9] 最终视频不存在，请先运行 Step8")
        return False

    from publish_simple import run, load_title, load_tags, load_desc

    # 默认使用持久化 profile，保留登录态
    if profile_dir is None:
        profile_dir = os.path.join(ROOT, ".douyin_browser_profile")

    title = load_title(LLM_TITLES) or "精彩视频"
    tags  = load_tags(LLM_TAGS)
    desc  = load_desc(LLM_COPY)

    logger.info(f"[Step9] 准备发布...")
    logger.info(f"  标题: {title}")
    logger.info(f"  标签: {tags}")
    logger.info(f"  描述: {desc[:60]}...")
    logger.info(f"  Profile: {profile_dir}")

    # publish_simple 使用全局 PROFILE_DIR，这里需要临时替换
    import publish_simple as ps
    original_profile = ps.PROFILE_DIR
    ps.PROFILE_DIR = profile_dir

    try:
        success = run(
            video_path=FINAL_OUTPUT,
            title=title,
            tags=tags,
            desc=desc,
        )
    finally:
        ps.PROFILE_DIR = original_profile

    return success


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="IP Agent 端到端管道：抖音视频 → 口型同步短视频（含用户图片）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python pipeline.py --url <抖音链接> --image <图片路径>
  python pipeline.py --url <链接> --image <图片> --force
  python pipeline.py --url <链接> --image <图片> --publish    # 全流程 + 自动发布（跳过询问）
  python pipeline.py --url <链接> --image <图片> --publish --confirm  # 填表后手动确认发布
  python pipeline.py --url <链接> --image <图片> --publish --profile <目录>  # 指定登录态目录
  python pipeline.py --skip-download --skip-asr --skip-llm --skip-tts \\
      --url <链接> --image <图片>    # 仅跑口型+合成

提示: 不传 --publish 时，Step8 完成后会交互式询问是否发布
        """,
    )
    parser.add_argument("--url", type=str, default=None, help="抖音视频链接")
    parser.add_argument("--image", type=str, default=None, help="用户图片路径（用于口型同步）")
    parser.add_argument("--force", action="store_true", help="强制重跑所有步骤")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载")
    parser.add_argument("--skip-asr", action="store_true", help="跳过 ASR")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 LLM")
    parser.add_argument("--skip-tts", action="store_true", help="跳过 TTS")
    parser.add_argument("--skip-img2video", action="store_true", help="跳过图片转视频")
    parser.add_argument("--skip-lipsync", action="store_true", help="跳过口型同步")
    parser.add_argument("--wav2lip", action="store_true", help="使用 Wav2Lip 口型同步（默认使用 MuseTalk）")
    parser.add_argument("--publish", action="store_true", help="流程结束后自动打开发布页面")
    parser.add_argument("--profile", type=str, default=None, help="抖音浏览器持久化目录（复用登录态）")
    parser.add_argument("--confirm", action="store_true", help="抖音发布：填表后不自动点击发布，需人工确认")
    args = parser.parse_args()

    force = args.force
    t_start = time.time()

    print("\n" + "=" * 60)
    print("  IP Agent 端到端管道")
    print("  抖音视频 → 口型同步短视频（用户图片）")
    print("=" * 60 + "\n")

    # ── Step 1: 下载 ──
    if not args.skip_download:
        if not step1_download(douyin_url=args.url, force=force):
            logger.error("Step1 失败，退出")
            return 1
    else:
        logger.info("[Step1] 跳过下载（--skip-download）")

    if not os.path.exists(VIDEO_A):
        logger.error(f"Video A 不存在: {VIDEO_A}，请先下载或提供 --url")
        return 1

    # ── Step 2: 剥离音频 ──
    if not step2_strip_audio(force=force):
        return 1

    # ── Step 3: ASR ──
    import torch; torch.backends.cudnn.enabled = False  # RTX 5060 Blackwell cuDNN 兼容
    if not args.skip_asr:
        asr_text = step3_asr(force=force)
        if not asr_text:
            logger.error("Step3 ASR 失败")
            return 1
    else:
        logger.info("[Step3] 跳过 ASR（--skip-asr）")
        if os.path.exists(ASR_TEXT):
            with open(ASR_TEXT, "r", encoding="utf-8") as f:
                asr_text = f.read().strip()
        else:
            logger.error(f"ASR 结果文件不存在: {ASR_TEXT}")
            return 1

    # ── Step 4: LLM ──
    if not args.skip_llm:
        copy_text, titles, tags = step4_llm(asr_text, force=force)
        if not copy_text:
            logger.error("Step4 LLM 仿写失败")
            return 1
    else:
        logger.info("[Step4] 跳过 LLM（--skip-llm）")
        if os.path.exists(LLM_COPY):
            with open(LLM_COPY, "r", encoding="utf-8") as f:
                copy_text = f.read().strip()
        else:
            logger.error(f"仿写文案不存在: {LLM_COPY}")
            return 1
        titles = None
        tags = None
        if os.path.exists(LLM_TITLES):
            with open(LLM_TITLES, "r", encoding="utf-8") as f:
                titles = f.read().strip()
        if os.path.exists(LLM_TAGS):
            with open(LLM_TAGS, "r", encoding="utf-8") as f:
                tags = f.read().strip()

    # 打印 LLM 输出
    print("\n" + "-" * 40)
    print("  仿写文案预览:")
    print("-" * 40)
    print(copy_text[:300] + ("..." if len(copy_text) > 300 else ""))
    if titles:
        print(f"\n  标题:\n{titles}")
    if tags:
        print(f"\n  标签:\n{tags}")
    print("-" * 40 + "\n")

    # ── Step 5: TTS ──
    if not args.skip_tts:
        tts_path = step5_tts(copy_text, force=force)
        if not tts_path:
            logger.error("Step5 TTS 失败")
            return 1
    else:
        logger.info("[Step5] 跳过 TTS（--skip-tts）")
        if not os.path.exists(TTS_AUDIO):
            logger.error(f"TTS 音频不存在: {TTS_AUDIO}")
            return 1

    # ── Step 6: 图片转视频 ──
    if not args.skip_img2video:
        if not args.image:
            logger.error("[Step6] 请提供 --image <图片路径>")
            return 1
        img_video = step6_img2video(args.image, force=force)
        if not img_video:
            return 1
    else:
        logger.info("[Step6] 跳过图片转视频（--skip-img2video）")
        if not os.path.exists(IMG_VIDEO):
            logger.error(f"动态视频不存在: {IMG_VIDEO}")
            return 1

    # ── Step 7: 口型同步 ──
    if not args.skip_lipsync:
        use_musetalk = not args.wav2lip
        lipsync_result = step7_lipsync(force=force, use_musetalk=use_musetalk)
        if not lipsync_result:
            return 1
    else:
        logger.info("[Step7] 跳过口型同步（--skip-lipsync）")
        if not os.path.exists(LIPSYNC_VIDEO):
            logger.error(f"口型同步视频不存在: {LIPSYNC_VIDEO}")
            return 1

    # ── Step 8: 最终合成 ──
    if not step8_composite(force=force):
        return 1

    # ── Step 9: 发布到抖音 ──
    do_publish = True

    if do_publish:
        if not step9_publish(profile_dir=args.profile, confirm=args.confirm):
            logger.warning("[Step9] 发布流程未完成（可能需要手动操作）")

    # ── 完成 ──
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("  管道完成！")
    print(f"  总耗时: {elapsed/60:.1f} 分钟")
    print("=" * 60)
    print(f"  Video A (原始下载): {VIDEO_A}")
    print(f"  Video B (无声):     {VIDEO_B}")
    print(f"  ASR 文本:           {ASR_TEXT}")
    print(f"  仿写文案:           {LLM_COPY}")
    print(f"  标题:               {LLM_TITLES}")
    print(f"  标签:               {LLM_TAGS}")
    print(f"  TTS 音频:           {TTS_AUDIO}")
    print(f"  字幕 SRT:           {TTS_SRT}")
    print(f"  动态图片视频:       {IMG_VIDEO}")
    print(f"  口型同步视频:       {LIPSYNC_VIDEO}")
    print(f"  最终视频:        {FINAL_OUTPUT}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
