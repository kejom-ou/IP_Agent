"""
完整端到端管道：
抖音链接 → 下载视频A → 剥离音频得视频B → ASR → LLM仿写/标题/标签 → TTS → 合成视频C
"""
import os, sys, subprocess, logging, argparse

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "local_models", "test_downloads")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 路径定义
# ---------------------------------------------------------------------------
VIDEO_A      = os.path.join(OUT_DIR, "douyin_video.mp4")       # 抖音原始视频
AUDIO_A      = os.path.join(OUT_DIR, "douyin_audio.wav")       # 剥离出的音频
VIDEO_B      = os.path.join(OUT_DIR, "video_b_silent.mp4")     # 无声视频（视频B）
ASR_TEXT     = os.path.join(OUT_DIR, "asr_result.txt")
LLM_COPY     = os.path.join(OUT_DIR, "llm_copywriting.txt")
LLM_TITLES   = os.path.join(OUT_DIR, "llm_titles.txt")
LLM_TAGS     = os.path.join(OUT_DIR, "llm_tags.txt")
TTS_AUDIO    = os.path.join(OUT_DIR, "tts_output.wav")
TTS_SRT      = os.path.join(OUT_DIR, "tts_timeline.srt")
VIDEO_C      = os.path.join(OUT_DIR, "video_c_final.mp4")

GAP_MS = 500  # TTS 段落间静音间隔
SRT_STYLE = "FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Alignment=2,MarginV=40"


# ===================================================================
# 工具函数
# ===================================================================

def run_ffmpeg(args: list, desc: str) -> bool:
    """运行 ffmpeg 命令并检查返回值"""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"] + args
    logger.info(f"[{desc}] ffmpeg {' '.join(cmd[5:])}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"[{desc}] ✅ 完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[{desc}] ❌ 失败: {e}")
        return False


# ===================================================================
# Step 1: 下载抖音视频（如有URL）→ Video A
# ===================================================================

def step1_download(douyin_url: str = None, force: bool = False):
    """下载抖音视频为 Video A，或使用已有文件"""
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


# ===================================================================
# Step 2: 剥离音频 → Audio A + Video B（无声视频）
# ===================================================================

def step2_strip_audio(force: bool = False):
    """从 Video A 提取音频 + 生成无声 Video B"""
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

    # 2b. 生成无声视频 B（去掉音轨）
    if not force and os.path.exists(VIDEO_B):
        logger.info(f"[Step2b] Video B 已存在，跳过: {VIDEO_B}")
    else:
        ok = run_ffmpeg(
            ["-i", VIDEO_A, "-an", "-c:v", "copy", VIDEO_B],
            "Step2b 剥离音频得Video B"
        )
        if not ok:
            # 若 copy codec 失败，尝试重编码
            ok = run_ffmpeg(
                ["-i", VIDEO_A, "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23", VIDEO_B],
                "Step2b 剥离音频得Video B (重编码)"
            )
            if not ok:
                return False
    return True


# ===================================================================
# Step 3: ASR 语音识别
# ===================================================================

def step3_asr(force: bool = False):
    """对 Audio A 进行语音识别"""
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


# ===================================================================
# Step 4: LLM 文案仿写 + 标题 + 标签
# ===================================================================

_LLM_COPY_PROMPT = """你是一个专业的短视频口播文案改写助手。严格遵守以下规则：

【铁律 - 绝对不能违反】
1. 原文讨论什么产品/品牌/话题，你就改写什么，绝对禁止替换成其他产品、品牌或话题
2. 原文提到的人名、地名、产品名、品牌名必须保留，一字不改
3. 绝对禁止自己编造新的产品、品牌或主题

【改写要求】
4. 保留原文所有关键信息和核心卖点
5. 优化语言表达，使文案更口语化、更有感染力，适合TTS语音合成口播
6. 按空行分段输出，每段控制在50-80字
7. 可以调整语序和用词，但不能改变原意和事实

【格式铁律 - 严格禁止以下所有内容】
- 禁止使用任何Markdown语法：**加粗**、*斜体*、# 标题、- 列表、1. 编号列表、> 引用
- 禁止使用英文标点：英文逗号, 英文句号. 英文冒号: 英文分号; 英文括号() 英文引号""
- 禁止使用任何特殊符号：破折号——、省略号……、方括号【】、书名号《》、星号*、下划线_
- 禁止使用 emoji 和特殊 Unicode 字符
- 只允许使用：中文汉字、中文标点（。，！？、：""''）、数字、百分号%、单位（万/亿/元/米/公里）

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


def _llm_generate(engine, system_prompt: str, user_content: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return engine.generate(messages, temperature=temperature, max_tokens=max_tokens)


def _clean_asr_text(text: str) -> str:
    """清洗 ASR 文本：去掉 emoji、特殊符号、多余空白"""
    import re
    # 去掉 emoji 和特殊 Unicode 符号（保留中文、英文、数字、常用标点）
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s.,!?;:()（）、，。！？；：""''【】《》\-/+%=#@&]', '', text)
    # 合并多余空白
    text = re.sub(r'\s+', ' ', text)
    # 去掉多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _clean_copywriting_text(text: str) -> str:
    """清洗 LLM 仿写文案：去掉 Markdown 语法、特殊标点，输出纯文本"""
    import re

    # 1. 去掉 Markdown 加粗/斜体标记
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **加粗** → 加粗
    text = re.sub(r'\*(.+?)\*', r'\1', text)        # *斜体* → 斜体
    text = re.sub(r'__.+?__', '', text)              # __下划线__ → 删除
    text = re.sub(r'~~.+?~~', '', text)              # ~~删除线~~ → 删除

    # 2. 去掉列表标记（行首的 - * · 1. 2. 等）
    text = re.sub(r'^[\s]*[-*·•▪▸►]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+[\.\、\)]\s*', '', text, flags=re.MULTILINE)

    # 3. 去掉 Markdown 标题标记
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # 4. 去掉英文标点替换为中文标点
    text = text.replace(',', '，').replace(';', '；').replace(':', '：')
    text = text.replace('?', '？').replace('!', '！')

    # 5. 去掉多余的特殊符号
    text = re.sub(r'——+', '，', text)          # 破折号 → 逗号
    text = re.sub(r'…{2,}', '，', text)         # 省略号 → 逗号

    # 6. 去掉成对的英文括号/引号
    text = text.replace('"', '').replace('"', '')
    text = text.replace('(', '（').replace(')', '）')

    # 7. 去掉残留 emoji 和非法字符
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s，。！？、：；""''（）%/\-+=#@&\.]', '', text)

    # 8. 清理多余空白和空行
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*\n', '', text)

    return text.strip()


def _split_paragraphs(text: str) -> list:
    """按句号精细分段：每个句号切一段，短句合并"""
    import re

    # 先用 \n\n 保留段落边界
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    segments = []
    for para in paragraphs:
        # 按句末标点切分单句
        raw_sentences = re.split(r'(?<=[。！？])', para)
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        buffer = ''
        for sent in sentences:
            combined = (buffer + sent).strip()
            # 太短则合并，超过 40 字则输出上一段
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


def step4_llm(asr_text: str, force: bool = False):
    """对 ASR 文案进行仿写 + 标题 + 标签生成"""
    if not asr_text:
        logger.error("[Step4] 无 ASR 文本输入")
        return None, None, None

    # 清洗 ASR 文本
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

    # 4c. 标签生成（小模型易重复循环，降低上限 + 异常保护）
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


# ===================================================================
# Step 5: TTS 逐段合成（已改造完毕，含时间戳 SRT）
# ===================================================================

def step5_tts(copy_text: str, force: bool = False):
    """逐段 TTS 合成（含静音间隔 + SRT 时间戳）"""
    if not copy_text:
        logger.error("[Step5] 无仿写文案")
        return None

    if not force and os.path.exists(TTS_AUDIO) and os.path.exists(TTS_SRT):
        logger.info(f"[Step5] TTS 音频和字幕已存在，跳过")
        return TTS_AUDIO

    from local_models.tts_engine import CosyVoiceEngine

    # 清洗文案：去特殊标点
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

    # 写 SRT 字幕
    from _run_tts import _srt_time, save_srt
    save_srt(timeline, TTS_SRT)

    # 打印时间轴
    logger.info(f"[Step5] TTS 完成: {audio_path}")
    for item in timeline:
        logger.info(f"  #{item['index']:2d} [{_srt_time(item['start_s'])} -> {_srt_time(item['end_s'])}] "
                     f"({item['dur_s']:.1f}s) {item['text'][:50]}...")

    engine.unload()
    return audio_path


# ===================================================================
# Step 6: 合成视频C（Video B + TTS 音频 + 字幕）
# ===================================================================

def step6_composite(force: bool = False):
    """用 ffmpeg 将无声视频 B + TTS 音频 + SRT 字幕 = 视频 C"""
    if not os.path.exists(VIDEO_B):
        logger.error("[Step6] Video B 不存在，请先运行 Step2")
        return False
    if not os.path.exists(TTS_AUDIO):
        logger.error("[Step6] TTS 音频不存在，请先运行 Step5")
        return False
    if not os.path.exists(TTS_SRT):
        logger.error("[Step6] SRT 字幕不存在，请先运行 Step5")
        return False

    if not force and os.path.exists(VIDEO_C):
        logger.info(f"[Step6] Video C 已存在，跳过: {VIDEO_C}")
        return True

    logger.info("[Step6] 合成 Video C (Video B + TTS音频 + SRT字幕)...")

    # ffmpeg subtitles 滤镜需要绝对路径（Windows 下路径分隔符需转义）
    srt_path = TTS_SRT.replace("\\", "/").replace(":", "\\:")

    filter_complex = (
        f"[0:v]subtitles='{srt_path}':force_style='{SRT_STYLE}'[v]"
    )

    ok = run_ffmpeg(
        [
            "-i", VIDEO_B,
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
            VIDEO_C,
        ],
        "Step6 合成Video C"
    )

    if ok and os.path.exists(VIDEO_C):
        size_mb = os.path.getsize(VIDEO_C) / (1024 * 1024)
        logger.info(f"[Step6] ✅ Video C 生成: {VIDEO_C} ({size_mb:.1f} MB)")
        return True
    logger.error("[Step6] Video C 合成失败")
    return False


# ===================================================================
# 主流程
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="IP Agent 完整管道")
    parser.add_argument("--url", type=str, default=None, help="抖音视频链接")
    parser.add_argument("--force", action="store_true", help="强制重跑所有步骤")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载（使用已有 Video A）")
    parser.add_argument("--skip-asr", action="store_true", help="跳过 ASR（使用已有结果）")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 LLM（使用已有结果）")
    parser.add_argument("--skip-tts", action="store_true", help="跳过 TTS（使用已有结果）")
    args = parser.parse_args()

    force = args.force

    print("\n" + "=" * 60)
    print("🚀 IP Agent 端到端管道")
    print("=" * 60)

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
        if os.path.exists(LLM_TITLES):
            with open(LLM_TITLES, "r", encoding="utf-8") as f:
                titles = f.read().strip()
        if os.path.exists(LLM_TAGS):
            with open(LLM_TAGS, "r", encoding="utf-8") as f:
                tags = f.read().strip()

    # 打印 LLM 输出
    print("\n" + "-" * 40)
    print("📝 仿写文案预览:")
    print("-" * 40)
    print(copy_text[:300] + ("..." if len(copy_text) > 300 else ""))
    if titles:
        print(f"\n🏷 标题:\n{titles}")
    if tags:
        print(f"\n🔖 标签:\n{tags}")
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

    # ── Step 6: 合成视频C ──
    if not step6_composite(force=force):
        return 1

    # ── 完成 ──
    print("\n" + "=" * 60)
    print("✅ 管道完成！")
    print(f"   Video A (原始):  {VIDEO_A}")
    print(f"   Video B (无声):  {VIDEO_B}")
    print(f"   ASR 文本:        {ASR_TEXT}")
    print(f"   仿写文案:        {LLM_COPY}")
    print(f"   标题:            {LLM_TITLES}")
    print(f"   标签:            {LLM_TAGS}")
    print(f"   TTS 音频:        {TTS_AUDIO}")
    print(f"   字幕 SRT:        {TTS_SRT}")
    print(f"   🎬 Video C:      {VIDEO_C}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
