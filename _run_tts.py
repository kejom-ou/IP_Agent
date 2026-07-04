"""
TTS 语音合成：基于 LLM 仿写文案，逐段生成音频 + 段落间静音间隔 + ffmpeg 时间戳
"""
import os, sys, logging, re

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from local_models.tts_engine import CosyVoiceEngine

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "local_models", "test_downloads")
COPYWRITING_FILE = os.path.join(OUTPUT_DIR, "llm_copywriting.txt")
OUTPUT_AUDIO = os.path.join(OUTPUT_DIR, "tts_output.wav")
OUTPUT_SRT = os.path.join(OUTPUT_DIR, "tts_timeline.srt")
OUTPUT_SEGMENTS = os.path.join(OUTPUT_DIR, "tts_segments.txt")

# 段落间静音间隔（毫秒），模拟真人说话的停顿
GAP_MS = 500


def split_paragraphs(text: str) -> list:
    """按句号精细分段：每个句号切一段，短句合并，保持字幕简洁"""
    # 先用 \n\n 保留段落边界
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    segments = []
    for para in paragraphs:
        # 按句末标点切分单句
        raw_sentences = re.split(r'(?<=[。！？])', para)
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        buffer = ''
        for sent in sentences:
            combined = (buffer + sent).strip()
            # 太短则合并，超过 50 字则输出上一段
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


def _srt_time(seconds: float) -> str:
    """将秒数转为 SRT 时间格式 HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_srt(timeline: list, filepath: str):
    """保存 SRT 字幕文件（ffmpeg 可直接读取）。"""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in timeline:
            f.write(f"{item['index']}\n")
            f.write(f"{_srt_time(item['start_s'])} --> {_srt_time(item['end_s'])}\n")
            f.write(f"{item['text']}\n\n")
    logger.info(f"[存档] SRT 字幕 -> {filepath}")


def save_segments_txt(timeline: list, filepath: str, gap_ms: int):
    """保存 ffmpeg segments 格式文件（可用于逐段处理）。"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# 段落间隔: {gap_ms}ms\n")
        f.write(f"# 格式: start_ms | end_ms | duration_ms | text\n\n")
        for item in timeline:
            s_ms = int(item['start_s'] * 1000)
            e_ms = int(item['end_s'] * 1000)
            d_ms = int(item['dur_s'] * 1000)
            f.write(f"[{s_ms:>8}ms - {e_ms:>8}ms]  ({d_ms:>5}ms)  {item['text'][:60]}\n")
    logger.info(f"[存档] 段落时间戳 -> {filepath}")


def print_timeline(timeline: list):
    """控制台输出时间轴一览。"""
    print("\n" + "=" * 70)
    print("📋 音频段落时间轴（严格对齐，间隔 {}ms）".format(GAP_MS))
    print("=" * 70)
    for item in timeline:
        print(f"  #{item['index']:2d}  [{_srt_time(item['start_s'])} --> {_srt_time(item['end_s'])}]  "
              f"({item['dur_s']:.2f}s)")
        # 截断显示文本
        txt_preview = item['text'][:70] + "..." if len(item['text']) > 70 else item['text']
        print(f"      {txt_preview}")
    print("=" * 70)


def main():
    # 1. 读取仿写文案
    if not os.path.exists(COPYWRITING_FILE):
        logger.error(f"文案文件不存在: {COPYWRITING_FILE}")
        sys.exit(1)

    with open(COPYWRITING_FILE, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        logger.error("文案为空")
        sys.exit(1)

    # 2. 按段落拆分（避免整段丢给 TTS 导致 OOM）
    segments = split_paragraphs(text)
    logger.info(f"文案共 {len(text)} 字符，拆分为 {len(segments)} 段")
    for i, s in enumerate(segments):
        logger.info(f"  段[{i+1}] {len(s)}字: {s[:60]}...")

    # 3. 逐段 TTS 合成（含静音间隔 + 时间轴）
    engine = CosyVoiceEngine()
    logger.info("开始逐段语音合成...")
    result = engine.synthesize_segments(
        segments=segments,
        speaker="中文女",
        speed=1.0,
        gap_ms=GAP_MS,
        output_path=OUTPUT_AUDIO,
    )

    if result is None:
        logger.error("❌ 语音合成失败")
        sys.exit(1)

    audio_path, timeline = result

    # 4. 打印时间轴
    print_timeline(timeline)

    # 5. 存档时间戳文件
    save_srt(timeline, OUTPUT_SRT)
    save_segments_txt(timeline, OUTPUT_SEGMENTS, GAP_MS)

    engine.unload()
    logger.info("TTS 引擎已卸载")
    logger.info(f"✅ 全部完成！音频: {audio_path}")


if __name__ == "__main__":
    main()
