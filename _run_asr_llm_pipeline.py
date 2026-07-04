"""
完整流水线：ASR 识别 → LLM 文案仿写 + 标题生成 + 标签生成
所有结果自动存档到 test_downloads/ 目录
"""
import os, sys, logging, datetime

# 修复 Windows GBK 编码下 emoji 打印报错问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from local_models.asr_engine import ASREngine
from local_models.llm_engine import TransformersLLM

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
BASE = os.path.dirname(__file__)
AUDIO_PATH = os.path.join(BASE, "local_models", "test_downloads", "douyin_audio.wav")
OUTPUT_DIR = os.path.join(BASE, "local_models", "test_downloads")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 严格 System Prompt
# ---------------------------------------------------------------------------

SYSTEM_COPYWRITING = """你是一个专业的短视频口播文案改写专家。请严格按以下规则改写文案：
1. 保留原文核心信息、关键卖点、数据、专业术语不做改动
2. 优化语言表达，使文案更口语化、更有感染力、更适合口播节奏
3. 保持原文的段落和叙述顺序，不要添加原文没有的信息
4. 可调整语序和用词，但不能改变原意，不能编造内容
5. 输出只包含改写后的文案全文，不要加任何前缀、后缀、解释或标记"""

SYSTEM_TITLE = """你是一个专业的短视频标题创作专家。请严格按以下规则生成标题：
1. 基于文案内容提取核心卖点，生成恰好3个标题
2. 标题要有爆点，能用疑问/感叹/悬念吸引点击
3. 每个标题15-25字，使用中文，不要带任何英文
4. 输出格式必须严格遵守（不要加任何额外文字）：
1. 标题一
2. 标题二
3. 标题三"""

SYSTEM_TAGS = """你是一个专业的抖音话题标签运营专家。请严格按以下规则生成标签：
1. 从文案中提取核心关键词，生成恰好10个标签，不要多也不要少
2. 前5个为流量大词（车型/品牌/行业），后5个为精准长尾词（具体卖点/场景）
3. 每个标签以#开头，纯中文，不使用英文或数字
4. 所有标签用英文逗号+空格分隔，写在一行内
5. 输出格式严格遵守（不要加任何额外文字、解释或换行）：
#标签一, #标签二, #标签三, #标签四, #标签五, #标签六, #标签七, #标签八, #标签九, #标签十"""


# ===================== Step 1: ASR 识别 =====================
logger.info("=" * 60)
logger.info("Step 1: ASR 语音识别")
logger.info("=" * 60)

asr = ASREngine()
if not asr.load():
    logger.error("ASR 模型加载失败")
    sys.exit(1)

original_text = asr.transcribe(AUDIO_PATH)
asr.unload()

if not original_text:
    logger.error("ASR 识别结果为空")
    sys.exit(1)

# 保存 ASR 结果
asr_file = os.path.join(OUTPUT_DIR, "asr_result.txt")
with open(asr_file, "w", encoding="utf-8") as f:
    f.write(original_text)
logger.info(f"[存档] ASR 结果 -> {asr_file}")
logger.info(f"原始文案 ({len(original_text)}字符)")

# ===================== Step 2: LLM 三项任务 =====================
logger.info("\n" + "=" * 60)
logger.info("Step 2: LLM 加载模型")
logger.info("=" * 60)

engine = TransformersLLM()
if not engine.load():
    logger.error("LLM 模型加载失败")
    sys.exit(1)


def llm_generate(system_prompt: str, user_prompt: str,
                 temperature: float = 0.7, max_tokens: int = 1024) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return engine.generate(messages, temperature=temperature, max_tokens=max_tokens)


def _clean_tags(raw_tags: str, max_tags: int = 10) -> list:
    """清洗 LLM 生成的标签：去重、去无效、截取前N个"""
    seen = set()
    cleaned = []
    for tag in raw_tags.replace("\n", ",").split(","):
        tag = tag.strip().strip("#").strip()
        if not tag or len(tag) < 2:
            continue
        # 过滤纯数字和纯标点
        if tag.isdigit() or all(c in ".,;:：；，。、" for c in tag):
            continue
        tag_normalized = f"#{tag}"
        if tag not in seen and len(tag_normalized) <= 20:
            seen.add(tag)
            cleaned.append(tag_normalized)
        if len(cleaned) >= max_tags:
            break
    return cleaned


# ---------- 任务 2a：文案仿写 ----------
logger.info("\n--- 任务 2a: 文案仿写 ---")
rewritten = llm_generate(
    SYSTEM_COPYWRITING,
    f"请改写以下口播文案：\n\n{original_text}",
    temperature=0.7, max_tokens=2048,
)
copywriting_file = os.path.join(OUTPUT_DIR, "llm_copywriting.txt")
with open(copywriting_file, "w", encoding="utf-8") as f:
    f.write(rewritten)
logger.info(f"[存档] 仿写文案 -> {copywriting_file}")
logger.info(f"仿写完成 ({len(rewritten)}字符)")

# ---------- 任务 2b：标题生成 ----------
logger.info("\n--- 任务 2b: 标题生成 ---")
titles = llm_generate(
    SYSTEM_TITLE,
    f"请根据以下口播文案生成恰好3个标题：\n\n{original_text}",
    temperature=0.7, max_tokens=256,
)
title_file = os.path.join(OUTPUT_DIR, "llm_titles.txt")
with open(title_file, "w", encoding="utf-8") as f:
    f.write(titles)
logger.info(f"[存档] 推荐标题 -> {title_file}")

# ---------- 任务 2c：标签生成 ----------
logger.info("\n--- 任务 2c: 标签生成 ---")
tags_raw = llm_generate(
    SYSTEM_TAGS,
    f"请根据以下口播文案生成恰好10个抖音话题标签：\n\n{original_text}",
    temperature=0.5, max_tokens=128,
)

# 后处理：去重、截取前10个有效标签（0.5B 小模型容易产生重复幻觉）
tags_cleaned = _clean_tags(tags_raw)
tags = ", ".join(tags_cleaned) if tags_cleaned else tags_raw

tags_file = os.path.join(OUTPUT_DIR, "llm_tags.txt")
with open(tags_file, "w", encoding="utf-8") as f:
    f.write(tags)
logger.info(f"[存档] 推荐标签 -> {tags_file} ({len(tags_cleaned)}个有效标签)")

# ---------- 卸载模型 ----------
engine.unload()

# ===================== 汇总输出（纯 ASCII 安全） =====================
sep = "-" * 50
print(f"\n{sep}")
print("[ASR 原始文案]")
print(sep)
print(original_text[:200] + "..." if len(original_text) > 200 else original_text)

print(f"\n{sep}")
print("[LLM 仿写文案]")
print(sep)
print(rewritten[:300] + "..." if len(rewritten) > 300 else rewritten)

print(f"\n{sep}")
print("[LLM 推荐标题]")
print(sep)
print(titles)

print(f"\n{sep}")
print("[LLM 推荐标签]")
print(sep)
print(tags)

print(f"\n{sep}")
print("[存档文件列表]")
print(sep)
for fn in ["asr_result.txt", "llm_copywriting.txt", "llm_titles.txt", "llm_tags.txt"]:
    fp = os.path.join(OUTPUT_DIR, fn)
    if os.path.exists(fp):
        kb = os.path.getsize(fp) / 1024
        print(f"  {fn} ({kb:.1f}KB)")
