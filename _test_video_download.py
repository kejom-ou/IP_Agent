"""
测试视频下载功能 — Playwright 拦截音视频流 + 合并 + ASR
抖音音视频分离，需同时捕获 video 和 audio 流后合并
"""
import asyncio
import os
import subprocess
import sys
import time
import requests

DOUYIN_URL = "https://v.douyin.com/TqiTGubKDMI/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "local_models", "test_downloads")
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_ONLY = os.path.join(OUTPUT_DIR, "douyin_video_only.mp4")
AUDIO_ONLY = os.path.join(OUTPUT_DIR, "douyin_audio_only.mp4")
VIDEO_MERGED = os.path.join(OUTPUT_DIR, "douyin_video.mp4")
AUDIO_WAV = os.path.join(OUTPUT_DIR, "douyin_audio.wav")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Referer": "https://www.douyin.com/",
}


def print_step(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def http_download(url: str, save_path: str, label: str = "") -> bool:
    """HTTP 流式下载"""
    print(f"  下载{label}: {url[:120]}...")
    try:
        resp = requests.get(url, headers=HEADERS, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        t0 = time.time()
        last_pct = -1
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded / total * 100)
                    if pct - last_pct >= 10:
                        speed = downloaded / max(time.time() - t0, 1) / 1024 / 1024
                        print(f"    {pct}% ({downloaded/1024/1024:.1f}/{total/1024/1024:.1f}MB, {speed:.1f}MB/s)")
                        last_pct = pct
        elapsed = time.time() - t0
        size_mb = os.path.getsize(save_path) / 1024 / 1024
        print(f"  OK {label}: {size_mb:.1f}MB ({elapsed:.1f}s)")
        return True
    except Exception as e:
        print(f"  下载{label}失败: {e}")
        return False


async def capture_douyin_media() -> tuple:
    """
    用 Playwright 打开抖音视频页，拦截音视频 CDN 地址。
    返回 (video_url, audio_url)
    """
    from playwright.async_api import async_playwright

    print("  启动 Playwright Chromium...")
    video_url = None
    audio_url = None

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/132.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        captured = []

        def on_response(response):
            url = response.url
            if "douyinvod.com" in url:
                captured.append(url)

        page.on("response", on_response)

        # 访问抖音首页
        print("  打开抖音首页...")
        try:
            await page.goto("https://www.douyin.com", wait_until="domcontentloaded", timeout=30000)
        except Exception:
            pass
        await asyncio.sleep(3)

        print("\n  " + "=" * 50)
        print("  请在浏览器中操作（如需登录请扫码）")
        print("  " + "=" * 50)

        # 访问视频页面
        print(f"\n  访问视频页面...")
        try:
            await page.goto(DOUYIN_URL, wait_until="domcontentloaded", timeout=30000)
        except Exception:
            pass

        # 等待视频加载
        print("  等待视频加载...")
        await asyncio.sleep(5)

        # 尝试滚动/点击触发加载
        try:
            await page.click("video", timeout=3000)
            await asyncio.sleep(3)
        except Exception:
            pass

        # 分析捕获的 URL
        for url in captured:
            if not video_url and "media-video" in url:
                video_url = url
                print(f"  [CDN Video] {url[:120]}")
            if not audio_url and "media-audio" in url:
                audio_url = url
                print(f"  [CDN Audio] {url[:120]}")

        await browser.close()

    return video_url, audio_url


async def main():
    # Step 1: 捕获音视频 CDN 地址
    print_step("Step 1/4: Playwright 拦截抖音 CDN 音视频地址")
    print("  Playwright 浏览器将打开，请在浏览器中操作\n")

    video_url, audio_url = await capture_douyin_media()

    if not video_url:
        print("  未获取到视频地址！")
        sys.exit(1)

    # Step 2: 下载视频流和音频流
    print_step("Step 2/4: HTTP 下载音视频流")

    if not http_download(video_url, VIDEO_ONLY, "视频流"):
        sys.exit(1)

    if audio_url:
        if not http_download(audio_url, AUDIO_ONLY, "音频流"):
            print("  音频流下载失败，尝试无音频继续...")
            audio_url = None

    # Step 3: ffmpeg 合并音视频
    print_step("Step 3/4: ffmpeg 合并音视频")

    if audio_url and os.path.exists(AUDIO_ONLY):
        cmd_merge = [
            "ffmpeg", "-i", VIDEO_ONLY, "-i", AUDIO_ONLY,
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", VIDEO_MERGED, "-y",
        ]
    else:
        # 视频已经包含音频，直接复制
        import shutil
        shutil.copy(VIDEO_ONLY, VIDEO_MERGED)

    if audio_url and os.path.exists(AUDIO_ONLY):
        try:
            result = subprocess.run(cmd_merge, check=True, capture_output=True, text=True, timeout=60)
            elapsed = time.time()
            for line in result.stderr.strip().split('\n')[-3:]:
                if "map" in line.lower() or "stream" in line.lower():
                    print(f"    {line}")
            print(f"  合并完成: {os.path.getsize(VIDEO_MERGED)/1024/1024:.1f}MB")
        except subprocess.CalledProcessError as e:
            print(f"  合并失败: {e.stderr[-300:]}")
            # 如果合并失败，尝试只复制视频
            import shutil
            shutil.copy(VIDEO_ONLY, VIDEO_MERGED)
            print("  使用纯视频继续...")
    else:
        print(f"  无独立音频流，直接使用视频: {os.path.getsize(VIDEO_MERGED)/1024/1024:.1f}MB")

    # Step 4: 提取 WAV 音频 + ASR
    print_step("Step 4/4: ffmpeg 提取音频 + ASR 转写")

    # 提取音频
    cmd_extract = [
        "ffmpeg", "-i", VIDEO_MERGED, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", AUDIO_WAV, "-y",
    ]
    try:
        subprocess.run(cmd_extract, check=True, capture_output=True, text=True, timeout=60)
    except subprocess.CalledProcessError as e:
        print(f"  音频提取失败: {e.stderr[-200:]}")
        sys.exit(1)

    if os.path.exists(AUDIO_WAV):
        size_mb = os.path.getsize(AUDIO_WAV) / (1024 * 1024)
        dur = os.path.getsize(AUDIO_WAV) / (16000 * 2)
        print(f"  音频: {AUDIO_WAV} ({size_mb:.2f}MB, ~{dur:.0f}s)")
    else:
        print("  音频文件不存在")
        sys.exit(1)

    # ASR
    print("\n  加载 ASR 模型 (SenseVoiceSmall)...")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    try:
        from local_models.asr_engine import ASREngine
        asr = ASREngine()
        t0 = time.time()
        if not asr.load():
            print("  ASR 加载失败")
            sys.exit(1)
        print(f"  ASR 加载完成 ({time.time()-t0:.1f}s)")

        print("  正在转写...")
        t0 = time.time()
        text = asr.transcribe(AUDIO_WAV)
        elapsed = time.time() - t0
        asr.unload()

        print(f"\n  转写完成 ({elapsed:.1f}s)")

        # 保存完整结果到 UTF-8 文件
        result_file = os.path.join(OUTPUT_DIR, "asr_result.txt")
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  [完整结果已保存: {result_file}]")
        print(f"  字数: {len(text)} 字")

        # GBK 安全预览
        safe_text = text.encode("gbk", errors="replace").decode("gbk", errors="replace")
        print(f"\n  {'─'*50}")
        print(f"  [ASR 结果预览 (前 200 字)]")
        print(f"  {safe_text[:200]}")
        if len(safe_text) > 200:
            print(f"  ... (完整结果见 asr_result.txt)")
        print(f"  {'─'*50}")
    except Exception as e:
        print(f"  ASR 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  全流程测试通过!")
    print(f"  合并视频: {VIDEO_MERGED}")
    print(f"  音频 WAV: {AUDIO_WAV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
