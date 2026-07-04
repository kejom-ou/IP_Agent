"""测试纯 HTTP 爬虫下载抖音视频音频"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_models.douyin_crawler import DouyinCrawler

TEST_URL = "https://v.douyin.com/TqiTGubKDMI/"

print("=" * 60)
print("抖音音频下载测试（纯 HTTP，无登录，无浏览器）")
print("=" * 60)

t0 = time.time()
crawler = DouyinCrawler()

# 测试音频下载
print("\n>>> download_audio()")
path = crawler.download_audio(TEST_URL)

if path:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\n[OK] 音频下载成功 ({time.time() - t0:.1f}s)")
    print(f"     文件: {path} ({size_mb:.1f}MB)")
    
    # 确认视频临时文件已被清理
    tmp_video = os.path.join(os.path.dirname(path), "_dl_stream.tmp")
    if os.path.exists(tmp_video):
        print(f"     [WARN] 视频临时文件未清理: {tmp_video}")
    else:
        print(f"     [OK] 视频临时文件已清理")
else:
    print(f"\n[FAIL] 音频下载失败 ({time.time() - t0:.1f}s)")
    sys.exit(1)
