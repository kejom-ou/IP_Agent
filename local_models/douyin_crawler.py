"""
抖音视频爬虫 — 纯 HTTP 爬虫模式（无需登录、无需浏览器、无需 Cookie）

工作流程：
  1. 解析短链接 → 获取 video_id
  2. 请求 iesdouyin 分享页 → 提取 SSR 中的 _ROUTER_DATA
  3. 解析 _ROUTER_DATA → 获取无水印 CDN 播放地址
  4. HTTP 流式下载 → ffmpeg 合并

特点：
  - 纯 HTTP 请求，不依赖浏览器
  - 不需要登录，不需要保存 Cookie
  - 每次请求都是无状态的
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# 移动端 UA 是获取 SSR 数据的关键
MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/16.0 Mobile/15E148 Safari/604.1"
)

PC_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/132.0.0.0 Safari/537.36"
)


class DouyinCrawler:
    """
    抖音视频爬虫 — 纯 HTTP 模式

    无需登录，无需浏览器，无需任何持久化。每次下载从头开始抓取。

    使用示例:
        crawler = DouyinCrawler()
        path = crawler.download("https://v.douyin.com/xxx/")
    """

    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        self._session = requests.Session()
        self._session.max_redirects = 10

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def download(
        self,
        video_url: str,
        output_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        下载抖音视频（纯 HTTP 爬虫）

        Args:
            video_url: 抖音分享链接（如 https://v.douyin.com/xxx/）
            output_dir: 输出目录，默认当前目录下的 test_downloads/

        Returns:
            下载后的视频文件路径，失败返回 None
        """
        _setup_logging()

        output_dir = output_dir or str(Path(__file__).resolve().parent / "test_downloads")
        os.makedirs(output_dir, exist_ok=True)

        stream_file = os.path.join(output_dir, "_dl_stream.tmp")
        output_file = os.path.join(output_dir, "douyin_video.mp4")

        # 下载视频流
        url = self._resolve_and_get_url(video_url)
        if not url or not self._http_download_video(url, stream_file):
            return None

        os.replace(stream_file, output_file)
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"下载完成: {output_file} ({size_mb:.1f}MB)")
        return output_file

    def download_audio(
        self,
        video_url: str,
        output_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        下载抖音视频的音频（纯 HTTP 爬虫 → ffmpeg 提取音频）

        流程：下载视频流 → ffmpeg 提取音频 → 删除临时视频

        Args:
            video_url: 抖音分享链接（如 https://v.douyin.com/xxx/）
            output_dir: 输出目录，默认当前目录下的 test_downloads/

        Returns:
            提取后的音频文件路径（WAV 格式），失败返回 None
        """
        _setup_logging()

        output_dir = output_dir or str(Path(__file__).resolve().parent / "test_downloads")
        os.makedirs(output_dir, exist_ok=True)

        video_tmp = os.path.join(output_dir, "_dl_stream.tmp")
        audio_file = os.path.join(output_dir, "douyin_audio.wav")

        # Step 1: 下载视频流
        logger.info("Step 1/3: 下载视频流...")
        url = self._resolve_and_get_url(video_url)
        if not url or not self._http_download_video(url, video_tmp):
            return None

        # Step 2: ffmpeg 提取音频
        logger.info("Step 2/3: ffmpeg 提取音频...")
        if not self._extract_audio(video_tmp, audio_file):
            # 清理视频临时文件
            if os.path.exists(video_tmp):
                os.remove(video_tmp)
            return None

        # Step 3: 删除视频临时文件
        logger.info("Step 3/3: 清理视频临时文件...")
        if os.path.exists(video_tmp):
            os.remove(video_tmp)

        size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        logger.info(f"音频提取完成: {audio_file} ({size_mb:.1f}MB)")
        return audio_file

    # ------------------------------------------------------------------
    # 核心逻辑
    # ------------------------------------------------------------------

    def _resolve_and_get_url(self, video_url: str) -> Optional[str]:
        """完整的 URL 解析流程：短链 → video_id → SSR → 播放地址"""
        video_id = self._resolve_share_url(video_url)
        if not video_id:
            logger.error("无法解析视频 ID")
            return None
        logger.info(f"  Video ID: {video_id}")

        video_info = self._extract_video_info(video_id)
        if not video_info:
            logger.error("无法提取视频信息")
            return None

        logger.info(f"  作者: {video_info.get('author', '?')}")
        logger.info(f"  描述: {video_info.get('desc', '')[:60]}...")

        play_url = video_info.get("play_url")
        if not play_url:
            logger.error("未找到可下载的视频地址")
            return None

        best_url = self._get_best_quality_url(play_url, video_info.get("bit_rates", []))
        logger.info(f"  视频URL: {best_url[:100]}...")
        return best_url

    def _resolve_share_url(self, share_url: str) -> Optional[str]:
        """解析抖音短链接 → 获取 video_id"""
        try:
            r = requests.get(
                share_url,
                headers={"User-Agent": MOBILE_UA},
                allow_redirects=True,
                timeout=15,
            )
            final_url = r.url

            # 尝试多种 video_id 格式
            for pattern in [
                r'/video/(\d+)',
                r'/note/(\d+)',
                r'modal_id=(\d+)',
                r'item_id=(\d+)',
                r'/share/video/(\d+)',
            ]:
                m = re.search(pattern, final_url)
                if m:
                    return m.group(1)

            logger.warning(f"无法从URL提取video_id: {final_url}")
            return None

        except Exception as e:
            logger.error(f"短链接解析失败: {e}")
            return None

    def _extract_video_info(self, video_id: str) -> Optional[dict]:
        """
        从 iesdouyin 分享页的 SSR 数据中提取视频信息。

        抖音分享页（用移动端 UA 访问）会在 HTML 中嵌入 _ROUTER_DATA，
        包含完整的视频信息，包括播放地址。
        """
        share_url = f"https://www.iesdouyin.com/share/video/{video_id}/"
        headers = {
            "User-Agent": MOBILE_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

        try:
            r = requests.get(share_url, headers=headers, timeout=15)
            r.raise_for_status()
        except Exception as e:
            logger.error(f"分享页请求失败: {e}")
            return None

        html = r.text

        # 保存页面 cookies 用于后续下载（非登录态，只是 session）
        self._page_cookies = r.cookies

        # 提取 _ROUTER_DATA
        m = re.search(r'window\._ROUTER_DATA\s*=\s*({.*?});?\s*</script>', html, re.DOTALL)
        if not m:
            logger.error("页面中未找到 _ROUTER_DATA")
            return None

        try:
            raw = m.group(1)
            decoded = urllib.parse.unquote(raw)
            data = json.loads(decoded)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"_ROUTER_DATA 解析失败: {e}")
            return None

        # 导航到视频数据路径
        try:
            loader = data.get("loaderData", {})
            video_page = loader.get("video_(id)/page", {})
            video_res = video_page.get("videoInfoRes", {})
            item_list = video_res.get("item_list", [])

            if not item_list:
                logger.error("item_list 为空")
                return None

            item = item_list[0]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"数据结构异常: {e}")
            return None

        # 提取视频信息
        video_data = item.get("video", {})
        author_data = item.get("author", {})

        result = {
            "desc": item.get("desc", ""),
            "author": author_data.get("nickname", ""),
            "duration": video_data.get("duration", 0),
            "play_url": self._extract_best_url(video_data),
            "bit_rates": self._extract_bit_rates(video_data),
        }

        return result

    def _extract_best_url(self, video_data: dict) -> Optional[str]:
        """从视频数据中提取最佳下载地址"""
        # 优先级: bit_rate 高清 > play_addr
        bit_rates = video_data.get("bit_rate", [])
        if bit_rates:
            # 取码率最高的
            best = max(bit_rates, key=lambda x: x.get("bit_rate", 0))
            play_addr = best.get("play_addr", {})
            urls = play_addr.get("url_list", [])
            if urls:
                return self._clean_url(urls[0])

        # 回退到普通播放地址
        play_addr = video_data.get("play_addr", {})
        urls = play_addr.get("url_list", [])
        if urls:
            return self._clean_url(urls[0])

        # 再尝试无水印下载地址
        download_addr = video_data.get("download_addr", {})
        urls = download_addr.get("url_list", [])
        if urls:
            return urls[0]

        return None

    def _extract_bit_rates(self, video_data: dict) -> list:
        """提取多码率信息"""
        result = []
        for br in (video_data.get("bit_rate") or []):
            play_addr = br.get("play_addr", {})
            urls = play_addr.get("url_list", [])
            if urls:
                result.append({
                    "gear_name": br.get("gear_name", ""),
                    "bit_rate": br.get("bit_rate", 0),
                    "url": urls[0],
                })
        return result

    def _get_best_quality_url(self, default_url: str, bit_rates: list) -> str:
        """选择最高画质URL"""
        if not bit_rates:
            return default_url
        # 已按码率排序
        best = max(bit_rates, key=lambda x: x.get("bit_rate", 0))
        url = best.get("url", "")
        return self._clean_url(url) if url else default_url

    @staticmethod
    def _clean_url(url: str) -> str:
        """清理视频 URL，尝试去除水印"""
        # 将 playwm（watermark）替换为 play（无水印）
        url = re.sub(r'/playwm/', '/play/', url)
        url = re.sub(r'watermark=1', 'watermark=0', url)
        return url

    # ------------------------------------------------------------------
    # HTTP 多线程分块下载
    # ------------------------------------------------------------------

    # 并行下载线程数（CDN 出口带宽一般 4-8 线程即可跑满）
    DOWNLOAD_WORKERS = 8
    # 每块最小 512KB，避免分块过碎
    MIN_CHUNK_SIZE = 512 * 1024

    def _http_download_video(self, url: str, save_path: str) -> bool:
        """多线程分块下载视频（基于 HTTP Range）"""
        t0 = time.time()

        try:
            # Step 1: 探测文件总大小 & 是否支持 Range
            total_size, supports_range = self._probe_url(url)
            if total_size is None:
                logger.warning("无法获取文件大小，回退到单线程流式下载")
                return self._http_download_single(url, save_path)

            if not supports_range:
                logger.warning("CDN 不支持 Range 请求，使用单线程流式下载")
                return self._http_download_single(url, save_path)

            # Step 2: 分块下载
            chunk_size = max(
                self.MIN_CHUNK_SIZE,
                total_size // self.DOWNLOAD_WORKERS,
            )
            # 计算分块边界，覆盖整个文件
            ranges = []
            start = 0
            while start < total_size:
                end = min(start + chunk_size - 1, total_size - 1)
                ranges.append((start, end))
                start = end + 1

            # 如果分块太少或只有一个，直接用单线程
            if len(ranges) <= 1:
                return self._http_download_single(url, save_path)

            logger.info(
                f"  [多线程] 文件大小: {total_size / 1024 / 1024:.1f}MB, "
                f"分 {len(ranges)} 块, {self.DOWNLOAD_WORKERS} 线程并行下载"
            )

            # Step 3: 并行下载各分块
            chunk_files = []
            futures = {}
            with ThreadPoolExecutor(max_workers=self.DOWNLOAD_WORKERS) as executor:
                for idx, (range_start, range_end) in enumerate(ranges):
                    tmp_path = save_path + f".part{idx:03d}"
                    chunk_files.append((idx, range_start, tmp_path))
                    futures[
                        executor.submit(
                            self._download_chunk,
                            url, range_start, range_end, tmp_path, idx,
                        )
                    ] = idx

                last_progress = -5  # 每 5% 打印一次
                downloaded = 0
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        ok, size = future.result()
                        if not ok:
                            logger.error(f"  [块 {idx}] 下载失败，终止")
                            # 清理临时文件
                            for _, _, fp in chunk_files:
                                if os.path.exists(fp):
                                    os.remove(fp)
                            return False
                        downloaded += size
                        progress = int(downloaded / total_size * 100)
                        if progress - last_progress >= 5:
                            logger.info(
                                f"  [进度] {progress}% ({downloaded / 1024 / 1024:.1f}/"
                                f"{total_size / 1024 / 1024:.1f}MB)"
                            )
                            last_progress = progress
                    except Exception as e:
                        logger.error(f"  [块 {idx}] 异常: {e}")
                        for _, _, fp in chunk_files:
                            if os.path.exists(fp):
                                os.remove(fp)
                        return False

            # Step 4: 按顺序合并分块
            logger.info("  [合并] 拼接各分块...")
            with open(save_path, "wb") as out:
                for _, _, tmp_path in sorted(chunk_files, key=lambda x: x[0]):
                    with open(tmp_path, "rb") as inp:
                        while True:
                            data = inp.read(4 * 1024 * 1024)  # 4MB 读取
                            if not data:
                                break
                            out.write(data)
                    os.remove(tmp_path)  # 合并完立即删除临时文件

            elapsed = time.time() - t0
            size_mb = total_size / (1024 * 1024)
            speed = size_mb / elapsed if elapsed > 0 else 0
            logger.info(f"  [OK] 下载完成: {size_mb:.1f}MB ({elapsed:.1f}s, {speed:.1f}MB/s)")
            return True

        except Exception as e:
            logger.error(f"多线程下载异常: {e}", exc_info=True)
            # 清理可能遗留的临时文件
            for f in Path(save_path).parent.glob(Path(save_path).name + ".part*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            return False

    def _probe_url(self, url: str) -> Tuple[Optional[int], bool]:
        """探测文件大小和是否支持 Range"""
        headers = {
            "User-Agent": MOBILE_UA,
            "Referer": "https://www.douyin.com/",
            "Range": "bytes=0-0",
        }
        try:
            resp = requests.get(
                url,
                headers=headers,
                cookies=getattr(self, "_page_cookies", None),
                timeout=15,
                allow_redirects=True,
            )
            # 206 = 支持 Range; 200 = 不支持
            supports_range = (resp.status_code == 206)
            content_range = resp.headers.get("Content-Range", "")
            if content_range:
                m = re.search(r"/\s*(\d+)", content_range)
                if m:
                    total = int(m.group(1))
                    return total, True
            content_length = resp.headers.get("Content-Length")
            if content_length:
                return int(content_length), supports_range
            return None, supports_range
        except Exception as e:
            logger.warning(f"探测失败: {e}")
            return None, False

    def _download_chunk(
        self, url: str, range_start: int, range_end: int,
        save_path: str, chunk_idx: int,
    ) -> Tuple[bool, int]:
        """下载单个分块，返回 (成功, 实际字节数)"""
        headers = {
            "User-Agent": MOBILE_UA,
            "Referer": "https://www.douyin.com/",
            "Accept": "*/*",
            "Accept-Encoding": "identity",
            "Range": f"bytes={range_start}-{range_end}",
        }
        try:
            resp = requests.get(
                url,
                headers=headers,
                cookies=getattr(self, "_page_cookies", None),
                stream=True,
                timeout=120,
                allow_redirects=True,
            )
            if resp.status_code not in (200, 206):
                logger.error(f"  [块 {chunk_idx}] HTTP {resp.status_code}")
                return False, 0

            written = 0
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        written += len(chunk)
            return True, written
        except Exception as e:
            logger.error(f"  [块 {chunk_idx}] 下载异常: {e}")
            return False, 0

    def _http_download_single(self, url: str, save_path: str) -> bool:
        """单线程流式下载（CDN 不支持 Range 时的回退方案）"""
        headers = {
            "User-Agent": MOBILE_UA,
            "Referer": "https://www.douyin.com/",
            "Accept": "*/*",
            "Accept-Encoding": "identity",
        }
        try:
            t0 = time.time()
            resp = requests.get(
                url,
                headers=headers,
                cookies=getattr(self, "_page_cookies", None),
                stream=True,
                timeout=120,
                allow_redirects=True,
            )
            if resp.status_code not in (200, 206):
                logger.error(f"下载失败: status={resp.status_code}")
                return False

            content_type = resp.headers.get("content-type", "")
            if "video" not in content_type and resp.status_code != 206:
                logger.warning(f"非视频响应: content-type={content_type}, body={resp.text[:200]}")
                return False

            content_length = resp.headers.get("content-length")
            total_size = int(content_length) if content_length else None

            downloaded = 0
            last_log = 0
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # 每 10% 打印进度
                        if total_size:
                            pct = int(downloaded / total_size * 100)
                            if pct - last_log >= 10:
                                logger.info(f"  [进度] {pct}%")
                                last_log = pct

            elapsed = time.time() - t0
            size_mb = downloaded / (1024 * 1024)
            speed = size_mb / elapsed if elapsed > 0 else 0
            logger.info(f"  [OK] 下载完成: {size_mb:.1f}MB ({elapsed:.1f}s, {speed:.1f}MB/s)")
            return True

        except Exception as e:
            logger.error(f"下载异常: {e}")
            return False

    @staticmethod
    def _extract_audio(video_path: str, audio_path: str) -> bool:
        """使用 ffmpeg 从视频中提取音频（PCM WAV）"""
        try:
            result = subprocess.run([
                "ffmpeg", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                audio_path, "-y",
            ], capture_output=True, timeout=60)

            if result.returncode != 0:
                logger.error(f"ffmpeg 提取音频失败: {result.stderr.decode(errors='replace')[:300]}")
                return False

            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                logger.info(f"  [OK] 音频提取: {size_mb:.1f}MB")
                return True
            return False

        except FileNotFoundError:
            logger.error("ffmpeg 未安装，请先安装 ffmpeg")
            return False
        except Exception as e:
            logger.error(f"音频提取异常: {e}")
            return False

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def is_douyin_url(url: str) -> bool:
        """判断是否为抖音链接"""
        return bool(re.search(r"(douyin\.com|iesdouyin\.com)", url))

    @staticmethod
    def extract_share_url(text: str) -> Optional[str]:
        """从文本中提取抖音分享链接"""
        m = re.search(r"https?://v\.douyin\.com/\S+", text)
        return m.group().rstrip(".,;!?，。；！？'\")]") if m else None


# ===========================================================================
# 便捷函数
# ===========================================================================

def download_douyin_video(video_url: str, output_dir: Optional[str] = None) -> Optional[str]:
    """便捷函数：下载抖音视频"""
    crawler = DouyinCrawler()
    return crawler.download(video_url, output_dir=output_dir)


def download_douyin_audio(video_url: str, output_dir: Optional[str] = None) -> Optional[str]:
    """便捷函数：下载抖音视频并提取音频"""
    crawler = DouyinCrawler()
    return crawler.download_audio(video_url, output_dir=output_dir)


def _setup_logging():
    """配置日志"""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )


# ===========================================================================
# 命令行入口
# ===========================================================================

if __name__ == "__main__":
    _setup_logging()
    if len(sys.argv) < 2:
        print("用法: python douyin_crawler.py <抖音分享链接>")
        sys.exit(1)

    crawler = DouyinCrawler()
    path = crawler.download(sys.argv[1])
    if path:
        print(f"\n下载成功: {path}")
    else:
        print("\n下载失败")
        sys.exit(1)
