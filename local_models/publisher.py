"""
抖音自动发布模块（Douyin Auto Publisher）
========================================
基于 Playwright + Chrome DevTools Protocol 实现抖音创作者平台自动发布。

依赖:
  - Playwright: pip install playwright && playwright install chromium
  - Chrome 需开启调试端口: --remote-debugging-port=9222

用法:
  from local_models.publisher import auto_publishing_videos_DY
  result = auto_publishing_videos_DY("/path/to/video.mp4", "视频标题", pulish_with_cover=False)
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DOUYIN_CREATOR_URL = "https://creator.douyin.com/creator-micro/content/upload"
DOUYIN_LOGIN_URL = "https://creator.douyin.com/"
CHROME_CDP_URL = "http://localhost:9222"
UPLOAD_TIMEOUT = 120_000       # 上传超时（ms）
PUBLISH_TIMEOUT = 60_000       # 发布超时（ms）

# ---------------------------------------------------------------------------
# Playwright + CDP 浏览器连接
# ---------------------------------------------------------------------------

def _get_browser():
    """连接到已开启调试端口的 Chrome"""
    from playwright.sync_api import sync_playwright

    playwright = sync_playwright().start()
    try:
        browser = playwright.chromium.connect_over_cdp(CHROME_CDP_URL)
        logger.info(f"已连接到 Chrome (CDP: {CHROME_CDP_URL})")
        return playwright, browser
    except Exception as e:
        logger.error(f"无法连接 Chrome CDP: {e}")
        logger.error("请先启动 Chrome: chrome --remote-debugging-port=9222")
        playwright.stop()
        raise


def _ensure_douyin_login(page) -> bool:
    """确保已登录抖音创作者平台"""
    page.goto(DOUYIN_CREATOR_URL, wait_until="domcontentloaded", timeout=30_000)
    time.sleep(3)

    # 检查是否已登录（页面无登录弹窗/跳转）
    if "login" not in page.url.lower() and "passport" not in page.url.lower():
        logger.info("抖音创作者平台 — 已登录")
        return True

    logger.warning("⚠️ 需要登录抖音创作者平台，请在浏览器中扫码登录...")
    page.goto(DOUYIN_LOGIN_URL, wait_until="domcontentloaded", timeout=30_000)

    # 等待手动登录（最多 120 秒）
    for i in range(120):
        time.sleep(1)
        if "login" not in page.url.lower() and "passport" not in page.url.lower():
            logger.info("抖音创作者平台 — 登录成功")
            return True
        if i % 10 == 0:
            logger.info(f"   等待登录中... ({i}s)")

    logger.error("登录超时（120s）")
    return False


# ---------------------------------------------------------------------------
# 抖音发布核心逻辑
# ---------------------------------------------------------------------------

def _upload_to_douyin(page, video_path: str, title: str,
                      pulish_with_cover: bool = False) -> Tuple[bool, str]:
    """上传视频到抖音创作者平台并发布"""
    abs_path = str(Path(video_path).resolve())
    if not os.path.exists(abs_path):
        return False, f"视频文件不存在: {abs_path}"

    logger.info(f"开始发布到抖音:")
    logger.info(f"  视频: {abs_path}")
    logger.info(f"  标题: {title[:50]}...")

    # 进入上传页面
    page.goto(DOUYIN_CREATOR_URL, wait_until="domcontentloaded", timeout=30_000)
    time.sleep(3)

    # ---- 步骤 1：上传视频文件 ----
    try:
        # 抖音创作者平台使用 <input type="file"> 上传
        file_input = page.locator('input[type="file"]').first
        file_input.wait_for(state="attached", timeout=10_000)
        file_input.set_input_files(abs_path)
        logger.info("✅ 视频文件已选择，等待上传...")
    except Exception as e:
        # 备选：尝试查找上传区域的 file input
        try:
            file_input = page.locator('input[accept*="video"]').first
            file_input.wait_for(state="attached", timeout=5_000)
            file_input.set_input_files(abs_path)
            logger.info("✅ 视频文件已选择（备选方式）")
        except Exception as e2:
            return False, f"未找到文件上传入口: {e2}"

    # 等待上传完成
    try:
        # 等待上传进度消失 / 发布按钮出现
        page.wait_for_selector(
            '.publish-btn, button:has-text("发布"), [class*="publish"]',
            timeout=UPLOAD_TIMEOUT,
        )
        time.sleep(2)
        logger.info("✅ 视频上传完成")
    except Exception:
        return False, "视频上传超时或失败"

    # ---- 步骤 2：填写标题 ----
    try:
        # 创作者平台标题输入框
        title_input = page.locator(
            '.editor-input, [contenteditable="true"], '
            '[placeholder*="标题"], [class*="title"] input, '
            'input[placeholder*="添加"]'
        ).first
        title_input.wait_for(state="visible", timeout=10_000)
        title_input.click()
        time.sleep(0.5)
        title_input.fill(title)
        logger.info(f"✅ 标题已填写")
    except Exception as e:
        logger.warning(f"⚠️ 标题填写失败（将跳过）: {e}")

    # ---- 步骤 3：封面处理 ----
    if pulish_with_cover:
        try:
            cover_input = page.locator(
                'input[type="file"][accept*="image"]'
            ).first
            if cover_input.count():
                logger.info("  封面设置留空（使用默认封面）")
        except Exception:
            pass

    # ---- 步骤 4：点击发布 ----
    try:
        publish_btn = page.locator(
            '.publish-btn, button:has-text("发布"), '
            '[class*="publish"] button, '
            'button:has-text("提交")'
        ).first
        publish_btn.wait_for(state="visible", timeout=10_000)
        publish_btn.click()
        logger.info("✅ 已点击发布按钮")
    except Exception as e:
        return False, f"未找到发布按钮: {e}"

    # 等待发布结果
    time.sleep(3)
    try:
        # 检查是否出现成功提示
        success = page.locator(
            ':has-text("发布成功"), :has-text("上传成功"), '
            ':has-text("作品已发布")'
        ).first
        success.wait_for(state="visible", timeout=PUBLISH_TIMEOUT)
        logger.info("🎉 抖音发布成功！")
        return True, "✅ 抖音发布成功"
    except Exception:
        # 可能有弹窗确认
        try:
            confirm_btn = page.locator(
                'button:has-text("确定"), button:has-text("确认"), '
                '[class*="confirm"] button'
            ).first
            if confirm_btn.is_visible(timeout=3_000):
                confirm_btn.click()
                time.sleep(2)
                logger.info("🎉 抖音发布成功（二次确认）")
                return True, "✅ 抖音发布成功"
        except Exception:
            pass

        # 不确定是否成功，假定已发布
        logger.warning("⚠️ 无法确认发布结果，请手动检查")
        return True, "⚠️ 发布状态未确认，请检查抖音"


# ---------------------------------------------------------------------------
# 对外接口（与 app.py 签名兼容）
# ---------------------------------------------------------------------------

def auto_publishing_videos_DY(
    video_path: str,
    title: str = "",
    pulish_with_cover: bool = False,
) -> str:
    """
    将视频发布到抖音

    Args:
        video_path: 视频文件路径
        title: 视频标题/文案
        pulish_with_cover: 是否附带自定义封面

    Returns:
        状态文本
    """
    if not video_path or not os.path.exists(video_path):
        return "❌ 视频文件不存在"

    title = title or "精彩视频"

    try:
        playwright, browser = _get_browser()
    except Exception as e:
        return f"❌ 浏览器连接失败: {e}"

    try:
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        pages = context.pages
        page = pages[0] if pages else context.new_page()

        if not _ensure_douyin_login(page):
            return "❌ 抖音登录失败，请在浏览器中手动登录后重试"

        ok, msg = _upload_to_douyin(page, video_path, title, pulish_with_cover)
        return msg

    except Exception as e:
        logger.error(f"抖音发布异常: {e}")
        return f"❌ 发布失败: {e}"
    finally:
        # 不关闭浏览器，保持 Chrome 连接
        pass


def auto_publishing_videos_XHS(
    video_path: str,
    title: str = "",
    pulish_with_cover: bool = False,
) -> str:
    """
    将视频发布到小红书（暂未实现，预留接口）

    小红书发布需要在手机端或使用官方 API，
    Web 端创作者中心功能有限。
    """
    return "⚠️ 小红书自动发布暂未实现（需手机端操作或官方 API）"


def auto_publishing_videos_SPH(
    video_path: str,
    title: str = "",
    pulish_with_cover: bool = False,
) -> str:
    """
    将视频发布到视频号/蝴蝶号（暂未实现，预留接口）

    视频号发布需要通过微信公众平台或手机端操作。
    """
    return "⚠️ 视频号自动发布暂未实现（需手机端操作或微信 API）"


def auto_publishing_videos_ALL(
    video_path: str,
    title: str = "",
    pulish_with_cover: bool = False,
) -> str:
    """
    一键发布到所有平台（抖音 + 小红书 + 视频号）
    """
    results = []
    results.append(f"【抖音】{auto_publishing_videos_DY(video_path, title, pulish_with_cover)}")
    results.append(f"【小红书】{auto_publishing_videos_XHS(video_path, title, pulish_with_cover)}")
    results.append(f"【视频号】{auto_publishing_videos_SPH(video_path, title, pulish_with_cover)}")
    return "\n".join(results)


def auto_publishing_videos_DY_ALL(
    link_input: str = "",
    two_line_input: str = "",
    pt_file_dropdown: Optional[str] = None,
    video_model_dropdown: Optional[str] = None,
    api_key: str = "",
    speed: float = 1.0,
    pt_files_info: str = "",
    background_image=None,
    background_image_list=None,
    check_box: bool = False,
    skip_bgm_add_box: bool = False,
    bgm_list: Optional[str] = None,
    user_upload_bgm=None,
    bgm_volume_control: float = 1.0,
    when_auto_use_cover_checkbox: bool = False,
    use_ai_checkbox: bool = False,
    cover_text: str = "",
    highlight_words_text: str = "",
    font_family_dropdown: Optional[str] = None,
    font_size_number: int = 48,
    font_color_picker: str = "#FFFFFF",
    highlight_color_picker: str = "#FFFF00",
    position_dropdown: str = "bottom",
    frame_time_number: Optional[float] = None,
    pulish_with_cover: bool = False,
    silence_check_box: bool = False,
    digital_human_version_dropdown: Optional[str] = None,
    subtitle_generation_type_dropdown: Optional[str] = None,
    template_id: Optional[str] = None,
) -> str:
    """
    一键追爆款并发布到抖音（完整流程的快捷入口）

    此函数由 app.py 调用，用于一键追爆款 + 发布的完整流程。
    当前实现简化为直接发布已有视频，全流程编排在 app.py 中处理。

    Args:
        link_input: 视频链接
        two_line_input: 自定义文案
        ... (其他参数与 app.py UI 一致)

    Returns:
        状态文本
    """
    # 该函数的完整流程由 app.py 的 Post_on_DY_ALL 事件处理
    # 此处提供占位实现，实际由上层调用者负责编排
    return "⚠️ 一键全流程发布请使用 app.py 主界面操作（完整管线已在 app.py 中编排）"


# ---------------------------------------------------------------------------
# 便捷测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if len(sys.argv) < 2:
        print("用法: python publisher.py <视频路径> [标题]")
        sys.exit(1)

    video = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else "精彩视频"
    print(auto_publishing_videos_DY(video, title))
