import os
import gradio as gr
import configparser
from fastapi import FastAPI, Request, Response
import logging
import sys
import time
import shutil
import uuid
import gc
import torch
import cv2

# 移除直接导入human_base核心模块，改为使用generate_video.py中的domake方法
from utils.video_processor import (
    download_and_extract_text,
)
from utils.key_manager import (
    save_api_key,
    delete_api_key,
    refresh_api_key,
)
from utils.voice_processor import (
    run_GPTvoice_command,
    handle_audio_creation,
    get_pt_files,
    download_audio,
    get_bgm_list,
    get_background_images,
    add_bgm_to_video_function,
    add_bgm_to_video_function_with_random_choice,
    save_subtitle_text,
    generate_subtitle_only,
    generate_audio_only
)
from utils.update_handler import (
    update_platform_elements,
    do_update,
)
from utils.service_launcher import (
    start_digit_human,
    start_cosyvoice,
)

from utils.video_cover_image import (
    generate_cover_image_gui,
)
from ai_processing.text_rewriter import (
    AI_write_descriptions,
    execute_rewrite,
)
from video_tools.generate_video import get_trained_models, generate_tuilionnx_video, get_face_list, refresh_face_list

from video_tools.subtitle_utils import (
    add_subtitles_to_video_with_style,
    FONT_FAMILIES,
)
from local_models.publisher import (
    auto_publishing_videos_DY,
    auto_publishing_videos_XHS,
    auto_publishing_videos_SPH,
)
from local_models.adapter import AI_generate_publish_content

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

#设置human_base的初始变量
if_gfpgan_default = True
if_res_default = False
if_ifface_default = True


# 设置标准输出无缓冲
sys.stdout.reconfigure(line_buffering=True)


def refresh_voice_list():
    """获取并返回最新的音色列表

    Returns:
        gr.update: 更新下拉列表的选项
    """
    pt_files = get_pt_files()  # 获取最新的音色文件列表
    choices = [name for name, _ in pt_files]  # 返回音色名称列表
    return gr.update(choices=choices)


def refresh_bgm_list():
    bgm_list = get_bgm_list()
    choices = [name for name, _ in bgm_list]
    return gr.update(choices=choices)


def refresh_background_images():
    images = get_background_images()
    choices = [name for name, _ in images]
    return gr.update(choices=choices)


def cancel_update():
    """取消更新操作

    Returns:
        tuple: 状态信息和对话框可见性更新
    """
    return ("用户取消更新", gr.update(visible=False))


def _resolve_bgm_path(bgm_dropdown_name: str, user_upload_bgm_path: str) -> str:
    """解析 BGM 文件路径（优先用户上传，其次下拉选择）"""
    # 优先用户上传的文件
    if user_upload_bgm_path and os.path.exists(user_upload_bgm_path):
        return user_upload_bgm_path
    # 从下拉列表中查找对应的路径
    if bgm_dropdown_name:
        for name, path in get_bgm_list():
            if name == bgm_dropdown_name and os.path.exists(path):
                return path
    return ""


def generate_video_preview(
    video_path: str,
    subtitle_text: str,
    font_family: str,
    font_size: int,
    font_color: str,
    outline_color: str,
    bottom_margin: int,
) -> str:
    """从视频中提取一帧并叠加字幕样式，生成预览图

    流程:
      ① OpenCV 读取视频 → 定位到中间帧（约第 2 秒处）
      ② PIL 在画面上根据字幕样式渲染字幕文本
      ③ 保存为临时 PNG 预览图 → 返回路径

    Args:
        video_path: 视频文件路径
        subtitle_text: 字幕文本内容（取前 30 字展示）
        font_family: 字幕字体名称
        font_size: 字号
        font_color: 字体颜色（#RRGGBB）
        outline_color: 描边颜色（#RRGGBB）
        bottom_margin: 底部边距（px）

    Returns:
        预览图文件路径
    """
    import tempfile
    from PIL import Image, ImageDraw, ImageFont

    if not video_path or not os.path.exists(video_path):
        return None

    # ── ① 提取视频帧 ──
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 定位到第 2 秒左右的帧（有画面的概率高，避免全黑开场帧）
    target_frame = min(int(2.0 * fps) if fps > 0 else total_frames // 3, total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame_bgr = cap.read()
    if not ret:
        # 回退：读第一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        return None

    # BGR → RGB → PIL
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    img_w, img_h = pil_img.size

    # ── ② 加载字体 ──
    try:
        font = ImageFont.truetype(font_family, font_size)
    except Exception:
        # 回退：尝试 PIL 默认字体路径，或使用默认字体
        try:
            # macOS/Linux 常见中文字体路径
            fallback_fonts = [
                "/System/Library/Fonts/PingFang.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "C:\\Windows\\Fonts\\msyh.ttc",
                "C:\\Windows\\Fonts\\simhei.ttf",
            ]
            font = None
            for fp in fallback_fonts:
                if os.path.exists(fp):
                    font = ImageFont.truetype(fp, font_size)
                    break
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

    # ── ③ 处理字幕文本（去换行，限制长度） ──
    display_text = subtitle_text.replace("\n", " ").strip()
    if len(display_text) > 40:
        display_text = display_text[:40] + "..."

    if not display_text:
        display_text = "（字幕预览：暂无文本）"

    # ── ④ 文字测量与居中 ──
    bbox = draw.textbbox((0, 0), display_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (img_w - text_w) // 2
    y = img_h - bottom_margin - text_h

    # ── ⑤ 描边渲染（模拟 outline 效果：8 个方向偏移绘制） ──
    outline_offsets = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ]
    for dx, dy in outline_offsets:
        draw.text((x + dx, y + dy), display_text, font=font, fill=outline_color)

    # ── ⑥ 主文字渲染 ──
    draw.text((x, y), display_text, font=font, fill=font_color)

    # ── ⑦ 保存为临时文件 ──
    fd, preview_path = tempfile.mkstemp(suffix=".png", prefix="video_preview_")
    os.close(fd)
    pil_img.save(preview_path, "PNG")
    logging.info(f"预览图已生成: {preview_path}")
    return preview_path


def auto_publish_dy_with_llm(
    video_path: str,
    script_text: str,
    bgm_dropdown: str,
    user_upload_bgm: str,
    pulish_with_cover: bool,
) -> str:
    """一键发布到抖音（自动调用 LLM 生成标题/文案/标签）

    流程：
    1. LLM 根据脚本生成 { title, description, tags }
    2. 解析用户选择的 BGM 路径
    3. 自动填入抖音发布页并发布
    """
    if not video_path:
        return "❌ 请先生成视频"

    # Step 1: LLM 生成标题、描述、标签
    content = AI_generate_publish_content(script_text)
    title = content.get("title", "") or script_text[:20]
    description = content.get("description", "")
    tags = content.get("tags", [])

    logging.info(f"LLM 生成 — 标题: {title}")
    logging.info(f"LLM 生成 — 描述: {description}")
    logging.info(f"LLM 生成 — 标签: {tags}")

    # Step 2: 解析 BGM
    bgm_path = _resolve_bgm_path(bgm_dropdown, user_upload_bgm)

    # Step 3: 发布
    return auto_publishing_videos_DY(
        video_path=video_path,
        title=title,
        description=description,
        tags=tags,
        bgm_path=bgm_path,
        pulish_with_cover=pulish_with_cover,
    )


def auto_full_pipeline(
    link_input: str,
    pt_file_dropdown: str,
    face_model: str,
    speed: float,
    bgm_list: str,
    user_upload_bgm: str,
    bgm_volume_control: float,
    pulish_with_cover: bool,
    skip_bgm_add_box: bool,
    # ── 字幕参数（默认值适配抖音/短视频场景）────
    subtitle_font: str = "Microsoft YaHei",
    subtitle_size: int = 11,
    subtitle_color: str = "#FFFFFF",
    subtitle_outline: str = "#000000",
    subtitle_margin: int = 60,
) -> str:
    """一键追爆款并发布到抖音（完整自动化流程）

    流程:
      ① 下载抖音视频 → 提取口播文案（yt-dlp + ASR）
      ② TTS 语音合成（CosyVoice）
      ③ 数字人口型合成（MuseTalk）
      ④ 嵌入字幕（自动语音识别 + 样式渲染）
      ⑤ 添加背景音乐（可选）
      ⑥ LLM 生成标题/描述/标签 + Playwright 自动发布

    Returns:
        每步状态汇总文本
    """
    results = []

    # ================================================================
    # Step 1: 下载视频 + 提取文案
    # ================================================================
    if not link_input:
        return "❌ 请先输入抖音视频分享链接"

    logging.info(f"📥 开始下载并提取文案: {link_input[:80]}...")
    text = download_and_extract_text(link_input)
    if not text:
        return "❌ 视频下载或文案提取失败，请检查链接是否正确（需支持 yt-dlp）"
    results.append(f"✅ [1/6] 文案提取成功 ({len(text)} 字)")
    logging.info(f"📝 提取文案: {text[:50]}...")

    # ================================================================
    # Step 2: TTS 语音合成
    # ================================================================
    logging.info("🔊 开始语音合成...")
    audio_path, audio_status = handle_audio_creation(text, pt_file_dropdown, speed)
    if not audio_path:
        results.append("❌ [2/6] TTS 语音合成失败")
        return "\n".join(results)
    results.append(f"✅ [2/6] 语音合成完成")

    # ================================================================
    # Step 3: 数字人口型合成
    # ================================================================
    logging.info("🎬 开始生成数字人视频...")
    try:
        video_result = generate_tuilionnx_video(
            face_model, None, audio_path,
            batch_size=4, sync_offset=0,
            scale_h=1.6, scale_w=3.6,
            compress=False, beautify_teeth=False,
            silence_check=False, add_watermark=True,
            bg_image=None, bg_image_list=None, check_box=False,
        )
        # generate_tuilionnx_video returns (video_path, time_str, file_data, url)
        video_path = video_result[0] if video_result else None
        if not video_path or not os.path.exists(video_path):
            results.append("❌ [3/6] 数字人视频生成失败")
            return "\n".join(results)
        results.append(f"✅ [3/6] 数字人视频生成完成")
    except Exception as e:
        results.append(f"❌ [3/6] 数字人视频生成异常: {e}")
        return "\n".join(results)

    # ================================================================
    # Step 4: 嵌入字幕
    # ================================================================
    logging.info("📝 为视频嵌入字幕...")
    try:
        subtitle_status, video_path = add_subtitles_to_video_with_style(
            video_path,
            subtitle_font,
            subtitle_size,
            subtitle_color,
            subtitle_outline,
            subtitle_margin,
        )
        if video_path and os.path.exists(video_path):
            results.append(f"✅ [4/6] 字幕已嵌入 ({subtitle_font}, {subtitle_size}px)")
        else:
            results.append(f"⚠️ [4/6] 字幕嵌入返回状态异常: {subtitle_status}")
    except Exception as e:
        logging.warning(f"字幕嵌入失败: {e}")
        results.append(f"⚠️ [4/6] 字幕嵌入失败（已跳过）: {e}")

    # ================================================================
    # Step 5: 添加 BGM（可选）
    # ================================================================
    if not skip_bgm_add_box:
        bgm_path = _resolve_bgm_path(bgm_list, user_upload_bgm)
        if bgm_path:
            logging.info("🎵 添加背景音乐...")
            try:
                _, video_path = add_bgm_to_video_function(
                    video_path, bgm_list, user_upload_bgm, bgm_volume_control
                )
                results.append(f"✅ [5/6] 背景音乐已添加")
            except Exception as e:
                logging.warning(f"BGM 添加失败: {e}")
                results.append(f"⚠️ [5/6] BGM 添加失败（已跳过）")
        else:
            results.append(f"⏭️ [5/6] 未选择背景音乐（已跳过）")
    else:
        results.append(f"⏭️ [5/6] 已勾选跳过 BGM")

    # ================================================================
    # Step 6: LLM 生成标题/描述/标签 + 发布
    # ================================================================
    logging.info("🤖 LLM 生成发布内容...")
    content = AI_generate_publish_content(text)
    title = content.get("title", "") or text[:20]
    description = content.get("description", "")
    tags = content.get("tags", [])

    logging.info("🚀 开始发布到抖音...")
    publish_result = auto_publishing_videos_DY(
        video_path=video_path,
        title=title,
        description=description,
        tags=tags,
        bgm_path="",
        pulish_with_cover=pulish_with_cover,
    )
    results.append(f"✅ [6/6] LLM 标题: {title}")
    results.append(f"   {publish_result}")

    return "\n".join(results)


def create_ui():
    """创建UI界面并绑定各种事件处理函数

    Returns:
        gr.Blocks: 创建好的Gradio Blocks界面
    """
    # 禁用Gradio分析功能
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

    with gr.Blocks(title="罗根 一键追爆智能体", analytics_enabled=False) as demo:
        app = demo.app

        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 允许所有源
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 单独处理 WebSocket 的 OPTIONS 请求
        @app.middleware("http")
        async def websocket_cors_middleware(request, call_next):
            if request.url.path == "/queue/join":
                if request.method == "OPTIONS":
                    response = Response()
                    response.headers["Access-Control-Allow-Origin"] = "*"
                    response.headers["Access-Control-Allow-Methods"] = (
                        "GET, POST, OPTIONS"
                    )
                    response.headers["Access-Control-Allow-Headers"] = "*"
                    return response
            return await call_next(request)

        with gr.Group(visible=False) as main_interface:  # 将整个界面包装在不可见组中
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        gr.Markdown(
                            """
                            AI
                            """
                        )

            with gr.Row():
                with gr.Row():
                    with gr.Column():
                        link_input = gr.Textbox(
                            label="请输入视频链接地址",
                            elem_classes="custom-textbox1",
                            lines=4,
                        )
                        with gr.Row():
                            Post_on_DY_ALL = gr.Button(
                                "一键追爆款并发布到各平台", elem_classes="blue-button"
                            )
                        # # 添加图像显示组件
                        # )
                    with gr.Column():
                        extract_text_button = gr.Button("提取视频文案")
                        text_input = gr.Textbox(
                            label="可手动修改文案",
                            elem_classes="custom-textbox",
                            value="",
                        )
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    api_key_input = gr.Textbox(
                                        label="输入Deepseek API Key",
                                        elem_classes=["compact-textbox"],
                                        value="",
                                    )
                                    save_api_key_button = gr.Button("保存API Key")
                                    delete_api_key_button = gr.Button("删除API Key")

                                    # 读取配置,选择API Key
                                    config = configparser.ConfigParser()
                                    config.read("config.ini", encoding="utf-8")
                                    keys = config.get("deepseek_apikey", "key").split(
                                        ","
                                    )
                                    default_key = keys[0] if keys else None
                                    # 添加key选择下拉框
                                    api_key = gr.Dropdown(
                                        choices=keys,
                                        value=None,
                                        label="必须先选择一个key",
                                    )
                                    refresh_api_key_button = gr.Button(
                                        "刷新keys",
                                        elem_classes=["custom-btn"],
                                        variant="primary",
                                        visible=False,
                                    )
                                with gr.Row():
                                    ai_mode = gr.Radio(
                                        elem_classes=["compact-radio"],
                                        choices=["AI自动仿写", "根据指令仿写"],
                                        label="仿写模式选择（以下两种方式均调用deepseek671b满血版）",
                                        value="AI自动仿写",
                                    )
                                    AI_prompt = gr.Textbox(
                                        label="prompt，即改写文案的规则和要求，例：请用幽默的口吻改写这段文案",
                                        elem_classes=[
                                            "custom-textbox",
                                            "compact-textbox",
                                        ],
                                        lines=1,
                                        visible=False,
                                    )
                                    # 修改为单个按钮
                                    AI_execute_button = gr.Button(
                                        "执行仿写",
                                        elem_classes=["custom-btn"],
                                        variant="primary",
                                    )
            with gr.Row():
                # 状态显示区域（统一的状态显示）
                status_output = gr.Textbox(label="状态信息")

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        video_model_dropdown = gr.Dropdown(
                            choices=get_trained_models(),
                            label="选择人物形象",
                            value=(
                                get_trained_models()[0]
                                if get_trained_models()
                                else None
                            ),
                        )
                        # 移除背景图片相关组件
                        # )
                        #     ),
                        # )
                        # )
                        # )
                        
                        # 创建隐藏的背景相关组件以保持兼容性
                        background_image = gr.Image(
                            label="背景图片功能已移除", type="filepath", visible=False
                        )
                        background_image_list = gr.Dropdown(
                            choices=[], label="背景图片功能已移除", visible=False
                        )
                        check_box = gr.Checkbox(
                            label="背景替换功能已移除", value=False, interactive=False, visible=False
                        )
                    with gr.Row():
                        video_output = gr.Video(label=" 视频预览 ", interactive=False)
                        Create_Digital_Human = gr.Button("生成视频")
                    with gr.Row():
                        srt_text_output = gr.Textbox(
                            lines=10,
                            label="字幕文本内容",
                            elem_classes=["custom-textbox"],
                        )
                        save_subtitle_button = gr.Button("保存字幕文本")

                with gr.Column():
                    with gr.Row():
                        pt_files = get_pt_files()
                        pt_file_dropdown = gr.Dropdown(
                            label="选择音色",
                            choices=[name for name, _ in pt_files],  # 显示名称列表
                            value=(
                                pt_files[0][0] if pt_files else None
                            ),  # 默认选择第一个音色的显示名称
                            type="index",  # 使用索引来选择
                        )
                        # 在适当的位置添加刷新按钮
                        refresh_button = gr.Button("刷新音色")

                        # 绑定按钮点击事件
                        refresh_button.click(
                            fn=refresh_voice_list,  # 使用具名函数替代lambda
                            outputs=[
                                pt_file_dropdown
                            ],  # 假设 pt_file_dropdown 是音色选择的下拉框
                        )
                    with gr.Row():
                        start_GPTvoice_button = gr.Button("启动语音接口")
                        voice_status = gr.Markdown(
                            "🔴 未启动", elem_classes="status-label"
                        )
                    speed = gr.Number(
                        value=1,
                        label="语速调节",
                        minimum=0.5,
                        maximum=2.0,
                        step=0.1,
                    )
                    Create_audio = gr.Button(" 生成音频")
                    audio_output = gr.Audio(
                        label="音频预览",
                        type="filepath",
                        interactive=True,
                        elem_id="audio_output",
                    )
                    download_button = gr.DownloadButton("下载音频")
                    Create_subtitle = gr.Button("单独生成字幕")


            # 字幕设置和视频描述区域
            with gr.Row():
                # 字幕设置
                with gr.Column(scale=2):
                    with gr.Row():
                        font_family = gr.Dropdown(
                            choices=FONT_FAMILIES,  # 使用字体真实名称列表
                            value=(
                                FONT_FAMILIES[0] if FONT_FAMILIES else "Microsoft YaHei"
                            ),
                            label="字体",
                        )
                        font_size = gr.Number(value=11, label="字体大小")
                    with gr.Row():
                        font_color = gr.ColorPicker(value="#FFFFFF", label="字体颜色")
                        outline_color = gr.ColorPicker(
                            value="#000000", label="描边颜色"
                        )
                    bottom_margin = gr.Number(value=60, label="底部边距")

                # 操作按钮和描述输入
                with gr.Column(scale=1):
                    add_subtitle_btn = gr.Button("添加字幕到视频", variant="primary")
                    AI_miaoshu = gr.Button("deepseek撰写视频描述与话题标签")
                    # 视频描述和话题（移到AI撰写按钮下方）
                    two_line_input = gr.Textbox(
                        label="视频描述和话题标签",
                        placeholder="视频描述（换行）#话题#话题",
                        lines=2,
                        interactive=True,
                    )
                with gr.Column(scale=1):
                    use_random_choice = gr.Button("随机选择背景音乐")
                    skip_bgm_add_box = gr.Checkbox(
                        label="全自动时是否跳过添加随机BGM",
                        value=False,
                        interactive=True,
                    )
                    bgm_list = gr.Dropdown(
                        choices=[name for name, _ in get_bgm_list()],
                        value=None,  # 默认选择第一个音色的显示名称
                        label="背景音乐",
                        interactive=True,
                    )
                    user_upload_bgm = gr.File(type="filepath")
                    refresh_bgm_button = gr.Button("刷新背景音乐")
                    bgm_volume_control = gr.Slider(
                        value=0.5,
                        label="背景音乐音量",
                        minimum=0,
                        maximum=1,
                        step=0.1,
                    )
                    refresh_bgm_button.click(
                        fn=refresh_bgm_list,
                        outputs=[bgm_list],
                    )
                    use_random_choice.click(
                        fn=add_bgm_to_video_function_with_random_choice,
                        inputs=[
                            video_output,
                            bgm_volume_control,
                        ],
                        outputs=[status_output, video_output],
                    )
                    add_bgm_to_video = gr.Button("添加背景音乐到视频")
                    add_bgm_to_video.click(
                        fn=add_bgm_to_video_function,
                        inputs=[
                            video_output,
                            bgm_list,
                            user_upload_bgm,
                            bgm_volume_control,
                        ],
                        outputs=[status_output, video_output],
                    )


            # 添加封面图
            with gr.Column(scale=2):
                with gr.Row():
                    when_auto_use_cover_checkbox = gr.Checkbox(
                        label="一键全流程时是否创建封面图", value=False
                    )
                    use_ai_checkbox = gr.Checkbox(
                        label="使用AI生成封面文案", value=False
                    )
                with gr.Row():
                    cover_text = gr.Textbox(
                        label="封面文案（非AI模式下必填）",
                        lines=2,
                        interactive=True,
                        placeholder="封面文案（非AI模式下必填）",
                    )
                    highlight_words_text = gr.Textbox(
                        label="高亮词（逗号分隔，非AI模式下必填）",
                        lines=2,
                        interactive=True,
                        placeholder="高亮词（逗号分隔，非AI模式下必填）",
                    )
                with gr.Row():
                    font_family_dropdown = gr.Dropdown(
                        choices=FONT_FAMILIES,
                        value=FONT_FAMILIES[0] if FONT_FAMILIES else "SimHei",
                        label="字体",
                        interactive=True,
                    )
                    font_size_number = gr.Number(
                        value=60, label="字体大小", interactive=True
                    )
                with gr.Row():
                    font_color_picker = gr.ColorPicker(
                        value="#FFFFFF", label="字体颜色", interactive=True
                    )
                    highlight_color_picker = gr.ColorPicker(
                        value="#FFD600", label="高亮颜色", interactive=True
                    )
                with gr.Row():
                    position_dropdown = gr.Dropdown(
                        choices=["top", "center", "bottom"],
                        value="bottom",
                        label="文字位置",
                        interactive=True,
                    )
                    frame_time_number = gr.Number(
                        value=None, label="抽帧时间点（秒，可选）", interactive=True
                    )
                with gr.Row():
                    generate_cover_btn = gr.Button(
                        "生成封面图", variant="primary", interactive=True
                    )
                with gr.Row():
                    cover_preview = gr.Image(label="封面预览", interactive=False)
                with gr.Row():
                    pulish_with_cover = gr.Checkbox(
                        label="发布时附带封面？", value=False
                    )
            # ── 效果预览区域 ──
            with gr.Row():
                preview_btn = gr.Button("效果预览", variant="primary")
            with gr.Row():
                effect_preview = gr.Image(label="字幕效果预览", interactive=False)

            # 发布按钮（页面最下方）
            with gr.Row():
                Post_on_DY = gr.Button("发布到dou音", size="large", variant="primary")
                Post_on_XHS = gr.Button("发布到小红薯", size="large", variant="primary")
                Post_on_SPH = gr.Button("发布到蝴蝶号", size="large", variant="primary")
            with gr.Row():
                Post_on_ALL = gr.Button(
                    "一键发布到各平台", size="large", variant="primary"
                )
                # 移除账号输入框（已移除登录系统）
                account = gr.Textbox(label="默认账号", value="", interactive=False, visible=False)
                pt_files_info = gr.Textbox(
                    label="音色信息，传递一个空值",
                    value="",
                    elem_classes="custom-textbox",
                )

            
            # 注释掉原有的human_base组件，保留以备后用
            # #迁移human_base中的组件到追爆后端主系统中
            
            # 新增tuilionnx数字人组件
            with gr.Column(scale=2):
                with gr.Column():
                    gr.Markdown("### TuiliONNX 数字人生成")
                    video = gr.Video(label="上传视频", interactive=False)
                    
                    with gr.Row():
                        batch_size = gr.Number(label="批次大小", value=4, minimum=1, maximum=16, interactive=True)
                        sync_offset = gr.Number(label="音画同步偏移", value=0, minimum=-10, maximum=10, interactive=True)
                    
                    with gr.Row():
                        scale_h = gr.Number(label="遮罩高度比例", value=1.6, minimum=0.5, maximum=3.0, step=0.1, interactive=True)
                        scale_w = gr.Number(label="遮罩宽度比例", value=3.6, minimum=0.5, maximum=5.0, step=0.1, interactive=True)
                        #增加是否进行压缩推理
                        compress_inference_check_box = gr.Checkbox(
                            label="是否进行压缩推理", value=False, interactive=True
                        )
                        # 增加是否美化牙齿
                        beautify_teeth_check_box = gr.Checkbox(
                            label="是否美化牙齿", value=False, interactive=True
                        )
                    tuilionnx_make_button = gr.Button("生成TuiliONNX数字人", variant="primary")

                with gr.Column():
                        face = gr.Dropdown(label="人物模型",choices=get_face_list(),interactive=True,value=None)
                        refresh_button = gr.Button("刷新视频模型列表")
                        refresh_button.click(fn=refresh_face_list, inputs=[face], outputs=[face])
                        output_time = gr.Textbox(label="生成时间",interactive=True)
                        output_url = gr.Textbox(label="分享视频下载URL",interactive=True)
                        one_list = gr.File(interactive=False,label="生成结果下载")
                        # 移除剪辑气口功能
                        # )
                        silence_check_box = gr.Checkbox(
                            label="剪辑气口功能已移除", value=False, interactive=False, visible=False
                        )
                        addAIWatermark_check_box = gr.Checkbox(
                            label="是否添加AI水印", value=True, interactive=True
                        )
                        digital_human_version_dropdown = gr.Dropdown(
                            choices=["旧版数字人", "新版数字人"],
                            value="新版数字人",
                            label="数字人版本选择",
                            interactive=True
                        )
                        subtitle_generation_type_dropdown = gr.Dropdown(
                            choices=["普通字幕生成", "高级字幕生成"],
                            value="高级字幕生成",
                            label="字幕生成类型",
                            interactive=True
                        )
            # 注释掉原有的make_button绑定，保留以备后用
            
            # 绑定新的TuiliONNX数字人生成按钮
            tuilionnx_make_button.click(
                generate_tuilionnx_video,
                inputs=[
                    face,
                    video,
                    audio_output, 
                    batch_size,
                    sync_offset,
                    scale_h,
                    scale_w,
                    compress_inference_check_box,
                    beautify_teeth_check_box,
                    silence_check_box,
                    addAIWatermark_check_box,
                    background_image,
                    background_image_list,
                    check_box
                ],
                outputs=[video_output, output_time, one_list, output_url]
            )

            # 调用API 为视频添加具有样式和特效的字幕
            with gr.Row():
                with gr.Column(scale=1):
                    upload_video_btn = gr.Button("利用返回的oss分享链接调用接口为视频添加字幕")
                    template_id = gr.Dropdown(
                            choices=[],  # 特效字幕功能已移除
                            value=None,
                            label="模板ID",
                        )
                    refresh_template_id_button = gr.Button("刷新视频模板列表")
                # 移除特效字幕相关UI组件
                #
                #
                #
                #     # 绑定按钮点击事件
                #                 template_id
                #             ],
                #         )
                #     )
                #     )
                #     )



            # 生成封面图按钮绑定的回调函数
            def handle_generate_cover(
                use_ai,
                api_key_value,
                cover_text_value,
                highlight_words_value,
                font_family_value,
                font_size_value,
                font_color_value,
                highlight_color_value,
                position_value,
                frame_time_value,
            ):
                # max_width, outline_size, outline_color 使用默认值
                image_path = generate_cover_image_gui(
                    use_ai=use_ai,
                    api_key=api_key_value,
                    text=cover_text_value,
                    highlight_words=highlight_words_value,
                    font_family=font_family_value,
                    font_size=int(font_size_value) if font_size_value else 60,
                    font_color=font_color_value,
                    highlight_color=highlight_color_value,
                    position=position_value,
                    frame_time=frame_time_value if frame_time_value else None,
                    # 默认参数
                    max_width=0.8,
                    outline_size=4,
                    outline_color="#000000",
                )
                return image_path

            generate_cover_btn.click(
                handle_generate_cover,
                inputs=[
                    use_ai_checkbox,
                    api_key,
                    cover_text,
                    highlight_words_text,
                    font_family_dropdown,
                    font_size_number,
                    font_color_picker,
                    highlight_color_picker,
                    position_dropdown,
                    frame_time_number,
                ],
                outputs=[cover_preview],
            )

            # 将提取文案的函数绑定到按钮点击事件，并指定输入输出
            extract_text_button.click(
                download_and_extract_text,
                inputs=[link_input],
                outputs=[text_input],
            )

            def toggle_controls(mode):
                return {
                    AI_prompt: gr.update(visible=(mode == "根据指令仿写")),
                    AI_execute_button: gr.update(visible=True),
                }

            ai_mode.change(
                fn=toggle_controls,
                inputs=[ai_mode],
                outputs=[AI_prompt, AI_execute_button],
            )

            AI_execute_button.click(
                execute_rewrite,
                inputs=[text_input, ai_mode, AI_prompt, api_key],
                outputs=[text_input],
            )

            # 启动语音接口服务
            start_GPTvoice_button.click(
                run_GPTvoice_command, inputs=[account], outputs=[voice_status]
            )

            # AI撰写描述
            AI_miaoshu.click(
                AI_write_descriptions,
                inputs=[text_input, api_key],  # 添加 model_dropdown 作为输入
                outputs=[two_line_input],
            )

            Create_audio.click(
                handle_audio_creation,
                inputs=[text_input, pt_file_dropdown, speed],
                outputs=[audio_output, status_output],
            )
            Create_subtitle.click(
                generate_subtitle_only,
                inputs=[audio_output, text_input, api_key],
                outputs=[srt_text_output, status_output],
            )
            save_subtitle_button.click(
                save_subtitle_text,
                inputs=[srt_text_output],
                outputs=[status_output],
            )
            download_button.click(
                download_audio,
                inputs=[audio_output],
                outputs=[download_button],
            )

            # 生成视频
            #     generate_digit_human,
            #         audio_output,
            #         video_model_dropdown,
            #         background_image,
            #         background_image_list,
            #         check_box,
            #         silence_check_box,
            #         addAIWatermark_check_box
            #     ],  # 添加模型选择下拉框
            # )


            # 一键追爆款并发布到抖音（全自动：下载→生成→字幕→BGM→发布）
            Post_on_DY_ALL.click(
                auto_full_pipeline,
                inputs=[
                    link_input,           # ① 抖音视频分享链接
                    pt_file_dropdown,     # ② TTS 音色
                    face,                 # ③ 数字人形象
                    speed,                # ④ 语速
                    bgm_list,             # ⑤ 背景音乐（下拉）
                    user_upload_bgm,      # ⑥ 背景音乐（上传）
                    bgm_volume_control,   # ⑦ BGM 音量
                    pulish_with_cover,    # ⑧ 附带封面
                    skip_bgm_add_box,     # ⑨ 跳过 BGM
                    font_family,          # ⑩ 字幕字体
                    font_size,            # ⑪ 字幕大小
                    font_color,           # ⑫ 字幕颜色
                    outline_color,        # ⑬ 描边颜色
                    bottom_margin,        # ⑭ 底部边距
                ],
                outputs=[status_output],
            )

            # 绑定字幕添加按钮事件
            add_subtitle_btn.click(
                fn=add_subtitles_to_video_with_style,
                inputs=[
                    video_output,  # 视频路径
                    font_family,  # 字体
                    font_size,  # 字体大小
                    font_color,  # 字体颜色
                    outline_color,  # 描边颜色
                    bottom_margin,  # 底部边距
                ],
                outputs=[status_output, video_output],
                show_progress=True,  # 显示进度
            )

            # ── 效果预览按钮：从视频抽帧 + 叠加字幕样式 → 预览图 ──
            preview_btn.click(
                fn=generate_video_preview,
                inputs=[
                    video_output,     # 视频路径
                    text_input,       # 字幕文案（来自提取的文案）
                    font_family,      # 字体
                    font_size,        # 字号
                    font_color,       # 字体颜色
                    outline_color,    # 描边颜色
                    bottom_margin,    # 底部边距
                ],
                outputs=[effect_preview],
            )

            # 发布到抖音（自动 LLM 生成标题/文案/标签 + 选 BGM）
            Post_on_DY.click(
                auto_publish_dy_with_llm,
                inputs=[video_output, text_input, bgm_list, user_upload_bgm, pulish_with_cover],
                outputs=[status_output],
            )
            # 发布到小红书
            Post_on_XHS.click(
                auto_publishing_videos_XHS,
                inputs=[video_output, two_line_input, pulish_with_cover],
                outputs=[status_output],
            )

            # 发布到视频号
            Post_on_SPH.click(
                auto_publishing_videos_SPH,
                inputs=[video_output, two_line_input, pulish_with_cover],
                outputs=[status_output],
            )

            # 一键发布到抖音小红书视频号
            Post_on_ALL.click(
                auto_publish_dy_with_llm,
                inputs=[video_output, text_input, bgm_list, user_upload_bgm, pulish_with_cover],
                outputs=[status_output],
            )

            # 更新确认界面
            with gr.Row(visible=False) as update_dialog:
                with gr.Column():
                    update_info = gr.Textbox(
                        label="更新信息", interactive=False, lines=8
                    )
                    with gr.Row():
                        confirm_btn = gr.Button("确认更新", variant="primary")
                        cancel_btn = gr.Button("取消")

            # 更新状态显示
            update_status = gr.Textbox(label="更新状态", interactive=False, lines=10)

            # 更新按钮
            update_elements_btn = gr.Button("检查更新")

            start_info = gr.Textbox(label="启动信息", interactive=False, lines=10)

            start_digit_human_button = gr.Button("启动digit_human")
            start_digit_human_button.click(
                start_digit_human,
                inputs=[account],
                outputs=[start_info],
            )
            start_cosyvoice_button = gr.Button("启动cosyvoice")
            start_cosyvoice_button.click(
                start_cosyvoice,
                inputs=[account],
                outputs=[start_info],
            )

            # 绑定事件
            update_elements_btn.click(
                fn=update_platform_elements, outputs=[update_info, update_dialog]
            )

            confirm_btn.click(fn=do_update, outputs=[update_status, update_dialog])

            cancel_btn.click(
                fn=cancel_update,  # 使用具名函数替代lambda
                outputs=[update_status, update_dialog],
            )

            # 保存API Key
            save_api_key_button.click(
                fn=save_api_key,
                inputs=[api_key_input],
                outputs=[status_output, api_key],  # 更新状态和下拉框
            )

            # 删除API Key
            delete_api_key_button.click(
                fn=delete_api_key,
                inputs=[api_key],
                outputs=[status_output, api_key],  # 更新状态和下拉框
            )

            # 刷新key
            refresh_api_key_button.click(fn=refresh_api_key, outputs=[api_key])

        # 添加一个说明文本，表明这是API服务
        gr.Markdown(
            """
        # API服务已启动
        此服务仅提供API访问，Web界面已禁用。
        """
        )
        
        # 移除特效字幕相关API接口
        #     )
        #     
        #     # 设置默认模板的API接口
        #     )
        #     
        #     # 获取默认模板的API接口
        #     )

    return demo
