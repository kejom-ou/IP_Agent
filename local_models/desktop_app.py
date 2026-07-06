"""
============================================================
口播智能体 - Windows 桌面版（PySide6）
============================================================
QQ/微信 风格原生桌面应用：

  ① 左侧侧边栏导航（带图标）
  ② 右侧内容区（QStackedWidget 切换页面）
  ③ 系统托盘（点击 X → 最小化到托盘，双击托盘图标 → 恢复窗口）
  ④ 托盘右键菜单（显示窗口 / 退出）
  ⑤ 5 个功能页面 + 一键全流程

使用方法:
    python local_models/desktop_app.py
    或双击 一键启动桌面版.bat

依赖:
    pip install PySide6
============================================================
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional

# ── 项目根目录 ──
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ── Windows: 提前切换事件循环 ──
if sys.platform == "win32":
    import asyncio as _asyncio
    try:
        _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("desktop_app")

# ===========================================================================
# PySide6 导入
# ===========================================================================
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QTextEdit, QLineEdit,
        QComboBox, QSlider, QFileDialog, QProgressBar,
        QGroupBox, QGridLayout, QStackedWidget,
        QSystemTrayIcon, QMenu, QFrame,
    )
    from PySide6.QtCore import (
        Qt, QThread, Signal, QUrl,
    )
    from PySide6.QtGui import (
        QColor, QPalette, QIcon, QFont, QAction, QPixmap, QPainter,
    )
    from PySide6.QtMultimedia import QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


# ===========================================================================
# 现代扁平化样式表
# ===========================================================================
APP_QSS = """
/* ── 全局 ── */
* {
    font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
}

/* ── 侧边栏 ── */
#sidebar {
    background-color: #2d2d3f;
    border-right: 1px solid #1e1e2d;
}
#sidebarBtn {
    background-color: transparent;
    color: #b0b0c8;
    border: none;
    border-radius: 8px;
    text-align: left;
    padding: 10px 14px;
    font-size: 13px;
}
#sidebarBtn:hover {
    background-color: #3d3d56;
    color: #ffffff;
}
#sidebarBtn:checked, #sidebarBtn[active="true"] {
    background-color: #4f4fb8;
    color: #ffffff;
}
#sidebarBtn QLabel {
    color: inherit;
    font-size: 13px;
}

/* ── 头部栏 ── */
#headerBar {
    background-color: #ffffff;
    border-bottom: 1px solid #e8e8ec;
}

/* ── 内容区 ── */
#contentArea {
    background-color: #f5f6fa;
}
#pageTitle {
    font-size: 18px;
    font-weight: bold;
    color: #1a1a2e;
    padding: 4px 0;
}
#pageDesc {
    font-size: 12px;
    color: #8888a0;
    padding: 2px 0 8px 0;
}

/* ── 分组框 ── */
QGroupBox {
    font-size: 13px;
    font-weight: bold;
    color: #333;
    border: 1px solid #e0e0e8;
    border-radius: 10px;
    margin-top: 14px;
    padding: 16px 14px 12px 14px;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: #4f4fb8;
}

/* ── 输入框 ── */
QTextEdit, QLineEdit {
    border: 1px solid #d8d8e0;
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 13px;
    background: #fafafc;
}
QTextEdit:focus, QLineEdit:focus {
    border-color: #4f4fb8;
    background: #ffffff;
}
QComboBox {
    border: 1px solid #d8d8e0;
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 13px;
    background: #fafafc;
    min-height: 20px;
}
QComboBox:hover {
    border-color: #4f4fb8;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox QAbstractItemView {
    border: 1px solid #e0e0e8;
    border-radius: 6px;
    padding: 4px;
    selection-background-color: #f0f0ff;
}

/* ── 按钮 ── */
QPushButton {
    background-color: #f0f0f5;
    border: 1px solid #e0e0e8;
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 13px;
    color: #333;
}
QPushButton:hover {
    background-color: #e8e8f8;
    border-color: #4f4fb8;
}
QPushButton:pressed {
    background-color: #d8d8f0;
}

/* 主要操作按钮 */
QPushButton#primaryBtn {
    background-color: #4f4fb8;
    color: #ffffff;
    border: none;
    font-weight: bold;
    padding: 10px 28px;
    font-size: 14px;
}
QPushButton#primaryBtn:hover {
    background-color: #4444a8;
}
QPushButton#primaryBtn:pressed {
    background-color: #3a3a98;
}

/* 危险/发布按钮 */
QPushButton#accentBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #ff6b6b, stop:1 #ff3d67);
    color: #ffffff;
    border: none;
    font-weight: bold;
    padding: 10px 28px;
    font-size: 14px;
}
QPushButton#accentBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #e85d5d, stop:1 #e5345a);
}

/* 一键全流程按钮 */
QPushButton#fullRunBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #667eea, stop:1 #764ba2);
    color: #ffffff;
    border: none;
    font-size: 15px;
    font-weight: bold;
    padding: 12px 36px;
    border-radius: 10px;
}
QPushButton#fullRunBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #5a6fd6, stop:1 #6a4190);
}

/* ── 进度条 ── */
QProgressBar {
    border: none;
    border-radius: 6px;
    text-align: center;
    height: 6px;
    font-size: 11px;
    background-color: #e8e8f0;
}
QProgressBar::chunk {
    background-color: #4f4fb8;
    border-radius: 6px;
}

/* ── 状态标签 ── */
QLabel#statusLabel {
    color: #8888a0;
    font-size: 12px;
    padding: 4px 0;
}
QLabel#stepLabel {
    color: #4f4fb8;
    font-size: 20px;
    font-weight: bold;
}

/* ── 底部状态栏 ── */
#statusBar {
    background-color: #ffffff;
    border-top: 1px solid #e8e8ec;
    padding: 6px 16px;
}

/* ── 滚动条 ── */
QScrollBar:vertical {
    width: 6px;
    background: transparent;
}
QScrollBar::handle:vertical {
    background: #d0d0d8;
    border-radius: 3px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #b0b0c0;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""


# ===========================================================================
# 异步工作线程
# ===========================================================================
class WorkerThread(QThread):
    """后台工作线程（支持安全中断）"""
    started = Signal(str)
    progress = Signal(str, int)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, task_func, task_name: str, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.task_name = task_name
        self.kwargs = kwargs
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True
        if self.isRunning():
            self.terminate()
            self.wait(3000)

    def run(self):
        if self._is_cancelled:
            return
        self.started.emit(f"⏳ {self.task_name} 开始...")
        self.progress.emit(f"执行中: {self.task_name}", 0)
        try:
            if self._is_cancelled:
                return
            result = self.task_func(**self.kwargs)
            if not self._is_cancelled:
                self.finished.emit(
                    result if isinstance(result, dict) else {"result": result}
                )
                self.progress.emit(f"✅ {self.task_name} 完成", 100)
        except Exception as e:
            logger.error(f"[{self.task_name}] 失败: {e}", exc_info=True)
            if not self._is_cancelled:
                self.error.emit(f"❌ {self.task_name} 失败: {str(e)}")


# ===========================================================================
# 侧边栏按钮
# ===========================================================================
class SidebarButton(QPushButton):
    """侧边栏导航按钮"""
    def __init__(self, icon_text: str, label: str, index: int, parent=None):
        super().__init__(parent)
        self._index = index
        self._label_text = label
        self.setObjectName("sidebarBtn")
        self.setText(f"  {icon_text}  {label}")
        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(True)
        self.setMinimumHeight(44)


# ===========================================================================
# 主窗口
# ===========================================================================
class DesktopApp(QMainWindow):
    SIDEBAR_BTNS = [
        ("📹", "视频输入"),
        ("✍️", "AI 仿写"),
        ("🔊", "语音合成"),
        ("🎭", "口型合成"),
        ("📺", "预览导出"),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("口播智能体")
        self.resize(1100, 750)
        self.setMinimumSize(960, 640)

        # 内部状态
        self.video_path: Optional[str] = None
        self.original_text: str = ""
        self.rewritten_text: str = ""
        self.video_title: str = ""
        self.video_tags: str = ""
        self.audio_path: Optional[str] = None
        self.final_video: Optional[str] = None
        self.worker: Optional[WorkerThread] = None
        self._quitting = False  # 区分"最小化到托盘"和"真正退出"

        # 构建 UI
        self._init_ui()
        self._apply_style()
        self._create_tray()

    # ── 窗口关闭：最小化到托盘（像 QQ/微信一样） ──
    def closeEvent(self, event):
        """点击 X → 最小化到系统托盘，不退出进程"""
        if self._quitting:
            # 真正退出
            self._do_cleanup()
            event.accept()
        else:
            # 最小化到托盘
            self.hide()
            if hasattr(self, "tray_icon"):
                self.tray_icon.showMessage(
                    "口播智能体",
                    "已最小化到系统托盘，双击图标恢复窗口",
                    QSystemTrayIcon.Information,
                    2000,
                )
            event.ignore()

    def _do_cleanup(self):
        """退出前清理所有资源（线程 / 子进程 / GPU模型单例 / 显存）"""
        logger.info("🛑 正在退出程序...")

        # ── 1. 停止媒体播放器 ──
        if hasattr(self, "media_player"):
            try:
                self.media_player.stop()
                self.media_player.setVideoOutput(None)
            except Exception:
                pass

        # ── 2. 取消后台工作线程 ──
        if self.worker and self.worker.isRunning():
            try:
                self.worker.cancel()
            except Exception as e:
                logger.warning(f"取消任务异常: {e}")

        # ── 3. 卸载所有 GPU 模型单例 (ASR / LLM / TTS / LipSync) ──
        # 必须先 del Python 对象让 CUDA tensor 引用归零，
        # 再 gc.collect() + empty_cache() 才能真正释放显存。
        try:
            from local_models.pipeline_gradio import (
                _unload_asr, _unload_llm, _unload_tts, _unload_lipsync,
            )
            _unload_lipsync()   # MuseTalk ~6-8 GB（优先级最高）
            _unload_tts()       # CosyVoice
            _unload_llm()       # Qwen INT4 ~0.5 GB
            _unload_asr()       # FunASR ~1-2 GB
            logger.info("🧹 所有 GPU 模型单例已卸载")
        except Exception as e:
            logger.warning(f"卸载模型单例异常: {e}")

        # ── 4. 杀孤儿子进程 ──
        orphan_patterns = ["ffmpeg", "ffprobe", "chromium", "chrome", "playwright"]
        try:
            import psutil
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    pname = (proc.info["name"] or "").lower()
                    if not any(pat in pname for pat in orphan_patterns):
                        continue
                    proc.kill()
                    logger.info(f"🔪 已终止子进程: {pname}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except ImportError:
            try:
                import subprocess
                subprocess.run(["taskkill", "/F", "/IM", "ffmpeg.exe"],
                               capture_output=True, timeout=5)
            except Exception:
                pass

        # ── 5. 强制回收已释放的 CUDA 显存 + Python GC ──
        try:
            import gc
            import torch
            gc.collect()          # 先回收 Python 侧已 del 的对象
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                vram = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"🧹 CUDA 显存已释放 (残余: {vram:.2f} GB)")
            gc.collect()          # 二次回收（empty_cache 后可能触发新的弱引用）
        except ImportError:
            pass

        # ── 6. 隐藏托盘图标 ──
        if hasattr(self, "tray_icon"):
            self.tray_icon.hide()

        logger.info("✅ 程序已退出")

    # ── 系统托盘 ──
    def _create_tray(self):
        """创建系统托盘图标"""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("系统托盘不可用")
            return

        # 使用纯色图标（PySide6 自绘）
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(79, 79, 184))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(4, 4, 24, 24, 6, 6)
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Microsoft YaHei", 14, QFont.Bold)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "AI")
        painter.end()
        icon = QIcon(pixmap)

        self.tray_icon = QSystemTrayIcon(icon, self)
        self.tray_icon.setToolTip("口播智能体")

        # 右键菜单
        menu = QMenu()
        show_action = QAction("📋 显示窗口", self)
        show_action.triggered.connect(self._show_from_tray)
        menu.addAction(show_action)
        menu.addSeparator()
        quit_action = QAction("❌ 退出程序", self)
        quit_action.triggered.connect(self._quit_app)
        menu.addAction(quit_action)
        self.tray_icon.setContextMenu(menu)

        # 双击托盘图标 → 恢复窗口
        self.tray_icon.activated.connect(self._on_tray_activated)

        self.tray_icon.show()

    def _on_tray_activated(self, reason):
        """托盘图标交互"""
        if reason == QSystemTrayIcon.DoubleClick:
            self._show_from_tray()

    def _show_from_tray(self):
        """从托盘恢复窗口"""
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def _quit_app(self):
        """真正退出程序"""
        self._quitting = True
        self.close()

    # ── UI 初始化 ──
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── 头部栏 ──
        root_layout.addWidget(self._create_header())

        # ── 主体：侧边栏 + 内容区 ──
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._create_sidebar())
        body.addWidget(self._create_content_area(), 1)
        root_layout.addLayout(body, 1)

        # ── 底部状态栏 ──
        root_layout.addWidget(self._create_status_bar())

    def _create_header(self):
        """顶部标题栏"""
        frame = QFrame()
        frame.setObjectName("headerBar")
        frame.setFixedHeight(50)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 0, 16, 0)

        # Logo
        logo = QLabel("🎬  口播智能体")
        logo.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        logo.setStyleSheet("color: #1a1a2e;")
        layout.addWidget(logo)

        layout.addStretch()

        # 一键全流程（头部快捷按钮）
        self.header_run_btn = QPushButton("🚀 一键全流程")
        self.header_run_btn.setObjectName("primaryBtn")
        self.header_run_btn.setFixedHeight(34)
        self.header_run_btn.clicked.connect(self._on_full_pipeline)
        layout.addWidget(self.header_run_btn)

        return frame

    def _create_sidebar(self):
        """左侧侧边栏（QQ/微信风格）"""
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(170)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(4)

        # 导航按钮组
        self.sidebar_buttons: list[SidebarButton] = []
        for i, (icon, label) in enumerate(self.SIDEBAR_BTNS):
            btn = SidebarButton(icon, label, i)
            btn.clicked.connect(lambda checked, idx=i: self._switch_page(idx))
            layout.addWidget(btn)
            self.sidebar_buttons.append(btn)

        layout.addStretch()

        # 底部版本信息
        ver = QLabel("v1.0.0")
        ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet("color: #6a6a8a; font-size: 11px;")
        layout.addWidget(ver)

        return sidebar

    def _create_content_area(self):
        """右侧内容区（QStackedWidget 切换页面）"""
        self.stack = QStackedWidget()
        self.stack.setObjectName("contentArea")

        # 创建 5 个页面
        self._create_page_input()       # 0: 视频输入
        self._create_page_rewrite()     # 1: AI 仿写
        self._create_page_tts()         # 2: TTS 语音合成
        self._create_page_lipsync()     # 3: 口型合成
        self._create_page_preview()     # 4: 预览导出

        self.stack.setCurrentIndex(0)
        self.sidebar_buttons[0].setChecked(True)
        return self.stack

    def _switch_page(self, index: int):
        """切换内容页面"""
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.sidebar_buttons):
            btn.setChecked(i == index)
            btn.setProperty("active", i == index)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _create_page_wrapper(self, title: str, desc: str) -> tuple[QWidget, QVBoxLayout]:
        """创建标准页面容器"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        layout.addWidget(title_label)

        desc_label = QLabel(desc)
        desc_label.setObjectName("pageDesc")
        layout.addWidget(desc_label)

        return page, layout

    def _add_progress(self, parent_layout, name: str) -> QProgressBar:
        """添加隐藏进度条"""
        bar = QProgressBar()
        bar.setVisible(False)
        bar.setMaximumHeight(6)
        parent_layout.addWidget(bar)
        setattr(self, name, bar)
        return bar

    def _add_status(self, parent_layout, name: str, text: str = "就绪"):
        """添加状态标签"""
        label = QLabel(text)
        label.setObjectName("statusLabel")
        parent_layout.addWidget(label)
        setattr(self, name, label)
        return label

    # ================================================================
    # 页面 0: 视频输入
    # ================================================================
    def _create_page_input(self):
        page, layout = self._create_page_wrapper(
            "📹 视频输入", "输入抖音链接或选择本地视频，自动提取口播文案"
        )

        # ── 抖音链接 ──
        grp_url = QGroupBox("🔗 抖音链接下载")
        url_layout = QHBoxLayout(grp_url)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("粘贴抖音分享链接，例如: https://v.douyin.com/xxxxx/")
        self.url_input.setMinimumHeight(38)
        url_layout.addWidget(self.url_input, 3)
        self.download_btn = QPushButton("📥 下载")
        self.download_btn.setObjectName("primaryBtn")
        self.download_btn.clicked.connect(self._on_download)
        url_layout.addWidget(self.download_btn, 1)
        layout.addWidget(grp_url)

        # ── 本地视频 ──
        grp_video = QGroupBox("📁 本地视频 & 文案提取")
        v_layout = QHBoxLayout(grp_video)
        v_layout.setSpacing(16)

        left = QVBoxLayout()
        self.upload_btn = QPushButton("📂 选择本地视频/音频")
        self.upload_btn.clicked.connect(self._on_upload_video)
        left.addWidget(self.upload_btn)
        self.video_path_label = QLabel("未选择文件")
        self.video_path_label.setObjectName("statusLabel")
        self.video_path_label.setWordWrap(True)
        left.addWidget(self.video_path_label)
        left.addStretch()
        v_layout.addLayout(left, 1)

        right = QVBoxLayout()
        right.setSpacing(6)
        self.extract_btn = QPushButton("🔍 提取视频文案")
        self.extract_btn.setObjectName("primaryBtn")
        self.extract_btn.clicked.connect(self._on_extract_text)
        right.addWidget(self.extract_btn)

        self._add_status(right, "asr_status", "等待操作")

        self.original_text_edit = QTextEdit()
        self.original_text_edit.setPlaceholderText("提取后文案将显示在这里，也可以手动粘贴...")
        self.original_text_edit.setMaximumHeight(130)
        right.addWidget(self.original_text_edit)
        v_layout.addLayout(right, 3)
        layout.addWidget(grp_video)

        self._add_progress(layout, "input_progress")
        layout.addStretch()
        self.stack.addWidget(page)

    # ================================================================
    # 页面 1: AI 仿写
    # ================================================================
    def _create_page_rewrite(self):
        page, layout = self._create_page_wrapper(
            "✍️ AI 仿写", "基于原文案用大模型仿写，自动生成标题和标签"
        )

        grp = QGroupBox("仿写设置")
        g_layout = QGridLayout(grp)
        g_layout.setSpacing(10)

        g_layout.addWidget(QLabel("仿写模式:"), 0, 0)
        self.rewrite_mode = QComboBox()
        self.rewrite_mode.addItems(["AI自动仿写", "根据指令仿写"])
        self.rewrite_mode.currentTextChanged.connect(self._on_rewrite_mode_changed)
        self.rewrite_mode.setMinimumHeight(36)
        g_layout.addWidget(self.rewrite_mode, 0, 1)

        self.custom_prompt_input = QTextEdit()
        self.custom_prompt_input.setPlaceholderText("输入仿写指令，例如：用幽默口吻改写，加入网络热梗")
        self.custom_prompt_input.setMaximumHeight(60)
        self.custom_prompt_input.setVisible(False)
        g_layout.addWidget(self.custom_prompt_input, 1, 0, 1, 2)

        self.rewrite_btn = QPushButton("🔄 执行 AI 仿写")
        self.rewrite_btn.setObjectName("primaryBtn")
        self.rewrite_btn.clicked.connect(self._on_rewrite)
        g_layout.addWidget(self.rewrite_btn, 2, 0, 1, 2)

        self._add_status_page(g_layout, 3, "rewrite_status", "等待操作")

        g_layout.addWidget(QLabel("仿写结果:"), 4, 0, 1, 2)
        self.rewritten_text_edit = QTextEdit()
        self.rewritten_text_edit.setPlaceholderText("仿写后的文案将显示在这里...")
        self.rewritten_text_edit.setMaximumHeight(100)
        g_layout.addWidget(self.rewritten_text_edit, 5, 0, 1, 2)

        g_layout.addWidget(QLabel("📌 视频标题:"), 6, 0)
        self.title_output = QLineEdit()
        self.title_output.setPlaceholderText("AI 自动生成")
        self.title_output.setMinimumHeight(34)
        g_layout.addWidget(self.title_output, 6, 1)

        g_layout.addWidget(QLabel("🏷️ 标签:"), 7, 0)
        self.tags_output = QLineEdit()
        self.tags_output.setPlaceholderText("AI 自动生成")
        self.tags_output.setMinimumHeight(34)
        g_layout.addWidget(self.tags_output, 7, 1)

        layout.addWidget(grp)
        self._add_progress(layout, "rewrite_progress")
        layout.addStretch()
        self.stack.addWidget(page)

    def _add_status_page(self, layout, row, name, text):
        """在 GridLayout 中添加状态标签"""
        label = QLabel(text)
        label.setObjectName("statusLabel")
        layout.addWidget(label, row, 0, 1, 2)
        setattr(self, name, label)

    # ================================================================
    # 页面 2: TTS 语音合成
    # ================================================================
    def _create_page_tts(self):
        page, layout = self._create_page_wrapper(
            "🔊 语音合成", "CosyVoice 文本转语音，支持内置音色和自定义参考音频"
        )

        grp = QGroupBox("TTS 设置")
        g_layout = QGridLayout(grp)
        g_layout.setSpacing(10)

        g_layout.addWidget(QLabel("音色来源:"), 0, 0)
        self.tts_mode = QComboBox()
        self.tts_mode.addItems(["内置音色", "上传音频", "录制音频"])
        self.tts_mode.currentTextChanged.connect(self._on_tts_mode_changed)
        self.tts_mode.setMinimumHeight(36)
        g_layout.addWidget(self.tts_mode, 0, 1)

        g_layout.addWidget(QLabel("内置音色:"), 1, 0)
        self.speaker_dd = QComboBox()
        self.speaker_dd.addItems(["中文女", "中文男", "默认女声", "默认男声"])
        self.speaker_dd.setMinimumHeight(36)
        g_layout.addWidget(self.speaker_dd, 1, 1)

        self.upload_audio_label = QLabel("上传参考音频:")
        self.upload_audio_label.setVisible(False)
        g_layout.addWidget(self.upload_audio_label, 2, 0)
        self.upload_audio_btn = QPushButton("选择音频文件")
        self.upload_audio_btn.setVisible(False)
        self.upload_audio_btn.clicked.connect(self._on_upload_audio)
        g_layout.addWidget(self.upload_audio_btn, 2, 1)
        self.upload_audio_path = QLabel("")
        self.upload_audio_path.setObjectName("statusLabel")
        self.upload_audio_path.setVisible(False)
        g_layout.addWidget(self.upload_audio_path, 3, 1)

        g_layout.addWidget(QLabel("参考音频文本:"), 4, 0)
        self.custom_audio_text = QLineEdit()
        self.custom_audio_text.setPlaceholderText("不填则使用默认文本")
        self.custom_audio_text.setMinimumHeight(34)
        g_layout.addWidget(self.custom_audio_text, 4, 1)

        g_layout.addWidget(QLabel("语速:"), 5, 0)
        speed_row = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(5, 20)
        self.speed_slider.setValue(10)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(1)
        speed_row.addWidget(self.speed_slider, 3)
        self.speed_label = QLabel("1.0x")
        self.speed_label.setMinimumWidth(40)
        speed_row.addWidget(self.speed_label)
        g_layout.addLayout(speed_row, 5, 1)
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_label.setText(f"{v / 10:.1f}x")
        )

        self.tts_btn = QPushButton("🔊 生成语音")
        self.tts_btn.setObjectName("primaryBtn")
        self.tts_btn.clicked.connect(self._on_tts)
        g_layout.addWidget(self.tts_btn, 6, 0, 1, 2)

        self._add_status_page(g_layout, 7, "tts_status", "等待操作")

        layout.addWidget(grp)
        self._add_progress(layout, "tts_progress")
        layout.addStretch()
        self.stack.addWidget(page)

    # ================================================================
    # 页面 3: 口型合成
    # ================================================================
    def _create_page_lipsync(self):
        page, layout = self._create_page_wrapper(
            "🎭 口型合成", "MuseTalk 数字人口型驱动，音频 + 人物视频 → 口型同步视频"
        )

        grp = QGroupBox("口型合成设置")
        g_layout = QVBoxLayout(grp)
        g_layout.setSpacing(10)

        row = QHBoxLayout()
        row.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(6)
        self.avatar_btn = QPushButton("🖼️ 选择数字人形象视频/图片")
        self.avatar_btn.clicked.connect(self._on_upload_avatar)
        left.addWidget(self.avatar_btn)
        self.avatar_label = QLabel("未选择（将使用参考视频）")
        self.avatar_label.setObjectName("statusLabel")
        self.avatar_label.setWordWrap(True)
        left.addWidget(self.avatar_label)
        left.addStretch()
        row.addLayout(left, 1)

        right = QVBoxLayout()
        right.setSpacing(6)
        self.lipsync_btn = QPushButton("🎬 生成口型视频")
        self.lipsync_btn.setObjectName("primaryBtn")
        self.lipsync_btn.clicked.connect(self._on_lipsync)
        right.addWidget(self.lipsync_btn)
        self._add_status(right, "lipsync_status", "等待操作")
        right.addStretch()
        row.addLayout(right, 1)

        g_layout.addLayout(row)
        layout.addWidget(grp)

        self._add_progress(layout, "lipsync_progress")
        layout.addStretch()
        self.stack.addWidget(page)

    # ================================================================
    # 页面 4: 预览导出
    # ================================================================
    def _create_page_preview(self):
        page, layout = self._create_page_wrapper(
            "📺 预览 & 导出", "预览生成的最终视频，导出到本地或发布到抖音"
        )

        grp = QGroupBox("视频预览")
        g_layout = QVBoxLayout(grp)
        g_layout.setSpacing(10)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(340)
        self.video_widget.setStyleSheet("background-color: #000; border-radius: 10px;")
        g_layout.addWidget(self.video_widget, 1)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)
        self.play_btn = QPushButton("▶ 播放")
        self.play_btn.clicked.connect(self._on_play)
        ctrl_row.addWidget(self.play_btn)
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.clicked.connect(self._on_pause)
        ctrl_row.addWidget(self.pause_btn)
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.clicked.connect(self._on_stop)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addStretch()
        self.save_btn = QPushButton("💾 导出到本地")
        self.save_btn.setObjectName("primaryBtn")
        self.save_btn.clicked.connect(self._on_save_video)
        ctrl_row.addWidget(self.save_btn)
        g_layout.addLayout(ctrl_row)

        self._add_status(g_layout, "preview_status", "尚未生成视频")
        layout.addWidget(grp)

        # 抖音发布按钮
        pub_row = QHBoxLayout()
        self.publish_btn = QPushButton("📤 发布到抖音")
        self.publish_btn.setObjectName("accentBtn")
        self.publish_btn.clicked.connect(self._on_publish)
        pub_row.addWidget(self.publish_btn)
        pub_row.addStretch()
        layout.addLayout(pub_row)

        # 播放器
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.errorOccurred.connect(
            lambda e: self.preview_status.setText(
                f"播放错误: {self.media_player.errorString()}"
            )
        )

        layout.addStretch()
        self.stack.addWidget(page)

    # ================================================================
    # 底部状态栏
    # ================================================================
    def _create_status_bar(self):
        frame = QFrame()
        frame.setObjectName("statusBar")
        frame.setFixedHeight(36)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 0, 16, 0)

        self.global_status = QLabel("👋 就绪  |  点击右上角 X 可最小化到托盘")
        self.global_status.setObjectName("statusLabel")
        layout.addWidget(self.global_status)

        layout.addStretch()

        # 当前步骤指示
        self.step_indicator = QLabel("")
        self.step_indicator.setObjectName("statusLabel")
        layout.addWidget(self.step_indicator)

        return frame

    # ================================================================
    # 样式
    # ================================================================
    def _apply_style(self):
        self.setStyleSheet(APP_QSS)

    # ================================================================
    # 辅助方法
    # ================================================================
    def _set_busy(self, busy: bool, progress_bar: QProgressBar = None):
        for btn in self.findChildren(QPushButton):
            if btn.objectName() in ("sidebarBtn",):
                continue
            btn.setEnabled(not busy)
        if progress_bar:
            progress_bar.setVisible(busy)
            progress_bar.setRange(0, 0 if busy else 100)

    def _run_worker(self, task_func, task_name: str, on_finished,
                    progress_bar: QProgressBar = None, **kwargs):
        if self.worker and self.worker.isRunning():
            logger.warning(f"已有任务在运行，正在取消: {self.worker.task_name}")
            self.worker.cancel()
            self.worker = None

        self._set_busy(True, progress_bar)
        self.global_status.setText(f"⏳ {task_name} 执行中...")
        self.worker = WorkerThread(task_func, task_name, **kwargs)
        self.worker.finished.connect(lambda r: on_finished(r))
        self.worker.finished.connect(
            lambda _: self._set_busy(False, progress_bar)
        )
        self.worker.error.connect(
            lambda e: self._on_worker_error(e, progress_bar)
        )
        self.worker.error.connect(
            lambda _: self._set_busy(False, progress_bar)
        )
        self.worker.progress.connect(
            lambda msg, p: self.global_status.setText(msg)
        )
        self.worker.start()

    def _on_worker_error(self, msg: str, progress_bar: QProgressBar = None):
        self.global_status.setText(msg)
        if progress_bar:
            progress_bar.setVisible(False)

    # ================================================================
    # 事件处理
    # ================================================================

    # ── 页面 0: 视频输入 ──
    def _on_upload_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频或音频文件", "",
            "视频/音频 (*.mp4 *.mov *.avi *.mkv *.wav *.mp3 *.m4a *.flac);;所有文件 (*)"
        )
        if path:
            self.video_path = path
            self.video_path_label.setText(os.path.basename(path))
            self.asr_status.setText(f"✅ 已选择: {os.path.basename(path)}")
            logger.info(f"[上传] 文件: {path}")

    def _on_download(self):
        url = self.url_input.text().strip()
        if not url:
            self.asr_status.setText("❌ 请输入抖音链接")
            return
        self._run_worker(
            self.__do_download, "抖音下载",
            self.__on_download_done, self.input_progress,
            url=url,
        )

    def __do_download(self, url: str) -> dict:
        from local_models.pipeline_gradio import step0_download_douyin
        result = step0_download_douyin(url)
        if isinstance(result, tuple):
            return {"video_path": result[0], "status": result[1]}
        return {"video_path": result, "status": "下载完成"}

    def __on_download_done(self, r: dict):
        vp = r.get("video_path")
        if vp and os.path.exists(vp):
            self.video_path = vp if isinstance(vp, str) else str(vp)
            self.video_path_label.setText(os.path.basename(self.video_path))
            self.asr_status.setText(f"✅ {r.get('status', '下载完成')}")
            self._do_extract()
        else:
            self.asr_status.setText(f"❌ {r.get('status', '下载失败')}")

    def _on_extract_text(self):
        if not self.video_path:
            self.asr_status.setText("❌ 请先选择视频或下载抖音视频")
            return
        self._do_extract()

    def _do_extract(self):
        self._run_worker(
            self.__do_extract, "文案提取",
            self.__on_extract_done, self.input_progress,
            video_path=self.video_path,
        )

    def __do_extract(self, video_path: str) -> dict:
        from local_models.pipeline_gradio import step1_extract_text
        text, status = step1_extract_text(video_path)
        return {"text": text, "status": status}

    def __on_extract_done(self, r: dict):
        self.original_text = r.get("text", "")
        self.original_text_edit.setPlainText(self.original_text)
        self.asr_status.setText(r.get("status", "✅ 完成"))

    # ── 页面 1: AI 仿写 ──
    def _on_rewrite_mode_changed(self, mode: str):
        self.custom_prompt_input.setVisible(mode == "根据指令仿写")

    def _on_rewrite(self):
        text = self.original_text_edit.toPlainText().strip()
        if not text:
            self.rewrite_status.setText("❌ 请先提取或输入文案")
            return
        mode = self.rewrite_mode.currentText()
        prompt = (
            self.custom_prompt_input.toPlainText().strip()
            if mode == "根据指令仿写" else ""
        )
        self._run_worker(
            self.__do_rewrite, "AI 仿写",
            self.__on_rewrite_done, self.rewrite_progress,
            text=text, mode=mode, prompt=prompt,
        )

    def __do_rewrite(self, text: str, mode: str, prompt: str) -> dict:
        from local_models.pipeline_gradio import step2_rewrite
        result = step2_rewrite(text, mode, prompt)
        if isinstance(result, tuple) and len(result) >= 4:
            return {
                "rewritten": result[0], "title": result[1],
                "tags": result[2], "status": result[3],
            }
        return {"rewritten": text, "title": "", "tags": "", "status": "仿写完成"}

    def __on_rewrite_done(self, r: dict):
        self.rewritten_text = r.get("rewritten", "")
        self.video_title = r.get("title", "")
        self.video_tags = r.get("tags", "")
        self.rewritten_text_edit.setPlainText(self.rewritten_text)
        self.title_output.setText(self.video_title)
        self.tags_output.setText(self.video_tags)
        self.rewrite_status.setText(r.get("status", "✅ 完成"))

    # ── 页面 2: TTS ──
    def _on_tts_mode_changed(self, mode: str):
        is_custom = mode in ("上传音频", "录制音频")
        self.speaker_dd.setVisible(not is_custom)
        self.upload_audio_label.setVisible(mode == "上传音频")
        self.upload_audio_btn.setVisible(mode == "上传音频")
        self.upload_audio_path.setVisible(mode == "上传音频")

    def _on_upload_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择参考音频", "",
            "音频文件 (*.wav *.mp3 *.m4a *.flac);;所有文件 (*)"
        )
        if path:
            self.tts_custom_audio = path
            self.upload_audio_path.setText(os.path.basename(path))

    def _on_tts(self):
        text = self.rewritten_text_edit.toPlainText().strip()
        if not text:
            text = self.original_text_edit.toPlainText().strip()
        if not text:
            self.tts_status.setText("❌ 请先获取文案")
            return

        mode = self.tts_mode.currentText()
        speaker = self.speaker_dd.currentText() if mode == "内置音色" else None
        speed = self.speed_slider.value() / 10.0
        custom_audio = getattr(self, "tts_custom_audio", None)
        custom_text = self.custom_audio_text.text().strip()

        self._run_worker(
            self.__do_tts, "TTS 语音合成",
            self.__on_tts_done, self.tts_progress,
            text=text, speed=speed, voice_mode=mode,
            speaker=speaker, custom_audio=custom_audio,
            custom_audio_text=custom_text,
        )

    def __do_tts(self, text, speed, voice_mode, speaker,
                 custom_audio, custom_audio_text) -> dict:
        from local_models.pipeline_gradio import step3_generate_audio
        ca = custom_audio if voice_mode == "上传音频" else None
        audio_path, status = step3_generate_audio(
            text=text, speed=speed, voice_mode=voice_mode,
            speaker=speaker, custom_audio=ca,
            custom_audio_text=custom_audio_text,
        )
        return {"audio_path": audio_path, "status": status}

    def __on_tts_done(self, r: dict):
        self.audio_path = r.get("audio_path")
        self.tts_status.setText(r.get("status", ""))

    # ── 页面 3: 口型合成 ──
    def _on_upload_avatar(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数字人形象视频/图片", "",
            "视频/图片 (*.mp4 *.mov *.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)"
        )
        if path:
            self.avatar_path = path
            self.avatar_label.setText(os.path.basename(path))

    def _on_lipsync(self):
        avatar = getattr(self, "avatar_path", None)
        if not avatar and not self.video_path:
            self.lipsync_status.setText("❌ 请先上传数字人形象或参考视频")
            return
        if not self.audio_path:
            self.lipsync_status.setText("❌ 请先生成语音（TTS）")
            return

        self._run_worker(
            self.__do_lipsync, "MuseTalk 口型合成",
            self.__on_lipsync_done, self.lipsync_progress,
            avatar_path=avatar, audio_path=self.audio_path,
            fallback_video=self.video_path,
        )

    def __do_lipsync(self, avatar_path, audio_path, fallback_video) -> dict:
        from local_models.pipeline_gradio import step4_lipsync
        video_path, status = step4_lipsync(avatar_path, audio_path, fallback_video)
        return {"video_path": video_path, "status": status}

    def __on_lipsync_done(self, r: dict):
        self.final_video = r.get("video_path")
        self.lipsync_status.setText(r.get("status", ""))
        if self.final_video:
            self._switch_page(4)
            self.preview_status.setText(f"✅ {r.get('status', '生成完成')}")

    # ── 页面 4: 预览 ──
    def _on_play(self):
        if self.final_video and os.path.exists(self.final_video):
            self.media_player.setSource(QUrl.fromLocalFile(self.final_video))
            self.media_player.play()

    def _on_pause(self):
        self.media_player.pause()

    def _on_stop(self):
        self.media_player.stop()

    def _on_save_video(self):
        if not self.final_video:
            self.preview_status.setText("❌ 尚未生成视频")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出视频", "output.mp4",
            "视频文件 (*.mp4);;所有文件 (*)"
        )
        if save_path:
            try:
                import shutil
                shutil.copy2(self.final_video, save_path)
                self.preview_status.setText(f"✅ 已导出到: {save_path}")
            except Exception as e:
                self.preview_status.setText(f"❌ 导出失败: {e}")

    # ── 全流程 & 发布 ──
    def _on_full_pipeline(self):
        if not self.video_path:
            self.global_status.setText("❌ 请先选择视频或下载抖音视频")
            return
        avatar = getattr(self, "avatar_path", None)
        self._run_worker(
            self.__do_full_pipeline, "一键全流程",
            self.__on_full_done, None,
            video_path=self.video_path, avatar_path=avatar,
        )

    def __do_full_pipeline(self, video_path: str,
                           avatar_path: Optional[str] = None) -> dict:
        from local_models.pipeline_gradio import (
            step1_extract_text, step2_rewrite,
            step3_generate_audio, step4_lipsync,
        )

        items = []
        # Step 1
        items.append("[1/4] 提取文案...")
        text, status = step1_extract_text(video_path)
        if not text:
            items.append(f"❌ {status}"); return {"status": "\n".join(items)}
        items.append(f"✅ {status}")

        # Step 2
        items.append("[2/4] AI 仿写...")
        res = step2_rewrite(text, "AI自动仿写", "")
        rewritten = res[0]; title = res[1] if len(res) > 1 else ""
        tags = res[2] if len(res) > 2 else ""
        rstatus = res[3] if len(res) > 3 else "OK"
        items.append(f"✅ {rstatus}")

        # Step 3
        items.append("[3/4] 语音合成...")
        audio, tstatus = step3_generate_audio(
            text=rewritten, speed=1.0, voice_mode="内置音色", speaker="中文女",
        )
        if not audio:
            items.append(f"❌ {tstatus}")
            return {"status": "\n".join(items), "rewritten": rewritten,
                    "title": title, "tags": tags}
        items.append(f"✅ {tstatus}")

        # Step 4
        items.append("[4/4] 口型合成（MuseTalk）...")
        video, vstatus = step4_lipsync(avatar_path, audio, video_path)
        items.append(f"✅ {vstatus}")

        return {
            "status": "\n".join(items),
            "rewritten": rewritten, "title": title, "tags": tags,
            "audio_path": audio, "video_path": video,
        }

    def __on_full_done(self, r: dict):
        self.global_status.setText("✅ 全流程完成")
        self.rewritten_text = r.get("rewritten", "")
        self.video_title = r.get("title", "")
        self.video_tags = r.get("tags", "")
        self.rewritten_text_edit.setPlainText(self.rewritten_text)
        self.title_output.setText(self.video_title)
        self.tags_output.setText(self.video_tags)
        self.audio_path = r.get("audio_path")
        self.final_video = r.get("video_path")
        if self.final_video:
            self._switch_page(4)
            self.preview_status.setText("✅ 全流程完成")

    def _on_publish(self):
        if not self.final_video:
            self.global_status.setText("❌ 请先生成最终视频")
            return
        self._run_worker(
            self.__do_publish, "发布到抖音",
            self.__on_publish_done, None,
            final_video=self.final_video,
            rewritten_text=self.rewritten_text,
            title=self.video_title, tags=self.video_tags,
        )

    def __do_publish(self, final_video, rewritten_text, title, tags) -> dict:
        from local_models.pipeline_gradio import step5_publish
        result = step5_publish(final_video, rewritten_text, title, tags)
        if isinstance(result, tuple):
            return {"video_path": result[0], "status": result[1]}
        return {"status": str(result)}

    def __on_publish_done(self, r: dict):
        self.global_status.setText(f"✅ {r.get('status', '发布完成')}")


# ===========================================================================
# 启动入口
# ===========================================================================
def main():
    if not HAS_PYSIDE6:
        print("=" * 60)
        print("  请先安装 PySide6:  pip install PySide6")
        print("=" * 60)
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("口播智能体")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.WindowText, QColor(31, 41, 55))
    app.setPalette(palette)

    window = DesktopApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
