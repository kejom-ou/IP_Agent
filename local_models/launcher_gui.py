"""
============================================================
旗博士追爆智能体 — 图形化一键启动器
使用 Python 内置 tkinter，无需额外安装任何包
============================================================
"""

import os
import sys
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

# 项目根目录（脚本在 local_models/ 下）
ROOT_DIR = Path(__file__).parent.parent.absolute()
MODELS_DIR = ROOT_DIR / "pretrained_models"
REQ_FILE = ROOT_DIR / "local_models" / "requirements.txt"


# ---- 全局状态 ----
class State:
    def __init__(self):
        self.python_exe = sys.executable
        self.python_version = ""
        self.pip_ok = False
        self.deps_ok = False
        self.ffmpeg_ok = False
        self.imagemagick_ok = False
        self.models_ok = False
        self.playwright_ok = False
        self.all_checked = False
        self.all_ready = False


state = State()


# ============================================================
# 步骤检查函数
# ============================================================

def check_tool(name, win_cmd, unix_cmd="which"):
    """检查系统是否有某个命令行工具"""
    try:
        if sys.platform == "win32":
            result = subprocess.run(["where", name], capture_output=True, text=True)
        else:
            result = subprocess.run(["which", name], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def check_pip():
    try:
        subprocess.run(
            [state.python_exe, "-m", "pip", "--version"],
            capture_output=True, timeout=10,
        )
        return True
    except Exception:
        return False


def check_core_deps():
    try:
        subprocess.run(
            [state.python_exe, "-c",
             "import torch, gradio, funasr, modelscope, numpy, cv2, soundfile"],
            capture_output=True, timeout=30,
        )
        return True
    except Exception:
        return False


def check_playwright():
    try:
        subprocess.run(
            [state.python_exe, "-c",
             "from playwright.sync_api import sync_playwright; "
             "p = sync_playwright().start(); p.chromium.launch(); p.stop(); print('ok')"],
            capture_output=True, timeout=30,
        )
        return True
    except Exception:
        return False


def install_deps(log_callback):
    if not state.python_exe:
        return False

    log_callback("📦 正在安装依赖（首次约 5-15 分钟）...\n")

    # 升级 pip
    log_callback("  ⏳ 升级 pip...")
    try:
        subprocess.run(
            [state.python_exe, "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True, timeout=60,
        )
        log_callback(" ✅\n")
    except Exception:
        log_callback(" ⚠️ 跳过\n")

    # 安装 PyTorch（先装）
    log_callback("  ⏳ 安装 PyTorch...")
    try:
        subprocess.run(
            [state.python_exe, "-m", "pip", "install", "torch>=2.1.0",
             "--index-url", "https://download.pytorch.org/whl/cpu", "-q"],
            capture_output=True, timeout=300,
        )
        log_callback(" ✅\n")
    except Exception:
        try:
            subprocess.run(
                [state.python_exe, "-m", "pip", "install", "torch>=2.1.0", "-q"],
                capture_output=True, timeout=300,
            )
            log_callback(" ✅\n")
        except Exception:
            log_callback(" ❌\n")
            return False

    # 安装 requirements.txt
    if REQ_FILE.exists():
        log_callback("  ⏳ 安装其余依赖...")
        try:
            subprocess.run(
                [state.python_exe, "-m", "pip", "install", "-r", str(REQ_FILE), "-q"],
                capture_output=True, timeout=600,
            )
            log_callback(" ✅\n")
        except Exception:
            log_callback(" ⚠️ 部分失败，尝试逐条安装...\n")
            with open(REQ_FILE) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        pkg = line.split("#")[0].strip()
                        try:
                            subprocess.run(
                                [state.python_exe, "-m", "pip", "install", pkg, "-q"],
                                capture_output=True, timeout=120,
                            )
                        except Exception:
                            pass
            log_callback(" ✅\n")
    else:
        log_callback("  ⚠️ 未找到 requirements.txt，跳过\n")

    # Playwright 浏览器
    log_callback("  ⏳ 安装 Playwright 浏览器...")
    try:
        subprocess.run(
            [state.python_exe, "-m", "playwright", "install", "chromium"],
            capture_output=True, timeout=300,
        )
        log_callback(" ✅\n")
    except Exception:
        log_callback(" ⚠️ 跳过（抖音发布不可用）\n")

    log_callback("\n✅ 依赖安装完成\n")
    return True


def download_models(log_callback):
    download_script = ROOT_DIR / "local_models" / "download_models.py"
    if not download_script.exists():
        log_callback("❌ 未找到 download_models.py\n")
        return False

    log_callback("📥 开始下载模型（约 ~6GB，可能需要 10-30 分钟）...\n")
    try:
        process = subprocess.Popen(
            [state.python_exe, str(download_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT_DIR),
        )
        for line in process.stdout:
            log_callback(line)
        process.wait()
        if process.returncode == 0:
            log_callback("\n✅ 模型下载完成\n")
            return True
        else:
            log_callback("\n❌ 模型下载失败\n")
            return False
    except Exception as e:
        log_callback(f"\n❌ 下载出错: {e}\n")
        return False


def check_models():
    required = [
        MODELS_DIR / "CosyVoice-300M-SFT",
        MODELS_DIR / "SenseVoiceSmall",
    ]
    return all(m.exists() for m in required)


# ============================================================
# GUI
# ============================================================

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox


class LauncherGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("旗博士追爆智能体 - 一键启动")
        self.root.geometry("720x600")
        self.root.resizable(True, True)
        self.root.minsize(600, 500)

        # 居中
        self.root.update_idletasks()
        w, h = 720, 600
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        # 样式
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Microsoft YaHei", 18, "bold"))
        style.configure("Step.TLabel", font=("Microsoft YaHei", 11))
        style.configure("StatusOK.TLabel", foreground="#27ae60", font=("Microsoft YaHei", 11))
        style.configure("StatusWarn.TLabel", foreground="#e67e22", font=("Microsoft YaHei", 11))
        style.configure("StatusErr.TLabel", foreground="#e74c3c", font=("Microsoft YaHei", 11))
        style.configure("Launch.TButton", font=("Microsoft YaHei", 14, "bold"), padding=10)
        style.configure("Action.TButton", font=("Microsoft YaHei", 10), padding=5)

        self._build_ui()
        self._start_check()

    def _build_ui(self):
        # 标题
        ttk.Label(self.root, text="🚀 旗博士追爆智能体", style="Title.TLabel").pack(pady=(15, 5))
        ttk.Label(self.root, text=f"项目路径: {ROOT_DIR}").pack()

        # 环境检查区
        check_frame = ttk.LabelFrame(self.root, text="环境检查", padding=10)
        check_frame.pack(fill="x", padx=15, pady=10)

        self.status_labels = {}
        checks = [
            ("python", "Python 环境"),
            ("pip", "pip 包管理"),
            ("deps", "核心依赖"),
            ("playwright", "抖音发布 (Playwright)"),
            ("ffmpeg", "FFmpeg"),
            ("imagemagick", "ImageMagick"),
            ("models", "AI 模型文件"),
        ]

        for i, (key, label) in enumerate(checks):
            frame = ttk.Frame(check_frame)
            frame.pack(fill="x", pady=2)
            self.status_labels[key] = ttk.Label(
                frame, text="⏳ 检查中...", style="Step.TLabel"
            )
            self.status_labels[key].pack(side="left")
            ttk.Label(frame, text=label, style="Step.TLabel").pack(side="right")

        # 按钮区
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=15, pady=5)

        self.install_deps_btn = ttk.Button(
            btn_frame, text="📦 安装缺失依赖",
            style="Action.TButton", command=self._thread_install_deps,
        )
        self.install_deps_btn.pack(side="left", padx=5)

        self.download_models_btn = ttk.Button(
            btn_frame, text="📥 下载 AI 模型",
            style="Action.TButton", command=self._thread_download_models,
        )
        self.download_models_btn.pack(side="left", padx=5)

        self.recheck_btn = ttk.Button(
            btn_frame, text="🔄 重新检查",
            style="Action.TButton", command=self._thread_check_all,
        )
        self.recheck_btn.pack(side="left", padx=5)

        # 启动按钮
        self.launch_btn = ttk.Button(
            self.root, text="🚀 一键启动",
            style="Launch.TButton", command=self._launch,
        )
        self.launch_btn.pack(pady=10)
        self.launch_btn["state"] = "disabled"

        # 日志区
        log_frame = ttk.LabelFrame(self.root, text="运行日志", padding=5)
        log_frame.pack(fill="both", expand=True, padx=15, pady=(5, 10))

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=10, font=("Consolas", 9), wrap="word",
        )
        self.log_text.pack(fill="both", expand=True)

        # 底部
        ttk.Label(self.root, text="提示: 首次使用请先安装依赖和模型，之后即可一键启动").pack(
            pady=(0, 10)
        )

    def log(self, msg):
        self.log_text.insert("end", msg)
        self.log_text.see("end")
        self.log_text.update()

    def set_status(self, key, text, color_tag):
        self.status_labels[key].config(text=text)
        # 动态设置前景色
        color_map = {"ok": "#27ae60", "warn": "#e67e22", "err": "#e74c3c", "pending": "#7f8c8d"}
        self.status_labels[key].config(foreground=color_map.get(color_tag, "#000"))

    def _run_check_all(self):
        self.log("=" * 50 + "\n")
        self.log("🔍 开始环境检查...\n")

        # Python
        self.set_status("python", "⏳ 检查中...", "pending")
        try:
            result = subprocess.run(
                [state.python_exe, "--version"], capture_output=True, text=True, timeout=10,
            )
            state.python_version = result.stdout.strip()
            self.set_status("python", f"✅ {state.python_version}", "ok")
        except Exception:
            self.set_status("python", "❌ Python 不可用", "err")
            state.all_checked = True
            return

        # pip
        self.set_status("pip", "⏳ 检查中...", "pending")
        state.pip_ok = check_pip()
        self.set_status("pip",
                        "✅ 可用" if state.pip_ok else "❌ 不可用",
                        "ok" if state.pip_ok else "err")

        # 核心依赖
        self.set_status("deps", "⏳ 检查中...", "pending")
        state.deps_ok = check_core_deps()
        self.set_status("deps",
                        "✅ 已安装" if state.deps_ok else "⚠️ 缺失",
                        "ok" if state.deps_ok else "warn")

        # Playwright
        self.set_status("playwright", "⏳ 检查中...", "pending")
        state.playwright_ok = check_playwright()
        self.set_status("playwright",
                        "✅ 可用" if state.playwright_ok else "⚠️ 未安装",
                        "ok" if state.playwright_ok else "warn")

        # FFmpeg
        self.set_status("ffmpeg", "⏳ 检查中...", "pending")
        ffmpeg_local = (ROOT_DIR / "ffmpeg" / "bin" / "ffmpeg.exe").exists()
        state.ffmpeg_ok = ffmpeg_local or check_tool("ffmpeg")
        self.set_status("ffmpeg",
                        "✅ 可用" if state.ffmpeg_ok else "⚠️ 未安装",
                        "ok" if state.ffmpeg_ok else "warn")

        # ImageMagick
        self.set_status("imagemagick", "⏳ 检查中...", "pending")
        magick_local = (ROOT_DIR / "ImageMagick-7.1.1-Q16-HDRI" / "magick.exe").exists()
        state.imagemagick_ok = magick_local or check_tool("magick")
        self.set_status("imagemagick",
                        "✅ 可用" if state.imagemagick_ok else "⚠️ 未安装",
                        "ok" if state.imagemagick_ok else "warn")

        # 模型
        self.set_status("models", "⏳ 检查中...", "pending")
        state.models_ok = check_models()
        self.set_status("models",
                        "✅ 就绪" if state.models_ok else "❌ 缺失",
                        "ok" if state.models_ok else "err")

        state.all_checked = True

        # 判断是否可启动
        critical_ok = state.pip_ok and state.deps_ok and state.models_ok
        state.all_ready = critical_ok

        self.log(f"\n{'✅ 所有检查通过！可以启动' if critical_ok else '⚠️ 部分检查未通过，请修复后再启动'}\n")
        self.log("=" * 50 + "\n")

        if critical_ok:
            self.launch_btn["state"] = "normal"
        else:
            self.launch_btn["state"] = "disabled"

    def _thread_check_all(self):
        self.launch_btn["state"] = "disabled"
        threading.Thread(target=self._run_check_all, daemon=True).start()

    def _start_check(self):
        threading.Thread(target=self._run_check_all, daemon=True).start()

    def _run_install_deps(self):
        if not state.python_exe:
            self.log("❌ Python 不可用，无法安装依赖\n")
            return
        self.install_deps_btn["state"] = "disabled"
        self.download_models_btn["state"] = "disabled"
        install_deps(self.log)
        self.install_deps_btn["state"] = "normal"
        self.download_models_btn["state"] = "normal"
        # 重新检查
        self._run_check_all()

    def _thread_install_deps(self):
        self.launch_btn["state"] = "disabled"
        threading.Thread(target=self._run_install_deps, daemon=True).start()

    def _run_download_models(self):
        self.install_deps_btn["state"] = "disabled"
        self.download_models_btn["state"] = "disabled"
        download_models(self.log)
        self.install_deps_btn["state"] = "normal"
        self.download_models_btn["state"] = "normal"
        self._run_check_all()

    def _thread_download_models(self):
        self.launch_btn["state"] = "disabled"
        threading.Thread(target=self._run_download_models, daemon=True).start()

    # ============================================================
    # Chrome 启动 + 抖音登录引导
    # ============================================================
    def _find_chrome(self):
        """查找浏览器可执行文件（系统 Chrome > Playwright Chromium）"""
        # 1) 系统 Chrome / Chromium
        if sys.platform == "darwin":
            candidates = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            ]
        elif sys.platform == "win32":
            candidates = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                os.path.join(os.environ.get("LOCALAPPDATA", ""),
                             r"Google\Chrome\Application\chrome.exe"),
            ]
        else:
            candidates = [
                shutil.which("google-chrome"),
                shutil.which("google-chrome-stable"),
                shutil.which("chromium"),
                shutil.which("chromium-browser"),
            ]
            candidates = [c for c in candidates if c]

        for p in candidates:
            if p and os.path.exists(p):
                return p

        # 2) Playwright 自带的 Chromium（无需用户单独装 Chrome）
        try:
            from playwright.sync_api import sync_playwright
            pw = sync_playwright().start()
            try:
                browser = pw.chromium.launch(
                    headless=True,
                    args=["--no-sandbox"],
                )
                # 获取 Playwright Chromium 可执行文件路径
                exe_path = browser.version  # 实际上这返回版本号…
                # 真正的方式：用 pw.chromium.executable_path
                exe_path = pw.chromium.executable_path
                browser.close()
                pw.stop()
                if exe_path and os.path.exists(exe_path):
                    return exe_path
            except Exception:
                pw.stop()
        except Exception:
            pass

        return None

    def _get_chrome_profile_dir(self):
        """获取专用 Chrome profile 目录（用于持久化抖音登录状态）"""
        if sys.platform == "win32":
            base = os.environ.get("APPDATA", os.path.expanduser("~"))
        elif sys.platform == "darwin":
            base = os.path.expanduser("~/Library/Application Support")
        else:
            base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))

        profile_dir = os.path.join(base, "QBS_Agent", "chrome_profile")
        os.makedirs(profile_dir, exist_ok=True)
        return profile_dir

    def _launch(self):
        """启动主应用"""
        self.log("\n🚀 正在启动应用...\n")

        # 唯一启动入口：local_models/pipeline_gradio.py
        launch_script = ROOT_DIR / "local_models" / "pipeline_gradio.py"
        server_port = 7860

        if not launch_script.exists():
            messagebox.showerror("错误", "未找到启动脚本 pipeline_gradio.py！")
            return

        # ============================================================
        # Step 1: 启动 Chrome 调试模式 + 抖音登录引导
        # ============================================================
        chrome_ready = False

        if state.playwright_ok:
            chrome_path = self._find_chrome()
            if chrome_path:
                self.log("🔧 启动 Chrome 调试模式 (CDP 端口 9222)...\n")
                profile_dir = self._get_chrome_profile_dir()

                try:
                    subprocess.Popen(
                        [chrome_path,
                         "--remote-debugging-port=9222",
                         f"--user-data-dir={profile_dir}",
                         "--no-first-run",
                         "--no-default-browser-check",
                         "https://creator.douyin.com/"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self.log("✅ Chrome 已启动（独立 profile，登录状态会保留）\n")
                    self.log("📋 请在打开的浏览器中登录抖音创作者平台\n")
                    self.log("   （首次需输入账号密码，后续自动登录）\n\n")

                    # 弹出确认框，等用户手动登录
                    result = messagebox.askquestion(
                        "抖音登录确认",
                        "已打开抖音创作者平台登录页。\n\n"
                        "请确认已在浏览器中完成登录后，\n"
                        "点击「是」继续启动应用。\n\n"
                        "点击「否」将跳过自动发布功能。",
                    )

                    if result == "yes":
                        chrome_ready = True
                        self.log("✅ 抖音登录已确认\n\n")
                    else:
                        self.log("⚠️ 跳过抖音登录，自动发布功能不可用\n\n")
                except Exception as e:
                    self.log(f"⚠️ Chrome 启动失败: {e}\n\n")
            else:
                self.log("⚠️ 未找到 Chrome 浏览器，抖音自动发布不可用\n\n")
        else:
            self.log("⚠️ Playwright 未安装，抖音自动发布不可用\n\n")

        # ============================================================
        # Step 2: 启动 pipeline_gradio.py
        # ============================================================
        self.log(f"🚀 启动: {launch_script.name}\n")
        self.log(f"   端口: {server_port}\n\n")

        # 设置环境变量
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        # 传递 Chrome CDP 端口，供 publisher.py 复用
        if chrome_ready:
            env["CHROME_CDP_PORT"] = "9222"

        # 启动进程
        try:
            subprocess.Popen(
                [state.python_exe, str(launch_script)],
                env=env, cwd=str(ROOT_DIR),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            messagebox.showerror("启动失败", f"无法启动应用:\n{e}")
            return

        # 等待服务就绪
        self.log("⏳ 等待服务启动...")
        for i in range(15):
            time.sleep(2)
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{server_port}", timeout=1)
                self.log(" ✅\n")
                break
            except Exception:
                if i < 14:
                    self.log(".")
                    self.log_text.update()
        else:
            self.log(" ⚠️ 超时\n")

        # 打开浏览器
        url = f"http://127.0.0.1:{server_port}"
        self.log(f"🌐 打开浏览器: {url}\n")

        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            self.log("⚠️ 无法自动打开浏览器，请手动访问\n")

        # 启动成功弹窗
        status_msg = "🎉 旗博士追爆智能体已启动！"
        if chrome_ready:
            status_msg += "\n\n✅ 抖音自动发布功能已就绪"
        else:
            status_msg += "\n\n⚠️ 抖音自动发布功能未启用（需 Chrome + Playwright）"

        messagebox.showinfo("启动成功", status_msg + f"\n\n访问地址: {url}")

    def run(self):
        self.root.mainloop()


# ============================================================
# 入口
# ============================================================

def main():
    app = LauncherGUI()
    app.run()


if __name__ == "__main__":
    main()
