"""
============================================================
Windows EXE 打包脚本
============================================================
将整个项目打包为单个 .exe 文件，方便分发。

使用方法:
    # 安装打包工具
    pip install pyinstaller

    # 打包（生成 dist/旗博士追爆智能体.exe）
    python local_models/build_exe.py

注意:
    - 模型文件（pretrained_models/）不会被内嵌，需单独分发
    - FFmpeg/ImageMagick 需用户自行安装或在同一目录
    - .exe 文件约 500MB-2GB（PyTorch + Gradio）
============================================================
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# 项目根目录（脚本在 local_models/ 下）
ROOT_DIR = Path(__file__).parent.parent.absolute()

# ---- 配置 ----
APP_NAME = "旗博士追爆智能体"
MAIN_SCRIPT = "local_models/pipeline_gradio.py"  # 唯一启动入口
ICON_FILE = None                                    # 暂未提供 .ico 文件

# 需要打包的数据目录
INCLUDE_DIRS = [
    ("local_models", "local_models"),
    ("utils", "utils"),                       # 如有
    ("ai_processing", "ai_processing"),        # 如有
]

# 文件夹（只创建，内容由用户自行放入）
COPY_AFTER_BUILD = [
    "pretrained_models",
    "ffmpeg",
]


def check_pyinstaller():
    """检查 PyInstaller 是否已安装"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--version"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"✅ PyInstaller {result.stdout.strip()}")
            return True
    except Exception:
        pass
    print("❌ PyInstaller 未安装")
    print("   请运行: pip install pyinstaller")
    return False


def build_exe():
    """执行 PyInstaller 打包"""
    main_script_path = ROOT_DIR / MAIN_SCRIPT
    if not main_script_path.exists():
        print(f"❌ 未找到启动脚本: {MAIN_SCRIPT}")
        sys.exit(1)

    build_dir = ROOT_DIR / "pyinstaller_build"
    dist_dir = ROOT_DIR / "dist"

    # 清理旧构建
    for d in [build_dir, dist_dir]:
        if d.exists():
            shutil.rmtree(d)

    # 构建 PyInstaller 命令
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--onefile",                # 单文件模式
        "--console",                 # 显示控制台窗口（调试用）
        f"--distpath={dist_dir}",
        f"--workpath={build_dir}",
        f"--specpath={build_dir}",
        "--noconfirm",
        "--clean",
    ]

    # 图标
    if ICON_FILE and Path(ICON_FILE).exists():
        cmd += ["--icon", ICON_FILE]

    # 添加数据目录
    for src_dir, dst_dir in INCLUDE_DIRS:
        full_src = ROOT_DIR / src_dir
        if full_src.exists():
            # Windows 路径分隔符用 ;
            separator = ";" if sys.platform == "win32" else ":"
            cmd += ["--add-data", f"{full_src}{separator}{dst_dir}"]

    # 隐藏导入（确保关键包被打包）
    hidden_imports = [
        "gradio", "gradio.themes",
        "torch", "torchvision", "torchaudio",
        "transformers",
        "funasr",
        "modelscope",
        "numpy", "cv2", "soundfile",
        "playwright",
        "fastapi", "uvicorn",
        "requests", "psutil",
        "yt_dlp",
    ]
    for mod in hidden_imports:
        cmd += ["--hidden-import", mod]

    # 排除不需要的大包以减小体积
    excludes = [
        "tkinter", "unittest", "test", "pydoc",
        "IPython", "jupyter", "notebook",
        "matplotlib", "pandas", "scipy",
        "sqlalchemy", "h5py", "tensorflow",
        "torchvision", "torchaudio",  # 视频/音频推理不需要
    ]
    for mod in excludes:
        cmd += ["--exclude-module", mod]

    # 目标脚本
    cmd.append(str(main_script_path))

    print("=" * 60)
    print("📦 PyInstaller 打包")
    print(f"   主脚本: {MAIN_SCRIPT}")
    print(f"   输出: {dist_dir / APP_NAME}.exe")
    print("   预计: 500MB - 2GB（取决于依赖）")
    print("=" * 60)
    print()

    print("🔨 开始构建...")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))

    if result.returncode == 0:
        exe_path = dist_dir / f"{APP_NAME}.exe"
        print(f"\n🎉 打包成功！")
        print(f"   输出文件: {exe_path}")

        # 复制辅助目录
        for dir_name in COPY_AFTER_BUILD:
            src = ROOT_DIR / dir_name
            if src.exists():
                dst = dist_dir / dir_name
                if not dst.exists():
                    print(f"   📁 复制 {dir_name}/ 到发布目录...")
                    shutil.copytree(src, dst)

        print(f"\n📋 发布清单（放入同一文件夹即可）：")
        print(f"   1. {APP_NAME}.exe")
        for dir_name in COPY_AFTER_BUILD:
            if (ROOT_DIR / dir_name).exists():
                print(f"   2. {dir_name}/ 目录")
        print(f"\n💡 运行方式：双击 {APP_NAME}.exe")
    else:
        print("\n❌ 打包失败！请检查上方错误信息")
        sys.exit(1)


def main():
    os.chdir(ROOT_DIR)

    if not check_pyinstaller():
        sys.exit(1)

    build_exe()


if __name__ == "__main__":
    main()
