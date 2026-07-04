@echo off
chcp 65001 >nul
title IP智能体 - 一键启动
setlocal enabledelayedexpansion

:: ============================================================
::  IP智能体 — Windows 一键启动器
::  自动检测环境、GPU、安装依赖、启动服务
:: ============================================================

:: 脚本位于 local_models/ 目录，项目根为上级目录
cd /d "%~dp0"
set "THIS_DIR=%~dp0"
pushd "%~dp0.."
set "ROOT_DIR=%CD%"
popd

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║       IP智能体 - 一键启动              ║
echo ╚══════════════════════════════════════════════════╝
echo.

:: ============================================================
:: 步骤 1：查找 Python
:: ============================================================
echo [1/8] 查找 Python 环境...

set "PYTHON_EXE="
set "CONDA_ENV=avatar"
set "USE_CONDA=0"

:: 1.1 优先检测项目内置 miniconda3
if exist "%ROOT_DIR%\miniconda3\envs\%CONDA_ENV%\python.exe" (
    set "PYTHON_EXE=%ROOT_DIR%\miniconda3\envs\%CONDA_ENV%\python.exe"
    set "CONDA_ACTIVATE=%ROOT_DIR%\miniconda3\Scripts\activate.bat"
    set "USE_CONDA=1"
    echo   ✅ 找到内置 Conda 环境: miniconda3\envs\%CONDA_ENV%
    goto :python_found
)

:: 1.2 检测系统 Conda
where conda >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('conda info --base 2^>nul') do set "CONDA_BASE=%%i"
    if exist "!CONDA_BASE!\envs\%CONDA_ENV%\python.exe" (
        set "PYTHON_EXE=!CONDA_BASE!\envs\%CONDA_ENV%\python.exe"
        set "CONDA_ACTIVATE=!CONDA_BASE!\Scripts\activate.bat"
        set "USE_CONDA=1"
        echo   ✅ 找到 Conda 环境: %CONDA_ENV%
        goto :python_found
    )
)

:: 1.3 检测系统 Python
where python >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%i"
    if not "!PYTHON_EXE!"=="" (
        echo   ✅ 找到系统 Python: !PYTHON_EXE!
        goto :python_found
    )
)

:: 1.4 常见安装路径
for %%p in (
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe"
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe"
) do (
    if exist %%p (
        set "PYTHON_EXE=%%~p"
        echo   ✅ 找到 Python: %%p
        goto :python_found
    )
)

echo   ❌ 未找到 Python！请安装 Python 3.10+ 或 miniconda
echo   下载地址: https://www.python.org/downloads/
pause
exit /b 1

:python_found
"%PYTHON_EXE%" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ❌ Python 启动失败，请检查安装
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('"%PYTHON_EXE%" --version') do echo   版本: %%v

:: ============================================================
:: 步骤 2：激活环境 & 检查 pip
:: ============================================================
echo.
echo [2/8] 配置 Python 环境...

if "%USE_CONDA%"=="1" (
    call "%CONDA_ACTIVATE%" "%CONDA_ENV%" >nul 2>&1
    if %errorlevel% neq 0 (
        echo   ⚠️ Conda 环境激活失败，尝试直接使用 Python
        set "USE_CONDA=0"
    ) else (
        echo   ✅ Conda 环境已激活
    )
)

:: 检查 pip
"%PYTHON_EXE%" -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ❌ pip 未安装，正在安装...
    "%PYTHON_EXE%" -m ensurepip --upgrade >nul 2>&1
    if %errorlevel% neq 0 (
        echo   ❌ pip 安装失败，请手动安装 pip
        pause
        exit /b 1
    )
)
echo   ✅ pip 可用

:: ============================================================
:: 步骤 3：检测显卡 & 智能安装/验证 PyTorch
:: ============================================================
echo.
echo [3/8] 检测显卡 & PyTorch 适配...

set "GPU_TYPE=none"
set "GPU_NAME="
set "TORCH_CUDA="
set "TORCH_URL="

:: 3.1 通过 nvidia-smi 获取显卡型号
for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format=csv^,noheader 2^>nul') do (
    if "!GPU_NAME!"=="" set "GPU_NAME=%%g"
)

if not "!GPU_NAME!"=="" (
    echo   🎮 检测到: !GPU_NAME!
    set "GPU_TYPE=nvidia"

    :: 判断是否为 RTX 50 系列 (Blackwell, sm_120) — 需要 PyTorch cu130
    echo !GPU_NAME! | findstr /i "RTX.*50[0-9][0-9]" >nul
    if !errorlevel! equ 0 (
        set "TORCH_CUDA=cu130"
        set "TORCH_URL=https://download.pytorch.org/whl/cu130"
        echo   🔧 RTX 50 系 ^(Blackwell/sm_120^)，需要 PyTorch + CUDA 13.0
        goto :torch_check
    )

    :: 其他 NVIDIA 显卡 (RTX 20/30/40, GTX 等) — cu124 即可
    set "TORCH_CUDA=cu124"
    set "TORCH_URL=https://download.pytorch.org/whl/cu124"
    echo   🔧 通用 NVIDIA 显卡，使用 PyTorch + CUDA 12.4
    goto :torch_check
)

:: 3.2 检测 AMD / Intel 显卡（wmic）
for /f "tokens=*" %%g in ('wmic path Win32_VideoController get name 2^>nul ^| findstr /i "AMD Radeon RX Arc"') do (
    if "!GPU_NAME!"=="" set "GPU_NAME=%%g"
)
if not "!GPU_NAME!"=="" (
    echo   ⚠️ 检测到: !GPU_NAME!
    echo   ⚠️ AMD/Intel 显卡不支持 pip 版 GPU 加速，将使用 CPU 模式
    set "GPU_TYPE=other"
    set "TORCH_CUDA=cpu"
    goto :torch_check
)

echo   ⚠️ 未检测到独立显卡，使用 CPU 模式
set "GPU_TYPE=none"
set "TORCH_CUDA=cpu"

:torch_check
:: 3.3 检查当前 PyTorch 是否已匹配
set "REINSTALL_TORCH=0"

if "%TORCH_CUDA%"=="cu130" (
    :: 需要 cu130: 检查 CUDA >= 13.0
    "%PYTHON_EXE%" -c "import torch; v=torch.version.cuda if torch.cuda.is_available() else ''; exit(0 if v and int(v.replace('.',''))>=130 else 1)" 2>nul
    if !errorlevel! neq 0 set "REINSTALL_TORCH=1"
)

if "%TORCH_CUDA%"=="cu124" (
    :: 需要 cu124: 检查 CUDA >= 12.0 且 < 13.0
    "%PYTHON_EXE%" -c "import torch; v=torch.version.cuda if torch.cuda.is_available() else ''; exit(0 if v and 120<=int(v.replace('.',''))<130 else 1)" 2>nul
    if !errorlevel! neq 0 set "REINSTALL_TORCH=1"
)

if "%TORCH_CUDA%"=="cpu" (
    :: CPU 模式: 有 torch 即可
    "%PYTHON_EXE%" -c "import torch" 2>nul
    if !errorlevel! neq 0 set "REINSTALL_TORCH=1"
)

:: 3.4 安装/更新 PyTorch
if %REINSTALL_TORCH% equ 1 (
    echo   📦 卸载旧版 PyTorch...
    "%PYTHON_EXE%" -m pip uninstall torch torchvision torchaudio -y -q 2>nul

    if "%TORCH_CUDA%"=="cu130" (
        echo   📥 安装 PyTorch CUDA 13.0 ^(适配 RTX 50，约 2GB^)...
        "%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url %TORCH_URL% 2>&1
        if !errorlevel! neq 0 (
            echo   ⚠️ cu130 失败，降级尝试 cu124...
            "%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 2>&1
        )
    ) else if "%TORCH_CUDA%"=="cu124" (
        echo   📥 安装 PyTorch CUDA 12.4 ^(约 2.5GB^)...
        "%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url %TORCH_URL% 2>&1
    ) else (
        echo   📥 安装 PyTorch CPU 版...
        "%PYTHON_EXE%" -m pip install torch torchvision torchaudio 2>&1
    )
    echo   ✅ PyTorch 安装完成
) else (
    "%PYTHON_EXE%" -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" 2>nul
    echo   ✅ PyTorch 已就绪
)

:: ============================================================
:: 步骤 4：检查并安装 FFmpeg
:: ============================================================
echo.
echo [4/8] 检查 FFmpeg...

set "FFMPEG_BIN=%ROOT_DIR%\ffmpeg\bin\ffmpeg.exe"
set "FFMPEG_OK=0"

:: 4.1 检查系统 PATH 中是否有 ffmpeg
where ffmpeg >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✅ 系统 FFmpeg 已安装
    set "FFMPEG_OK=1"
    goto :ffmpeg_done
)

:: 4.2 检查项目内置 ffmpeg
if exist "%FFMPEG_BIN%" (
    echo   ✅ 项目内置 FFmpeg 已存在
    set "FFMPEG_OK=1"
    goto :ffmpeg_done
)

:: 4.3 尝试 winget 安装（Win10/Win11 自带）
where winget >nul 2>&1
if %errorlevel% equ 0 (
    echo   📦 通过 winget 安装 FFmpeg...
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements >nul 2>&1
    if %errorlevel% equ 0 (
        echo   ✅ winget 安装 FFmpeg 成功
        where ffmpeg >nul 2>&1 && set "FFMPEG_OK=1" && goto :ffmpeg_done
    )
)

:: 4.4 下载便携版到项目目录（约 50MB）
echo   📥 下载 FFmpeg 便携版（约 50MB），请稍候...
set "FFMPEG_ZIP=%ROOT_DIR%\ffmpeg-temp.zip"
set "FFMPEG_EXTRACT_DIR=%ROOT_DIR%\ffmpeg-temp"
set "FFMPEG_TARGET=%ROOT_DIR%\ffmpeg"

"%PYTHON_EXE%" -c "import urllib.request, zipfile, os, shutil; url='https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'; zip_path=r'%FFMPEG_ZIP%'; extract_dir=r'%FFMPEG_EXTRACT_DIR%'; target=r'%FFMPEG_TARGET%'; os.makedirs(extract_dir, exist_ok=True); print('下载中...'); urllib.request.urlretrieve(url, zip_path); print('解压中...'); zf=zipfile.ZipFile(zip_path,'r'); zf.extractall(extract_dir); zf.close(); inner=[d for d in os.listdir(extract_dir) if d.startswith('ffmpeg')][0]; src=os.path.join(extract_dir,inner); shutil.move(src,target); shutil.rmtree(extract_dir,ignore_errors=True); os.remove(zip_path); print('done')" 2>nul

if exist "%FFMPEG_BIN%" (
    echo   ✅ FFmpeg 便携版安装成功
    set "FFMPEG_OK=1"
) else (
    echo   ⚠️ FFmpeg 自动安装失败，视频合成功能可能不可用
)

:ffmpeg_done
:: 把 FFmpeg 加入 PATH
if %FFMPEG_OK% equ 1 (
    if exist "%FFMPEG_BIN%" set "PATH=%ROOT_DIR%\ffmpeg\bin;%PATH%"
)

:: ============================================================
:: 步骤 5：尝试启动图形化启动器
:: ============================================================
echo.
echo [5/8] 启动图形化启动器...

set "GUI_LAUNCHER=%ROOT_DIR%\local_models\launcher_gui.py"

if exist "%GUI_LAUNCHER%" (
    "%PYTHON_EXE%" -c "import tkinter" 2>nul
    if !errorlevel! equ 0 (
        echo 🖥️  正在启动图形界面...
        start "" "%PYTHON_EXE%" "%GUI_LAUNCHER%"
        exit /b 0
    ) else (
        echo   ⚠️ tkinter 不可用，使用命令行模式
    )
) else (
    echo   ⚠️ GUI 启动器不存在，使用命令行模式
)

:: ============================================================
:: 兜底: 命令行模式（tkinter 不可用时）
:: ============================================================
echo.
echo [命令行模式] 检查并启动...
echo ══════════════════════════════════════════════════

:: 安装依赖
set "REQ_FILE=%ROOT_DIR%\local_models\requirements.txt"
"%PYTHON_EXE%" -c "import torch, gradio, funasr, modelscope" 2>nul
if %errorlevel% neq 0 (
    echo   📦 安装依赖（首次约 5-15 分钟）...
    "%PYTHON_EXE%" -m pip install --upgrade pip -q 2>nul
    "%PYTHON_EXE%" -m pip install -r "%REQ_FILE%" -q 2>nul
)

:: 启动（唯一入口：local_models/pipeline_gradio.py）
set "LAUNCH_SCRIPT=%ROOT_DIR%\local_models\pipeline_gradio.py"
set "SERVER_PORT=7860"

if not exist "%LAUNCH_SCRIPT%" (
    echo ❌ 未找到启动脚本: local_models\pipeline_gradio.py
    pause
    exit /b 1
)

echo 🚀 启动: %LAUNCH_SCRIPT%
echo    地址: http://127.0.0.1:%SERVER_PORT%
echo ══════════════════════════════════════════════════

set "PYTHONPATH=%ROOT_DIR%;%PYTHONPATH%"
start "" "%PYTHON_EXE%" "%LAUNCH_SCRIPT%"

:: 等待服务就绪
set "READY=0"
for /l %%i in (1,1,15) do (
    timeout /t 2 /nobreak >nul
    "%PYTHON_EXE%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:%SERVER_PORT%', timeout=1)" 2>nul
    if !errorlevel! equ 0 (
        set "READY=1"
        goto :service_ready
    )
)

:service_ready
if %READY% equ 1 (
    echo ✅ 服务就绪，打开浏览器...
    start http://127.0.0.1:%SERVER_PORT%
) else (
    echo ⚠️ 请手动访问: http://127.0.0.1:%SERVER_PORT%
    start http://127.0.0.1:%SERVER_PORT%
)

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║  🎉 IP智能体 已启动                    ║
echo ║                                                ║
echo ║  界面地址: http://127.0.0.1:7860               ║
echo ║  关闭此窗口将停止服务                          ║
echo ╚══════════════════════════════════════════════════╝
echo.
pause >nul
