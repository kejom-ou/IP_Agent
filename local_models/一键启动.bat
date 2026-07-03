@echo off
chcp 65001 >nul
title 旗博士追爆智能体 - 一键启动
setlocal enabledelayedexpansion

:: ============================================================
::  旗博士追爆智能体 — Windows 一键启动器
::  自动检测环境、安装依赖、启动服务
:: ============================================================

:: 脚本位于 local_models/ 目录，项目根为上级目录
cd /d "%~dp0"
set "THIS_DIR=%~dp0"
pushd "%~dp0.."
set "ROOT_DIR=%CD%"
popd

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║       旗博士追爆智能体 - 一键启动              ║
echo ╚══════════════════════════════════════════════════╝
echo.

:: ============================================================
:: 步骤 1：查找 Python
:: ============================================================
echo [1/6] 查找 Python 环境...

set "PYTHON_EXE="
set "CONDA_ENV=avatar"
set "USE_CONDA=0"

:: 1.1 优先检测项目内置 miniconda3
if exist "%ROOT_DIR%miniconda3\envs\%CONDA_ENV%\python.exe" (
    set "PYTHON_EXE=%ROOT_DIR%miniconda3\envs\%CONDA_ENV%\python.exe"
    set "CONDA_ACTIVATE=%ROOT_DIR%miniconda3\Scripts\activate.bat"
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
echo [2/6] 配置 Python 环境...

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
:: 尝试启动图形化启动器
:: ============================================================
echo.
echo [3/3] 启动图形化启动器...

set "GUI_LAUNCHER=%ROOT_DIR%local_models\launcher_gui.py"

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
set "REQ_FILE=%ROOT_DIR%local_models\requirements.txt"
"%PYTHON_EXE%" -c "import torch, gradio, funasr, modelscope" 2>nul
if %errorlevel% neq 0 (
    echo   📦 安装依赖（首次约 5-15 分钟）...
    "%PYTHON_EXE%" -m pip install --upgrade pip -q 2>nul
    "%PYTHON_EXE%" -m pip install torch>=2.1.0 -q 2>nul
    "%PYTHON_EXE%" -m pip install -r "%REQ_FILE%" -q 2>nul
)

:: 启动（唯一入口：local_models/pipeline_gradio.py）
set "LAUNCH_SCRIPT=%ROOT_DIR%local_models\pipeline_gradio.py"
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
echo ║  🎉 旗博士追爆智能体 已启动                    ║
echo ║                                                ║
echo ║  界面地址: http://127.0.0.1:7860               ║
echo ║  关闭此窗口将停止服务                          ║
echo ╚══════════════════════════════════════════════════╝
echo.
pause >nul
