@echo off
chcp 936 >nul 2>&1
title 打包 口播智能体.exe
setlocal enabledelayedexpansion

:: ============================================================
::  口播智能体 - PyInstaller 打包脚本
::  输出 dist/口播智能体/口播智能体.exe
:: ============================================================

:: Navigate to script directory, project root is one level up
cd /d "%~dp0"
pushd "%~dp0.."
set "ROOT_DIR=%CD%"
popd
cd /d "%ROOT_DIR%"

echo.
echo ============================================================
echo       打包 口播智能体.exe (PyInstaller ^+ onedir)
echo ============================================================
echo.

:: ============================================================
:: Step 1: Locate Python (conda ip_agent_gpu)
:: ============================================================
echo [1/4] Locating Python environment...

set "PYTHON_EXE="
set "CONDA_ENV=ip_agent_gpu"

if exist "%ROOT_DIR%\miniconda3\envs\%CONDA_ENV%\python.exe" (
    set "PYTHON_EXE=%ROOT_DIR%\miniconda3\envs\%CONDA_ENV%\python.exe"
    set "CONDA_ACTIVATE=%ROOT_DIR%\miniconda3\Scripts\activate.bat"
    echo   [OK] Bundled Conda: miniconda3\envs\%CONDA_ENV%
    goto :python_found
)

where conda >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('conda info --base 2^>nul') do set "CONDA_BASE=%%i"
    if exist "!CONDA_BASE!\envs\%CONDA_ENV%\python.exe" (
        set "PYTHON_EXE=!CONDA_BASE!\envs\%CONDA_ENV%\python.exe"
        set "CONDA_ACTIVATE=!CONDA_BASE!\Scripts\activate.bat"
        echo   [OK] Conda env: %CONDA_ENV%
        goto :python_found
    )
)

echo   [FAIL] Python environment 'ip_agent_gpu' not found
echo   请先配置 conda 环境后再打包
pause
exit /b 1

:python_found
echo   Python: %PYTHON_EXE%

:: ============================================================
:: Step 2: Install PyInstaller
:: ============================================================
echo.
echo [2/4] Checking PyInstaller...

"%PYTHON_EXE%" -m pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo   [!] Installing PyInstaller...
    "%PYTHON_EXE%" -m pip install pyinstaller
    if !errorlevel! neq 0 (
        echo   [FAIL] PyInstaller install failed
        pause
        exit /b 1
    )
)
echo   [OK] PyInstaller ready

:: ============================================================
:: Step 3: Activate conda env (if applicable)
:: ============================================================
echo.
echo [3/4] Configuring environment...

if defined CONDA_ACTIVATE (
    call "%CONDA_ACTIVATE%" "%CONDA_ENV%" >nul 2>&1
)

:: ============================================================
:: Step 4: Run PyInstaller
:: ============================================================
echo.
echo [4/4] Building .exe (this will take 5-15 minutes)...
echo   Mode: onedir (folder with exe + dependencies)
echo   Output: dist\口播智能体\
echo ============================================================
echo.

:: 清理旧的 build
if exist "build" (
    echo   Cleaning old build cache...
    rmdir /s /q "build"
)

:: 清理旧的 dist
if exist "dist\口播智能体" (
    echo   Cleaning old dist...
    rmdir /s /q "dist\口播智能体"
)

:: 执行打包
pushd "%ROOT_DIR%"
"%PYTHON_EXE%" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    "local_models\desktop_app.spec"
set "BUILD_RESULT=%errorlevel%"
popd

if %BUILD_RESULT% neq 0 (
    echo.
    echo ============================================================
    echo   [FAIL] 打包失败! 错误码: %BUILD_RESULT%
    echo   常见问题:
    echo     - 模块缺失: 检查 hiddenimports
    echo     - 文件未找到: 检查 spec 中 datas 路径
    echo     - 内存不足: 关闭其他程序后重试
    echo ============================================================
    pause
    exit /b %BUILD_RESULT%
)

:: ============================================================
:: Step 5: Done
:: ============================================================
echo.
echo ============================================================
echo   [OK] 打包完成!
echo.
echo   输出位置: %ROOT_DIR%\dist\口播智能体\
echo   启动文件: %ROOT_DIR%\dist\口播智能体\口播智能体.exe
echo.
echo   预计体积: 3-6 GB (含 PyTorch + CUDA 依赖)
echo   发布方式: 打包整个 "口播智能体" 文件夹为 zip
echo   用户使用: 解压后双击 口播智能体.exe 即可
echo.
echo   [!] 注意: 该 exe 仅能在相同 CUDA 版本的机器上运行
echo   [!] 首次启动可能需要安装 vc_redist (Visual C++ Redist)
echo ============================================================
echo.
pause
exit /b 0
