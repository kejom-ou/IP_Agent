@echo off
::: Force GBK code page
chcp 936 >nul 2>&1
title 口播智能体 - 桌面版
setlocal enabledelayedexpansion

::: ============================================================
:::  口播智能体 - Windows 桌面版启动器
:::  使用 PySide6 原生窗口，无需浏览器
::: ============================================================

::: Navigate to script directory, project root is one level up
cd /d "%~dp0"
pushd "%~dp0.."
set "ROOT_DIR=%CD%"
popd

echo.
echo ============================================================
echo       口播智能体 - Windows 桌面版 (PySide6)
echo ============================================================
echo.

::: ============================================================
::: Step 1: Locate Python (conda ip_agent_gpu)
::: ============================================================
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

where python >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%i"
    if not "!PYTHON_EXE!"=="" (
        echo   [!] Using system Python (recommend conda ip_agent_gpu)
        goto :python_found
    )
)

echo   [FAIL] Python not found
pause
exit /b 1

:python_found
echo   Version: 
"%PYTHON_EXE%" --version

::: ============================================================
::: Step 2: Check PySide6
::: ============================================================
echo.
echo [2/4] Checking PySide6...

"%PYTHON_EXE%" -c "import PySide6; print('PySide6 OK')" 2>nul
if %errorlevel% neq 0 (
    echo   [!] PySide6 not installed, installing...
    "%PYTHON_EXE%" -m pip install PySide6
    if !errorlevel! neq 0 (
        echo   [FAIL] PySide6 install failed
        echo   Please install manually: pip install PySide6
        pause
        exit /b 1
    )
    echo   [OK] PySide6 installed
) else (
    echo   [OK] PySide6 ready
)

::: ============================================================
::: Step 3: Environment
::: ============================================================
echo.
echo [3/4] Configuring environment...

set "TORCH_CUDNN_V8_API_DISABLED=1"
set "PYTHONPATH=%ROOT_DIR%;%PYTHONPATH%"

if defined CONDA_ACTIVATE (
    call "%CONDA_ACTIVATE%" "%CONDA_ENV%" >nul 2>&1
)

set "LAUNCH_SCRIPT=%ROOT_DIR%\local_models\desktop_app.py"
if not exist "%LAUNCH_SCRIPT%" (
    echo [FAIL] desktop_app.py not found
    pause
    exit /b 1
)

::: ============================================================
::: Step 4: Launch desktop app
::: ============================================================
echo.
echo [4/4] Launching desktop application...
echo ============================================================
echo.
echo   Notes:
echo     - Native Windows window (no browser needed)
echo     - MuseTalk lip-sync: ~6GB VRAM, loaded on demand
echo     - Close the window to exit
echo.
echo ============================================================
echo.

start "" "%PYTHON_EXE%" "%LAUNCH_SCRIPT%"

echo Desktop app started in a new window.
echo You can close this console window.
echo.
timeout /t 3 >nul
exit /b 0
