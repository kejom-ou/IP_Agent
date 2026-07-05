@echo off
:: Force GBK code page for cmd.exe (avoids Chinese encoding issues)
chcp 936 >nul 2>&1
title IP Agent - One-Click Launcher (MuseTalk)
setlocal enabledelayedexpansion

:: ============================================================
::  IP Agent - Windows One-Click Launcher
::  Auto-detect environment, GPU, install deps, start Gradio Web UI
::
::  Mode:     MuseTalk lip-sync (default)
::  Env:      conda ip_agent_gpu
:: ============================================================

:: Script lives in local_models/, project root is one level up
cd /d "%~dp0"
set "THIS_DIR=%~dp0"
pushd "%~dp0.."
set "ROOT_DIR=%CD%"
popd

echo.
echo ============================================================
echo        IP Agent - One-Click Launcher (MuseTalk)
echo ============================================================
echo.

:: ============================================================
:: Step 1: Locate Python (conda ip_agent_gpu)
:: ============================================================
echo [1/6] Locating Python environment...

set "PYTHON_EXE="
set "CONDA_ENV=ip_agent_gpu"
set "USE_CONDA=0"

:: 1.1 Prefer bundled miniconda3
if exist "%ROOT_DIR%\miniconda3\envs\%CONDA_ENV%\python.exe" (
    set "PYTHON_EXE=%ROOT_DIR%\miniconda3\envs\%CONDA_ENV%\python.exe"
    set "CONDA_ACTIVATE=%ROOT_DIR%\miniconda3\Scripts\activate.bat"
    set "USE_CONDA=1"
    echo   [OK] Bundled Conda: miniconda3\envs\%CONDA_ENV%
    goto :python_found
)

:: 1.2 Detect system Conda
where conda >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('conda info --base 2^>nul') do set "CONDA_BASE=%%i"
    if exist "!CONDA_BASE!\envs\%CONDA_ENV%\python.exe" (
        set "PYTHON_EXE=!CONDA_BASE!\envs\%CONDA_ENV%\python.exe"
        set "CONDA_ACTIVATE=!CONDA_BASE!\Scripts\activate.bat"
        set "USE_CONDA=1"
        echo   [OK] Conda env: %CONDA_ENV%
        goto :python_found
    )
)

:: 1.3 Fallback to system Python
where python >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%i"
    if not "!PYTHON_EXE!"=="" (
        echo   [!] Using system Python (recommend conda ip_agent_gpu)
        goto :python_found
    )
)

:: 1.4 Common install paths fallback
for %%p in (
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
) do (
    if exist %%p (
        set "PYTHON_EXE=%%~p"
        echo   [!] Found Python: %%p (recommend conda env)
        goto :python_found
    )
)

echo   [FAIL] Python not found. Create conda env first:
echo     conda create -n ip_agent_gpu python=3.10
echo     conda activate ip_agent_gpu
echo     pip install -r local_models\requirements.txt
pause
exit /b 1

:python_found
"%PYTHON_EXE%" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [FAIL] Python failed to launch
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('"%PYTHON_EXE%" --version') do echo   Version: %%v

:: ============================================================
:: Step 2: Activate Conda + verify pip
:: ============================================================
echo.
echo [2/6] Configuring Python environment...

if "%USE_CONDA%"=="1" (
    call "%CONDA_ACTIVATE%" "%CONDA_ENV%" >nul 2>&1
    if %errorlevel% neq 0 (
        echo   [!] Conda activate failed, using Python directly
        set "USE_CONDA=0"
    ) else (
        echo   [OK] Conda env activated: %CONDA_ENV%
    )
)

"%PYTHON_EXE%" -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [FAIL] pip not available
    pause
    exit /b 1
)
echo   [OK] pip ready

:: ============================================================
:: Step 3: Detect GPU & PyTorch
:: ============================================================
echo.
echo [3/6] Detecting GPU & PyTorch...

set "GPU_NAME="
for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format=csv^,noheader 2^>nul') do (
    if "!GPU_NAME!"=="" set "GPU_NAME=%%g"
)

if "!GPU_NAME!"=="" (
    echo   [!] No NVIDIA GPU detected - will use CPU (very slow)
) else (
    echo   [OK] GPU: !GPU_NAME!
)

:: Check PyTorch CUDA availability
"%PYTHON_EXE%" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, VRAM {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')" 2>nul
if %errorlevel% equ 0 (
    echo   [OK] PyTorch CUDA ready
) else (
    echo   [!] PyTorch CUDA unavailable
    echo   Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
)

:: ============================================================
:: Step 4: FFmpeg
:: ============================================================
echo.
echo [4/6] Checking FFmpeg...

set "FFMPEG_OK=0"
set "FFMPEG_BIN=%ROOT_DIR%\ffmpeg\bin\ffmpeg.exe"

where ffmpeg >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] System FFmpeg available
    set "FFMPEG_OK=1"
    goto :ffmpeg_done
)

if exist "%FFMPEG_BIN%" (
    set "PATH=%ROOT_DIR%\ffmpeg\bin;%PATH%"
    echo   [OK] Bundled FFmpeg available
    set "FFMPEG_OK=1"
    goto :ffmpeg_done
)

echo   [!] FFmpeg not found - video composition will fail
echo   Install: winget install --id Gyan.FFmpeg -e

:ffmpeg_done

:: ============================================================
:: Step 5: Check core dependencies
:: ============================================================
echo.
echo [5/6] Checking core dependencies...

"%PYTHON_EXE%" -c "import torch; import gradio; import funasr; import modelscope; import soundfile; print('Core deps OK')" 2>nul
if %errorlevel% neq 0 (
    echo   [!] Core deps missing, installing...
    set "REQ_FILE=%ROOT_DIR%\local_models\requirements.txt"
    if exist "!REQ_FILE!" (
        "%PYTHON_EXE%" -m pip install --upgrade pip -q 2>nul
        "%PYTHON_EXE%" -m pip install -r "!REQ_FILE!"
        if !errorlevel! neq 0 (
            echo   [FAIL] Dep install failed - check network
            pause
            exit /b 1
        )
        echo   [OK] Dependencies installed
    ) else (
        echo   [FAIL] requirements.txt not found
        pause
        exit /b 1
    )
) else (
    echo   [OK] Core deps ready
)

:: ============================================================
:: Step 6: Set env vars & launch
:: ============================================================
echo.
echo [6/6] Launching IP Agent Web UI...

:: RTX 5060 Blackwell cuDNN compatibility - must be set before torch import
set "TORCH_CUDNN_V8_API_DISABLED=1"

:: Add project root to Python path
set "PYTHONPATH=%ROOT_DIR%;%PYTHONPATH%"

:: Launch script
set "LAUNCH_SCRIPT=%ROOT_DIR%\local_models\pipeline_gradio.py"
set "SERVER_PORT=7861"

if not exist "%LAUNCH_SCRIPT%" (
    echo [FAIL] Launch script not found: local_models\pipeline_gradio.py
    pause
    exit /b 1
)

echo   [OK] Launching: pipeline_gradio.py
echo   [OK] URL: http://127.0.0.1:%SERVER_PORT%
echo ============================================================
echo.
echo   Notes:
echo     - ASR/TTS/LLM auto load/unload to fit 8GB VRAM
echo     - MuseTalk lip-sync uses ~6GB VRAM, loaded last
echo     - Close this window to stop the service
echo     - Full CLI pipeline: python pipeline.py --url ^<url^> --image ^<path^>
echo.
echo ============================================================
echo.

start "" "%PYTHON_EXE%" "%LAUNCH_SCRIPT%"

:: Wait for service to be ready
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
    echo [OK] Service ready, opening browser...
    start http://127.0.0.1:%SERVER_PORT%
) else (
    echo [!] Service slow to start, open manually: http://127.0.0.1:%SERVER_PORT%
    start http://127.0.0.1:%SERVER_PORT%
)

echo.
echo ============================================================
echo   IP Agent Started
echo.
echo   URL:    http://127.0.0.1:%SERVER_PORT%
echo   Stop:   Close this window
echo ============================================================
echo.
pause >nul
