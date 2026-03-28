@echo off
echo ================================================
echo   Zerotwo Wake Word System - Environment Setup
echo ================================================
echo.

set PROJ_DIR=%~dp0
set VENV_DIR=%PROJ_DIR%wake_env

:: Try py launcher with 3.10 first, then fallback
echo [1/6] Creating virtual environment with Python 3.10...
py -3.10 -m venv "%VENV_DIR%" 2>nul
if errorlevel 1 (
    echo Trying python3...
    python -m venv "%VENV_DIR%"
)
if errorlevel 1 (
    echo ERROR: Could not create venv. Make sure Python 3.10 is installed.
    pause
    exit /b 1
)

echo [2/6] Activating environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo [3/6] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [4/6] Installing core dependencies...
pip install "numpy==1.24.4" --quiet
pip install "protobuf==3.20.3" --quiet
pip install "onnxruntime==1.17.0" --quiet

echo [5/6] Installing audio + data dependencies...
pip install "librosa==0.10.1" --quiet
pip install soundfile --quiet
pip install audiomentations --quiet
pip install tqdm --quiet
pip install scipy --quiet
pip install requests --quiet
pip install pyyaml --quiet

echo [6/6] Installing PyTorch (CPU)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet

echo.
echo [OPTIONAL] Installing openWakeWord (may take a moment)...
pip install openwakeword --quiet

echo.
echo ================================================
echo   Setup COMPLETE!
echo   Activate with: wake_env\Scripts\activate
echo ================================================
pause
