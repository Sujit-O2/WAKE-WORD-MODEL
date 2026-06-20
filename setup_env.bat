@echo off
echo ================================================
echo   Zerotwo Wake Word System - Environment Setup
echo ================================================
echo.

set PROJ_DIR=%~dp0
set VENV_DIR=%PROJ_DIR%wake_env

:: Step 1: Create virtual environment
echo [1/4] Creating virtual environment...
python -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo ERROR: Could not create venv. Make sure Python 3.10+ is installed.
    pause
    exit /b 1
)

:: Step 2: Activate
echo [2/4] Activating environment...
call "%VENV_DIR%\Scripts\activate.bat"

:: Step 3: Upgrade pip
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: Step 4: Install all dependencies from requirements.txt
echo [4/4] Installing dependencies from requirements.txt...
echo       (For GPU: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121)
pip install -r "%PROJ_DIR%requirements.txt" --quiet

echo.
echo ================================================
echo   Setup COMPLETE!
echo.
echo   Activate with:  wake_env\Scripts\activate
echo   Or run directly:
echo     wake_env\Scripts\python.exe 01_generate_positive.py
echo     wake_env\Scripts\python.exe 06_train.py
echo ================================================
pause
