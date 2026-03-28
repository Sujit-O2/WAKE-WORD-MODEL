@echo off
echo ================================================
echo   ZEROTWO - FULL DATA GENERATION PIPELINE
echo ================================================
echo.

set PROJ=c:\Users\sujit\Downloads\syh\test\zerotwo_wake
set PY=%PROJ%\wake_env\Scripts\python.exe

echo [Step 1] Installing ffmpeg bundle for M4A conversion...
%PY% -m pip install imageio-ffmpeg --quiet
echo   Done.

echo.
echo [Step 2] Fixing piper espeak-ng-data structure...
%PY% "%PROJ%\fix_piper.py"
if errorlevel 1 (
    echo.
    echo !! Piper fix FAILED - check errors above !!
    pause
    exit /b 1
)

echo.
echo [Step 3] Launching 3 generators in parallel...
echo.

:: Window 1 — Positive TTS (6 voices, 1500 samples)
start "1-POSITIVE TTS [1500 samples, 6 voices]" cmd /k "cd /d %PROJ% && %PY% 01_generate_positive.py && echo DONE - press any key && pause"

:: Window 2 — Real M4A recordings (99 files -> WAV)
start "2-REAL M4A CONVERT [99 recordings]" cmd /k "cd /d %PROJ% && %PY% 00_process_real.py && echo DONE - press any key && pause"

:: Window 3 — Negative speech (1500 samples)
start "3-NEGATIVE SPEECH [1500 samples]" cmd /k "cd /d %PROJ% && %PY% 03_generate_negative.py && echo DONE - press any key && pause"

echo.
echo ================================================
echo   All 3 windows launched and running!
echo.  
echo   Wait for ALL 3 to show DONE, then run:
echo     %PY% 04_clean_and_organize.py
echo     %PY% 05_augment.py
echo     %PY% 06_train.py
echo ================================================
pause
