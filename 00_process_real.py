"""
00_process_real.py  (FIXED)
===========================
Converts 99 real "Zerotwo" M4A recordings to 16kHz mono WAV.
Uses ffmpeg (available via Anaconda) for reliable M4A decoding.

Input:  C:/Users/sujit/Downloads/syh/*.m4a  (99 recordings)
Output: dataset_raw/real_positive/  (WAV, 16kHz mono)
"""

import os
import sys
import subprocess
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_DIR    = Path(__file__).parent
# Your real recordings are in syh/ folder
RECORDINGS_DIR = Path(r"C:\Users\sujit\Downloads\syh")
OUTPUT_DIR     = PROJECT_DIR / "dataset_raw" / "real_positive"
SAMPLE_RATE    = 16000


def find_ffmpeg():
    """Find ffmpeg -- tries imageio-ffmpeg bundle first, then system PATH."""
    # 1. imageio-ffmpeg bundled binary (most reliable in venv)
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        exe = get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    # 2. System PATH
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    # 3. Common Anaconda locations
    candidates = [
        r"D:\Anaconda3\Library\bin\ffmpeg.exe",
        r"C:\Anaconda3\Library\bin\ffmpeg.exe",
        r"C:\ProgramData\anaconda3\Library\bin\ffmpeg.exe",
        r"C:\Users\sujit\anaconda3\Library\bin\ffmpeg.exe",
        r"C:\Users\sujit\Anaconda3\Library\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def convert_m4a(m4a_path: Path, out_path: Path, ffmpeg: str) -> bool:
    """Convert M4A to 16kHz mono WAV using ffmpeg."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",                   # overwrite
        "-i", str(m4a_path),    # input
        "-ar", str(SAMPLE_RATE),# resample to 16kHz
        "-ac", "1",             # mono
        "-af", "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,"
               "areverse,silenceremove=start_periods=1:start_silence=0.1:"
               "start_threshold=-50dB,areverse,"      # trim silence both ends
               "loudnorm=I=-16:TP=-3:LRA=11",         # normalize loudness
        "-acodec", "pcm_s16le", # 16-bit PCM
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000
    except Exception as e:
        return False


def fallback_librosa(m4a_path: Path, out_path: Path) -> bool:
    """Fallback: use librosa + soundfile if ffmpeg unavailable."""
    try:
        import soundfile as sf
        import librosa
        audio, sr = librosa.load(str(m4a_path), sr=SAMPLE_RATE, mono=True)
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        if len(audio_trimmed) < 0.2 * SAMPLE_RATE:
            return False
        peak = np.max(np.abs(audio_trimmed))
        if peak > 0:
            audio_trimmed = audio_trimmed / peak * 0.7
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio_trimmed.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
        return out_path.exists() and out_path.stat().st_size > 500
    except Exception as e:
        return False


def main():
    print("=" * 60)
    print("  REAL RECORDINGS CONVERTER")
    print(f"  Source: {RECORDINGS_DIR}")
    print("=" * 60)

    # Find M4A files
    m4a_files = sorted(RECORDINGS_DIR.glob("*.m4a"))
    if not m4a_files:
        m4a_files = sorted(RECORDINGS_DIR.glob("**/*.m4a"))
    print(f"\n  Found {len(m4a_files)} M4A recordings")

    if not m4a_files:
        print("  [X] No M4A files found!")
        sys.exit(1)

    # Find converter
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        print(f"  [OK] Using ffmpeg: {ffmpeg}")
        converter = "ffmpeg"
    else:
        print("  [!] ffmpeg not found -- using librosa fallback")
        converter = "librosa"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ok_count = fail_count = 0

    pbar = tqdm(m4a_files, desc="Converting M4A->WAV", unit="file")
    for m4a_path in pbar:
        # Clean filename
        stem = m4a_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
        out_path = OUTPUT_DIR / f"real_{stem}.wav"

        if out_path.exists() and out_path.stat().st_size > 1000:
            ok_count += 1
            pbar.set_postfix(ok=ok_count)
            continue

        if converter == "ffmpeg":
            ok = convert_m4a(m4a_path, out_path, ffmpeg)
        else:
            ok = fallback_librosa(m4a_path, out_path)

        if ok:
            ok_count += 1
        else:
            fail_count += 1
        pbar.set_postfix(ok=ok_count, fail=fail_count)

    print("\n" + "=" * 60)
    print(f"  [OK] Converted: {ok_count} recordings")
    print(f"  [X] Failed:    {fail_count}")
    print(f"  📁 Saved to: {OUTPUT_DIR}")
    print("  -> Will be augmented 10x in 05_augment.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
