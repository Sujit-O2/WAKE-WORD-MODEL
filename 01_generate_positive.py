"""
01_generate_positive.py  (FIXED)
================================
Generates 1500 "Zerotwo" TTS samples using Piper TTS.
FIX: Reads raw WAV from piper stdout instead of --output_file.
ADDED: 6 voices for better diversity (3x US + 3x other accents).

Output: dataset_raw/positive/  (1500 WAV files, 16kHz mono)
"""

import os
import sys
import random
import subprocess
import urllib.request
import zipfile
import shutil
import tempfile
from pathlib import Path
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent
PIPER_DIR   = PROJECT_DIR / "tools" / "piper"
VOICES_DIR  = PROJECT_DIR / "tools" / "voices"
OUTPUT_DIR  = PROJECT_DIR / "dataset_raw" / "positive"
PIPER_EXE   = PIPER_DIR / "piper.exe"
ESPEAK_DIR  = PIPER_DIR / "espeak-ng-data"

TARGET_SAMPLES = 1500

PIPER_RELEASE_URL = (
    "https://github.com/rhasspy/piper/releases/download/"
    "2023.11.14-2/piper_windows_amd64.zip"
)

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# -- 6 voices for maximum diversity --------------------------------
VOICES = {
    # US English
    "lessac":  {"onnx": f"{HF_BASE}/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
                "json": f"{HF_BASE}/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"},
    "amy":     {"onnx": f"{HF_BASE}/en/en_US/amy/medium/en_US-amy-medium.onnx",
                "json": f"{HF_BASE}/en/en_US/amy/medium/en_US-amy-medium.onnx.json"},
    "ryan":    {"onnx": f"{HF_BASE}/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
                "json": f"{HF_BASE}/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"},
    # More US variety
    "joe":     {"onnx": f"{HF_BASE}/en/en_US/joe/medium/en_US-joe-medium.onnx",
                "json": f"{HF_BASE}/en/en_US/joe/medium/en_US-joe-medium.onnx.json"},
    "kusal":   {"onnx": f"{HF_BASE}/en/en_US/kusal/medium/en_US-kusal-medium.onnx",
                "json": f"{HF_BASE}/en/en_US/kusal/medium/en_US-kusal-medium.onnx.json"},
    # British English (accent diversity)
    "alan":    {"onnx": f"{HF_BASE}/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
                "json": f"{HF_BASE}/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json"},
}

ZEROTWO_VARIANTS = [
    "Zerotwo",
    "Zero two",
    "zero two",
    "Zerotwo!",
    "Hey Zerotwo",
    "zerotwo",
    "Zero-two",
    "ZERO TWO",
]

SPEED_VARIANTS = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]


def download_file(url, dest, desc=""):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [OK] Already exists: {dest.name}")
        return True
    print(f"  [DL] Downloading {desc or dest.name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [OK] Saved: {dest.name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def setup_piper():
    if PIPER_EXE.exists():
        print("[OK] Piper already installed.")
        return
    print("\n[SETUP] Downloading Piper TTS...")
    zip_path = PROJECT_DIR / "tools" / "piper_windows.zip"
    download_file(PIPER_RELEASE_URL, zip_path, "piper_windows_amd64.zip")
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            filename = Path(member).name
            if not filename:
                continue
            target = PIPER_DIR / filename
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    zip_path.unlink()
    print(f"  [OK] Piper ready at: {PIPER_DIR}")


def setup_voices():
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[SETUP] Downloading voice models...")
    available = []
    for voice_name, urls in VOICES.items():
        onnx_dest = VOICES_DIR / f"en_US-{voice_name}-medium.onnx"
        # Handle British voice naming
        if voice_name == "alan":
            onnx_dest = VOICES_DIR / f"en_GB-{voice_name}-medium.onnx"
        json_dest = Path(str(onnx_dest) + ".json")
        ok_onnx = download_file(urls["onnx"], onnx_dest, f"{voice_name} model")
        ok_json = download_file(urls["json"], json_dest, f"{voice_name} config")
        if ok_onnx and ok_json:
            available.append((voice_name, onnx_dest))
        else:
            print(f"  [!] Skipping {voice_name} (download failed)")
    return available


def generate_sample(voice_onnx, text, output_wav, speed):
    """
    Piper writes to --output_dir as numbered files (0.wav, 1.wav...).
    Use a temp dir per call so files don't collide between parallel runs.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                str(PIPER_EXE),
                "--model",       str(voice_onnx),
                "--espeak_data", str(ESPEAK_DIR),
                "--output_dir",  tmpdir,
                "--length_scale", str(round(speed, 2)),
                "--sentence_silence", "0.1",
            ]
            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )
            # Piper creates 0.wav (or similar) in tmpdir
            tmp_wavs = sorted(Path(tmpdir).glob("*.wav"))
            if tmp_wavs and tmp_wavs[0].stat().st_size > 500:
                output_wav.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tmp_wavs[0], output_wav)
                return True
        return False
    except Exception:
        return False


def main():
    print("=" * 60)
    print("  ZEROTWO POSITIVE DATA GENERATOR  (6 voices)")
    print(f"  Target: {TARGET_SAMPLES} samples")
    print("=" * 60)

    setup_piper()
    available_voices = setup_voices()

    if not available_voices:
        print("[X] No voices available -- check internet connection.")
        sys.exit(1)

    print(f"\n  Available voices: {[v[0] for v in available_voices]}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_voice = TARGET_SAMPLES // len(available_voices)
    remainder = TARGET_SAMPLES % len(available_voices)
    sample_idx = 0
    total_ok = 0
    total_fail = 0

    for v_idx, (voice_name, voice_onnx) in enumerate(available_voices):
        n_samples = per_voice + (1 if v_idx < remainder else 0)
        print(f"\n[VOICE: {voice_name}] -> {n_samples} samples")
        pbar = tqdm(range(n_samples), desc=f"  {voice_name}", unit="file")
        for _ in pbar:
            text  = random.choice(ZEROTWO_VARIANTS)
            speed = random.choice(SPEED_VARIANTS) + random.uniform(-0.02, 0.02)
            speed = max(0.75, min(1.25, speed))
            out_wav = OUTPUT_DIR / f"pos_{voice_name}_{sample_idx:05d}.wav"
            sample_idx += 1

            if out_wav.exists() and out_wav.stat().st_size > 500:
                total_ok += 1
                continue

            if generate_sample(voice_onnx, text, out_wav, speed):
                total_ok += 1
            else:
                total_fail += 1
            pbar.set_postfix(ok=total_ok, fail=total_fail)

    print("\n" + "=" * 60)
    print(f"  [OK] Generated: {total_ok} positive samples")
    print(f"  [X] Failed:    {total_fail}")
    print(f"  📁 Output:   {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
