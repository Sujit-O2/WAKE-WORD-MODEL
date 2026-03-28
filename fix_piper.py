"""
fix_piper.py — Run this ONCE to fix piper's espeak-ng-data structure
and verify piper actually works before the big generation run.
"""
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
PIPER_DIR   = PROJECT_DIR / "tools" / "piper"
VOICES_DIR  = PROJECT_DIR / "tools" / "voices"
ESPEAK_DIR  = PIPER_DIR / "espeak-ng-data"
PIPER_EXE   = PIPER_DIR / "piper.exe"

print("=" * 55)
print("  PIPER FIX — Rebuilding espeak-ng-data structure")
print("=" * 55)

# ── Step 1: Create espeak-ng-data/ and move language files ────────
ESPEAK_DIR.mkdir(parents=True, exist_ok=True)
moved = 0
skipped_extensions = {".exe", ".dll", ".json", ".onnx", ".zip", ".txt", ".bat", ".py"}

for f in list(PIPER_DIR.iterdir()):
    if f.is_file() and f.suffix.lower() not in skipped_extensions:
        dest = ESPEAK_DIR / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
            moved += 1
    elif f.is_dir() and f.name not in ("espeak-ng-data", "voices"):
        # Move subdirectories too (some versions have subdirs in espeak data)
        dest = ESPEAK_DIR / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
            moved += 1

print(f"  ✓ Moved {moved} espeak data files → espeak-ng-data/")
print(f"  📁 espeak-ng-data has {sum(1 for _ in ESPEAK_DIR.rglob('*'))} items")

# ── Step 2: Test piper with espeak_data flag ───────────────────────
print("\n  Testing piper...")
voice_onnx = VOICES_DIR / "en_US-lessac-medium.onnx"

if not voice_onnx.exists():
    print(f"  ✗ Voice model not found: {voice_onnx}")
    sys.exit(1)

cmd = [
    str(PIPER_EXE),
    "--model",       str(voice_onnx),
    "--espeak_data", str(ESPEAK_DIR),
    "--sentence_silence", "0.1",
]

result = subprocess.run(
    cmd,
    input=b"Zerotwo",
    capture_output=True,
    timeout=30,
)

if result.returncode == 0 and len(result.stdout) > 500:
    test_wav = PROJECT_DIR / "test_piper_output.wav"
    test_wav.write_bytes(result.stdout)
    print(f"  ✓ PIPER WORKS! Output: {len(result.stdout):,} bytes")
    print(f"  ✓ Saved test audio: {test_wav.name}")
else:
    stderr = result.stderr.decode("utf-8", errors="replace")
    print(f"  ✗ Piper still failing (rc={result.returncode})")
    print(f"  stderr: {stderr[:400]}")
    sys.exit(1)

print("\n  ✅ Piper is ready. Now run the generators!")
print("=" * 55)
