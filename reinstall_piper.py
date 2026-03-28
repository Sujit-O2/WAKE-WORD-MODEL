"""
reinstall_piper.py
==================
Re-downloads and properly extracts piper maintaining the espeak-ng-data
directory structure (voices/, lang/ subdirs). Run this once.
"""
import urllib.request
import zipfile
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
TOOLS_DIR   = PROJECT_DIR / "tools"
PIPER_DIR   = TOOLS_DIR / "piper"
VOICES_DIR  = TOOLS_DIR / "voices"
PIPER_EXE   = PIPER_DIR / "piper.exe"
ESPEAK_DIR  = PIPER_DIR / "espeak-ng-data"

URL = ("https://github.com/rhasspy/piper/releases/download/"
       "2023.11.14-2/piper_windows_amd64.zip")

print("=" * 55)
print("  PIPER REINSTALL (proper directory structure)")
print("=" * 55)

# ── Backup voice models (they're fine) ───────────────
print("\n[1] Backing up voice models...")
voice_backup = TOOLS_DIR / "voices_backup"
if VOICES_DIR.exists() and not voice_backup.exists():
    shutil.copytree(VOICES_DIR, voice_backup)
    print(f"  Backed up to: voices_backup/")

# ── Remove old broken piper dir ───────────────────────
print("\n[2] Removing old piper installation...")
if PIPER_DIR.exists():
    shutil.rmtree(PIPER_DIR)
    print("  Removed tools/piper/")
PIPER_DIR.mkdir(parents=True, exist_ok=True)

# ── Download fresh zip ────────────────────────────────
zip_path = TOOLS_DIR / "piper_windows.zip"
print(f"\n[3] Downloading piper (~30MB)...")
print(f"  URL: {URL}")
urllib.request.urlretrieve(URL, zip_path)
print(f"  Downloaded: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

# ── Extract maintaining directory structure ────────────
print("\n[4] Extracting with proper directory structure...")
with zipfile.ZipFile(zip_path, "r") as zf:
    members = zf.infolist()
    print(f"  {len(members)} entries in zip")
    for member in members:
        parts = Path(member.filename).parts
        if len(parts) <= 1:
            continue  # skip root dir entry
        # Strip the leading 'piper/' component
        rel_path = Path(*parts[1:])
        target = PIPER_DIR / rel_path
        if member.filename.endswith("/"):  # directory entry
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

zip_path.unlink()
print("  Extraction complete!")

# ── Show structure ─────────────────────────────────────
espeak_items = list(ESPEAK_DIR.rglob("*")) if ESPEAK_DIR.exists() else []
print(f"\n  espeak-ng-data: {'EXISTS' if ESPEAK_DIR.exists() else 'MISSING'}")
print(f"  espeak-ng-data items: {len(espeak_items)}")
subdirs = [d.name for d in ESPEAK_DIR.iterdir() if d.is_dir()] if ESPEAK_DIR.exists() else []
print(f"  Subdirs: {subdirs}")

# ── Restore voice models ───────────────────────────────
print("\n[5] Restoring voice models...")
if voice_backup.exists():
    if VOICES_DIR.exists():
        shutil.rmtree(VOICES_DIR)
    shutil.copytree(voice_backup, VOICES_DIR)
    shutil.rmtree(voice_backup)
    print(f"  Voices restored: {len(list(VOICES_DIR.glob('*.onnx')))} models")

# ── Test piper ─────────────────────────────────────────
print("\n[6] Testing piper...")
voice_onnx = VOICES_DIR / "en_US-lessac-medium.onnx"
if not voice_onnx.exists():
    print(f"  ⚠  Voice not found: {voice_onnx.name}")
else:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [str(PIPER_EXE), "--model", str(voice_onnx),
             "--output_dir", tmpdir],
            input=b"Zerotwo",
            capture_output=True,
            timeout=30,
        )
        wavs = list(Path(tmpdir).glob("*.wav"))
        if wavs and wavs[0].stat().st_size > 500:
            shutil.copy2(wavs[0], PROJECT_DIR / "test_piper_ok.wav")
            print(f"  ✅ PIPER WORKS! {wavs[0].stat().st_size:,} bytes")
            print(f"     Saved: test_piper_ok.wav")
        else:
            print(f"  ✗ Still failing (rc={result.returncode})")
            print(f"  stderr: {result.stderr.decode(errors='replace')[:300]}")

print("\n" + "=" * 55)
print("  Done! Now run the generators.")
print("=" * 55)
