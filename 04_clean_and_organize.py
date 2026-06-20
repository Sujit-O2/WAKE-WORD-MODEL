"""
04_clean_and_organize.py
========================
Cleans, validates, and organizes all generated audio into the final
training dataset structure.

Input dirs:
  dataset_raw/positive/       <- TTS generated
  dataset_raw/real_positive/  <- Your real recordings
  dataset_raw/negative/       <- Negative speech
  dataset_raw/background/     <- Noise files

Output dirs:
  dataset/positive/   <- All positive samples
  dataset/negative/   <- All negative speech
  dataset/background/ <- All background noise
"""

import sys
import shutil
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

# -----------------------------------------------------------------
PROJECT_DIR  = Path(__file__).parent
RAW_DIR      = PROJECT_DIR / "dataset_raw"
DATASET_DIR  = PROJECT_DIR / "dataset"
SAMPLE_RATE  = 16000
MIN_DUR_SEC  = 0.3   # minimum duration in seconds
MAX_DUR_SEC  = 5.0   # maximum duration in seconds
# -----------------------------------------------------------------

CATEGORY_MAP = {
    "positive": [
        RAW_DIR / "positive",
        RAW_DIR / "real_positive",
    ],
    "negative":   [RAW_DIR / "negative"],
    "background": [RAW_DIR / "background"],
}


def validate_and_clean(src_path: Path, dst_path: Path) -> bool:
    """
    Load audio, validate, normalize, and save to destination.
    Returns True if file passed validation.
    """
    try:
        audio, sr = librosa.load(str(src_path), sr=SAMPLE_RATE, mono=True)

        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=25)

        # Duration check
        dur = len(audio) / SAMPLE_RATE
        if dur < MIN_DUR_SEC or dur > MAX_DUR_SEC:
            return False

        # Check not empty / silent
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-5:
            return False

        # Peak normalize to -3 dBFS
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.7

        # Convert to float32
        audio = audio.astype(np.float32)

        # Save
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst_path), audio, SAMPLE_RATE, subtype="PCM_16")
        return True

    except Exception as e:
        return False


def process_category(category: str, src_dirs: list, global_idx: dict):
    """Process all files for a category."""
    dst_dir = DATASET_DIR / category
    dst_dir.mkdir(parents=True, exist_ok=True)

    all_files = []
    for src_dir in src_dirs:
        if src_dir.exists():
            all_files.extend(sorted(src_dir.glob("*.wav")))

    if not all_files:
        print(f"  [!] No files found for category '{category}'")
        return 0, 0

    ok_count   = 0
    skip_count = 0
    idx        = global_idx.get(category, 0)

    pbar = tqdm(all_files, desc=f"  {category:12s}", unit="file")
    for src_path in pbar:
        dst_name = f"{category[:3]}_{idx:06d}.wav"
        dst_path = dst_dir / dst_name

        if dst_path.exists() and dst_path.stat().st_size > 200:
            ok_count += 1
            idx += 1
            continue

        ok = validate_and_clean(src_path, dst_path)
        if ok:
            ok_count += 1
            idx += 1
        else:
            skip_count += 1

        pbar.set_postfix(ok=ok_count, skip=skip_count)

    global_idx[category] = idx
    return ok_count, skip_count


def main():
    print("=" * 60)
    print("  DATA CLEANING & ORGANIZATION")
    print("=" * 60)

    totals = {}
    global_idx = {}

    for category, src_dirs in CATEGORY_MAP.items():
        print(f"\n[{category.upper()}]")
        ok, skip = process_category(category, src_dirs, global_idx)
        totals[category] = {"ok": ok, "skip": skip}

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    total_files = 0
    for cat, counts in totals.items():
        n = counts["ok"]
        total_files += n
        print(f"  {cat:12s}: {n:5d} files  ({counts['skip']} removed)")
    print(f"  {'TOTAL':12s}: {total_files:5d} files")
    print(f"\n  📁 Dataset at: {DATASET_DIR}")
    print("=" * 60)
    print("\n  [OK] Ready to augment + train!")


if __name__ == "__main__":
    main()
