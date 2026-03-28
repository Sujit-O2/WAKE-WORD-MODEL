"""
05_augment.py
=============
Augments the real voice recordings 10x using audiomentations.
This maximizes value from the 27 real "Zerotwo" recordings.

Input:  dataset_raw/real_positive/
Output: dataset/positive/  (adds augmented versions alongside TTS data)
"""

import sys
import random
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

try:
    from audiomentations import (
        Compose, AddGaussianNoise, TimeStretch, PitchShift,
        Shift, Gain, AddGaussianSNR
    )
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "audiomentations", "-q"])
    from audiomentations import (
        Compose, AddGaussianNoise, TimeStretch, PitchShift,
        Shift, Gain, AddGaussianSNR
    )

# ─────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent
REAL_DIR     = PROJECT_DIR / "dataset_raw"  / "real_positive"
OUTPUT_DIR   = PROJECT_DIR / "dataset" / "positive"
SAMPLE_RATE  = 16000
AUGMENT_FACTOR = 10   # Generate 10 variants per real recording
# ─────────────────────────────────────────────────────────────────

# ── Augmentation pipelines (applied randomly) ────────────────────

LIGHT_AUG = Compose([
    AddGaussianSNR(min_snr_db=15, max_snr_db=30, p=0.5),
    Gain(min_gain_db=-3, max_gain_db=3, p=0.5),
    Shift(min_shift=-0.1, max_shift=0.1, p=0.5),
])

MEDIUM_AUG = Compose([
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.025, p=0.6),
    TimeStretch(min_rate=0.85, max_rate=1.15, p=0.6),
    PitchShift(min_semitones=-2.0, max_semitones=2.0, p=0.5),
    Gain(min_gain_db=-5, max_gain_db=5, p=0.4),
])

HEAVY_AUG = Compose([
    AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.05, p=0.8),
    TimeStretch(min_rate=0.80, max_rate=1.20, p=0.7),
    PitchShift(min_semitones=-3.0, max_semitones=3.0, p=0.7),
    Shift(min_shift=-0.15, max_shift=0.15, p=0.5),
    AddGaussianSNR(min_snr_db=8, max_snr_db=20, p=0.5),
])

AUG_PIPELINES = [LIGHT_AUG, MEDIUM_AUG, MEDIUM_AUG, HEAVY_AUG]


def augment_and_save(audio: np.ndarray, out_path: Path) -> bool:
    try:
        pipeline = random.choice(AUG_PIPELINES)
        aug_audio = pipeline(samples=audio.copy(), sample_rate=SAMPLE_RATE)

        # Normalize
        peak = np.max(np.abs(aug_audio))
        if peak > 0:
            aug_audio = aug_audio / peak * 0.7

        aug_audio = np.clip(aug_audio, -1.0, 1.0).astype(np.float32)
        sf.write(str(out_path), aug_audio, SAMPLE_RATE, subtype="PCM_16")
        return True
    except Exception as e:
        return False


def main():
    print("=" * 60)
    print("  REAL RECORDINGS AUGMENTER")
    print(f"  Augment factor: {AUGMENT_FACTOR}x")
    print("=" * 60)

    real_files = sorted(REAL_DIR.glob("*.wav"))
    if not real_files:
        print(f"\n⚠  No real WAV files found in {REAL_DIR}")
        print("   Run 00_process_real.py first.")
        return

    print(f"\n  Found {len(real_files)} real recordings")
    print(f"  Will generate {len(real_files) * AUGMENT_FACTOR} augmented samples")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ok_count   = 0
    fail_count = 0
    aug_idx    = 90000  # High offset to avoid collisions with TTS files

    for src_path in tqdm(real_files, desc="Real files", unit="file"):
        # Load once
        try:
            audio, _ = librosa.load(str(src_path), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"  ✗ Cannot load {src_path.name}: {e}")
            continue

        # Generate AUGMENT_FACTOR variants
        for variant_idx in range(AUGMENT_FACTOR):
            out_path = OUTPUT_DIR / f"pos_real_{aug_idx:06d}.wav"
            aug_idx += 1

            if out_path.exists() and out_path.stat().st_size > 200:
                ok_count += 1
                continue

            ok = augment_and_save(audio, out_path)
            if ok:
                ok_count += 1
            else:
                fail_count += 1

        # Also copy the original (unaugmented) for variety
        orig_out = OUTPUT_DIR / f"pos_real_{aug_idx:06d}_orig.wav"
        aug_idx += 1
        if not orig_out.exists():
            import shutil
            shutil.copy2(src_path, orig_out)
            ok_count += 1

    print("\n" + "=" * 60)
    print(f"  ✓ Augmented: {ok_count} files total")
    print(f"  ✗ Failed:    {fail_count}")
    print(f"  📁 Saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
