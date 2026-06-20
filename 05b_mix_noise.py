"""
05b_mix_noise.py
================
Mixes positive "Zerotwo" audio samples with background noise at various
SNR levels. This is CRITICAL for noise robustness -- the model must learn
to detect the wake word even in noisy environments.

Input:  dataset/positive/ (clean wake word audio)
        dataset/background/ (noise files)
Output: dataset/positive_mixed/ (wake word + noise at various SNR levels)

SNR levels used: 0dB, 5dB, 10dB, 15dB, 20dB
Each positive sample gets 3-5 noise-mixed versions.
"""

import os
import sys
import random
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

# -----------------------------------------------------------------
PROJECT_DIR   = Path(__file__).parent
POS_DIR       = PROJECT_DIR / "dataset" / "positive"
NOISE_DIR     = PROJECT_DIR / "dataset" / "background"
OUTPUT_DIR    = PROJECT_DIR / "dataset" / "positive_mixed"
SAMPLE_RATE   = 16000
TARGET_LEN    = 16000        # 1 second
MIXES_PER_SAMPLE = 4         # noise versions per clean sample
SNR_DB_LEVELS = [0, 5, 10, 15, 20]   # dB -- lower = more noise
# -----------------------------------------------------------------


def load_audio(path: str) -> np.ndarray | None:
    try:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return audio
    except Exception:
        return None


def compute_snr_mixed(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix clean speech with noise at a specific SNR (dB)."""
    # Trim or repeat noise to match clean length
    if len(noise) < len(clean):
        repeats = len(clean) // len(noise) + 1
        noise = np.tile(noise, repeats)
    
    # Random crop noise to same length as clean
    start = random.randint(0, len(noise) - len(clean))
    noise = noise[start:start + len(clean)]
    
    # Compute power
    clean_power = np.mean(clean ** 2) + 1e-9
    noise_power = np.mean(noise ** 2) + 1e-9
    
    # Scale noise to achieve target SNR
    # SNR(dB) = 10*log10(Psignal/Pnoise) -> ratio = 10^(snr_db/10)
    snr_linear = 10 ** (snr_db / 10.0)
    scale = np.sqrt(clean_power / (noise_power * snr_linear + 1e-9))
    noise_scaled = noise * scale
    
    # Mix
    mixed = clean + noise_scaled
    
    # Normalize to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed / peak * 0.9
    
    return mixed.astype(np.float32)


def main():
    print("=" * 60)
    print("  NOISE MIXING FOR WAKE WORD ROBUSTNESS")
    print(f"  SNR levels: {SNR_DB_LEVELS}")
    print(f"  Mixes per sample: {MIXES_PER_SAMPLE}")
    print("=" * 60)

    if not POS_DIR.exists():
        print(f"  !! Positive dir not found: {POS_DIR}")
        return
    if not NOISE_DIR.exists():
        print(f"  !! Noise dir not found: {NOISE_DIR}")
        return

    # Load all noise files into memory (they're small)
    print("\n  Loading noise files...")
    noise_files = sorted(NOISE_DIR.glob("*.wav"))
    noise_audios = []
    for nf in tqdm(noise_files, desc="  Loading noise", unit="file"):
        audio = load_audio(str(nf))
        if audio is not None and len(audio) > 100:
            noise_audios.append(audio)
    print(f"  Loaded {len(noise_audios)} noise files")

    if not noise_audios:
        print("  !! No noise files loaded. Aborting.")
        return

    # Process positive files
    pos_files = sorted(POS_DIR.glob("*.wav"))
    print(f"\n  Found {len(pos_files)} positive files")
    print(f"  Will generate ~{len(pos_files) * MIXES_PER_SAMPLE} mixed samples")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    fail_count = 0
    out_idx = 0

    for pos_path in tqdm(pos_files, desc="  Mixing", unit="file"):
        clean = load_audio(str(pos_path))
        if clean is None:
            fail_count += 1
            continue

        # Pad/trim to target length
        if len(clean) < TARGET_LEN:
            clean = np.pad(clean, (0, TARGET_LEN - len(clean)))
        else:
            # Random crop
            start = random.randint(0, len(clean) - TARGET_LEN)
            clean = clean[start:start + TARGET_LEN]

        # Generate MIXES_PER_SAMPLE noisy versions
        for _ in range(MIXES_PER_SAMPLE):
            noise = random.choice(noise_audios)
            snr_db = random.choice(SNR_DB_LEVELS)
            mixed = compute_snr_mixed(clean, noise, snr_db)

            out_path = OUTPUT_DIR / f"mix_{out_idx:06d}.wav"
            out_idx += 1

            try:
                sf.write(str(out_path), mixed, SAMPLE_RATE, subtype="PCM_16")
                ok_count += 1
            except Exception:
                fail_count += 1

    print("\n" + "=" * 60)
    print(f"  Done! Mixed samples: {ok_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)
    print("\n  Next: Run 06_train.py to train with noise-robust data")


if __name__ == "__main__":
    main()
