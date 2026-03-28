"""
02_generate_noise.py
====================
Generates 1500 background / ambient noise files using numpy.
100% offline — no downloads needed.

Noise types:
  - White noise
  - Pink noise (1/f approximation)
  - Brown noise (Brownian)
  - Fan / HVAC hum
  - Rain simulation
  - Traffic rumble
  - Keyboard clicks
  - Silence (near-silence)
  - Mixed / layered combinations

Output: zerotwo_wake/dataset_raw/background/  (1500 WAV files, 16kHz mono)
"""

import os
import sys
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR  = PROJECT_DIR / "dataset_raw" / "background"
TARGET      = 1500          # files to generate
SAMPLE_RATE = 16000         # Hz
MIN_DUR_SEC = 1.0           # minimum clip duration
MAX_DUR_SEC = 3.0           # maximum clip duration
# ─────────────────────────────────────────────────────────────────


# ── Noise generators ──────────────────────────────────────────────

def white_noise(n: int, amp: float = 0.05) -> np.ndarray:
    return (np.random.randn(n) * amp).astype(np.float32)


def pink_noise(n: int, amp: float = 0.04) -> np.ndarray:
    """Voss-McCartney 1/f pink noise approximation."""
    rows  = 16
    array = np.zeros((rows, n // rows + 1))
    cols  = array.shape[1]
    power = np.zeros(cols)
    for i in range(rows):
        power += np.random.randn(cols)
        array[i] = power
    # Trim / expand to exactly n
    pink = array.sum(axis=0)
    pink = pink[:n] if len(pink) >= n else np.tile(pink, n // len(pink) + 1)[:n]
    pink /= (np.max(np.abs(pink)) + 1e-9)
    return (pink * amp).astype(np.float32)


def brown_noise(n: int, amp: float = 0.06) -> np.ndarray:
    """Brownian / red noise (integrated white noise)."""
    wn  = np.random.randn(n)
    bn  = np.cumsum(wn)
    bn -= bn.mean()
    bn /= (np.max(np.abs(bn)) + 1e-9)
    return (bn * amp).astype(np.float32)


def fan_hum(n: int, sr: int, amp: float = 0.03) -> np.ndarray:
    """Steady electrical hum (50 Hz + harmonics) with slight flutter."""
    t    = np.linspace(0, n / sr, n)
    hum  = np.sin(2 * np.pi * 50 * t)  * 0.5
    hum += np.sin(2 * np.pi * 100 * t) * 0.25
    hum += np.sin(2 * np.pi * 150 * t) * 0.12
    hum += np.sin(2 * np.pi * 200 * t) * 0.06
    # Add slight flutter
    flutter = 1.0 + 0.03 * np.sin(2 * np.pi * 0.5 * t)
    hum     = hum * flutter
    hum    += white_noise(n, amp=0.01)    # add light noise
    hum    /= (np.max(np.abs(hum)) + 1e-9)
    return (hum * amp).astype(np.float32)


def rain_noise(n: int, sr: int, amp: float = 0.04) -> np.ndarray:
    """Simulated rain: high-pass filtered noise + occasional drops."""
    from scipy.signal import butter, filtfilt
    wn   = np.random.randn(n)
    # High-pass filter at 1kHz to approximate rain hiss
    b, a = butter(2, 1000 / (sr / 2), btype="high")
    rain = filtfilt(b, a, wn)
    # Occasional droplet splashes
    n_drops = random.randint(5, 30)
    for _ in range(n_drops):
        pos  = random.randint(0, max(1, n - sr // 10))
        drop = np.random.randn(sr // 10) * 0.5
        drop = drop * np.exp(-np.linspace(0, 8, len(drop)))
        end  = min(pos + len(drop), n)
        rain[pos:end] += drop[:end - pos]
    rain /= (np.max(np.abs(rain)) + 1e-9)
    return (rain * amp).astype(np.float32)


def traffic_rumble(n: int, sr: int, amp: float = 0.05) -> np.ndarray:
    """Low-frequency traffic rumble: low-pass filtered brown noise."""
    from scipy.signal import butter, filtfilt
    bn   = np.random.randn(n).cumsum()
    bn  -= bn.mean()
    # Low-pass filter at 400 Hz
    b, a = butter(3, 400 / (sr / 2), btype="low")
    traf = filtfilt(b, a, bn)
    traf /= (np.max(np.abs(traf)) + 1e-9)
    # Add faint honks (random short sinusoids)
    if random.random() < 0.3:
        for _ in range(random.randint(1, 3)):
            pos  = random.randint(0, max(1, n - sr // 4))
            dur  = sr // 4
            end  = min(pos + dur, n)
            freq = random.choice([440, 660, 880, 523])
            t    = np.linspace(0, dur / sr, end - pos)
            traf[pos:end] += np.sin(2 * np.pi * freq * t) * 0.3
    traf /= (np.max(np.abs(traf)) + 1e-9)
    return (traf * amp).astype(np.float32)


def keyboard_clicks(n: int, sr: int, amp: float = 0.04) -> np.ndarray:
    """Simulated typing: sparse short impulses + white noise bed."""
    buf   = white_noise(n, amp=0.005)
    n_clicks = random.randint(3, 20)
    for _ in range(n_clicks):
        pos   = random.randint(0, max(1, n - 160))
        click = white_noise(160, amp=0.5) * np.exp(-np.linspace(0, 6, 160))
        end   = min(pos + 160, n)
        buf[pos:end] += click[:end - pos]
    buf /= (np.max(np.abs(buf)) + 1e-9)
    return (buf * amp).astype(np.float32)


def near_silence(n: int, amp: float = 0.002) -> np.ndarray:
    """Very quiet background hiss — simulates 'silence' with noise floor."""
    return white_noise(n, amp=amp)


def mixed(n: int, sr: int, amp: float = 0.05) -> np.ndarray:
    """Combine 2–3 noise types at random weights."""
    sources = [
        white_noise(n) * random.uniform(0.3, 0.7),
        pink_noise(n)  * random.uniform(0.3, 0.7),
    ]
    if random.random() < 0.5:
        sources.append(fan_hum(n, sr) * random.uniform(0.3, 0.7))
    mix = sum(sources)
    mix /= (np.max(np.abs(mix)) + 1e-9)
    return (mix * amp).astype(np.float32)


# ── Registry ──────────────────────────────────────────────────────
GENERATORS = {
    "white":     lambda n, sr: white_noise(n,     amp=random.uniform(0.02, 0.08)),
    "pink":      lambda n, sr: pink_noise(n,      amp=random.uniform(0.02, 0.07)),
    "brown":     lambda n, sr: brown_noise(n,     amp=random.uniform(0.03, 0.08)),
    "fan":       lambda n, sr: fan_hum(n, sr,     amp=random.uniform(0.02, 0.06)),
    "rain":      lambda n, sr: rain_noise(n, sr,  amp=random.uniform(0.03, 0.07)),
    "traffic":   lambda n, sr: traffic_rumble(n, sr, amp=random.uniform(0.03, 0.07)),
    "keyboard":  lambda n, sr: keyboard_clicks(n, sr, amp=random.uniform(0.02, 0.05)),
    "silence":   lambda n, sr: near_silence(n,   amp=random.uniform(0.001, 0.005)),
    "mixed":     lambda n, sr: mixed(n, sr,       amp=random.uniform(0.03, 0.07)),
}

# Weighted distribution (traffic/rain/fan are most realistic for mobile)
NOISE_WEIGHTS = {
    "white":    10,
    "pink":     10,
    "brown":    10,
    "fan":      15,
    "rain":     15,
    "traffic":  15,
    "keyboard": 10,
    "silence":   5,
    "mixed":    10,
}


def pick_noise_type() -> str:
    types  = list(NOISE_WEIGHTS.keys())
    weights= list(NOISE_WEIGHTS.values())
    return random.choices(types, weights=weights, k=1)[0]


def main():
    # scipy check
    try:
        import scipy.signal
    except ImportError:
        print("Installing scipy...")
        os.system(f"{sys.executable} -m pip install scipy -q")
        import scipy.signal

    print("=" * 60)
    print("  ZEROTWO BACKGROUND NOISE GENERATOR")
    print(f"  Target: {TARGET} noise files")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_ok  = 0
    total_err = 0

    pbar = tqdm(range(TARGET), desc="Generating noise", unit="file")
    for i in pbar:
        out_path = OUTPUT_DIR / f"noise_{i:05d}.wav"

        # Skip if already done
        if out_path.exists() and out_path.stat().st_size > 200:
            total_ok += 1
            continue

        try:
            noise_type = pick_noise_type()
            dur_sec    = random.uniform(MIN_DUR_SEC, MAX_DUR_SEC)
            n_samples  = int(dur_sec * SAMPLE_RATE)

            audio = GENERATORS[noise_type](n_samples, SAMPLE_RATE)

            # Safety clip to [-1, 1]
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

            sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
            total_ok += 1
            pbar.set_postfix(type=noise_type[:6], ok=total_ok)

        except Exception as e:
            total_err += 1
            pbar.set_postfix(err=str(e)[:30])

    print("\n" + "=" * 60)
    print(f"  ✓ Generated: {total_ok} noise files")
    print(f"  ✗ Errors:    {total_err}")
    print(f"  📁 Saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
