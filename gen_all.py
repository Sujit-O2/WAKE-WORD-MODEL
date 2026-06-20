"""
gen_all.py - Fast data generation using piper-tts Python API.
Generates positive, negative, and noise data in one script.
"""
import os, sys, random, tempfile, shutil, subprocess
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

PROJECT = Path(__file__).parent
RAW = PROJECT / "dataset_raw"
SAMPLE_RATE = 16000

POS_DIR = RAW / "positive"
NEG_DIR = RAW / "negative"
BG_DIR = RAW / "background"
POS_DIR.mkdir(parents=True, exist_ok=True)
NEG_DIR.mkdir(parents=True, exist_ok=True)
BG_DIR.mkdir(parents=True, exist_ok=True)

# Load piper
from piper import PiperVoice

VOICES_DIR = PROJECT / "tools" / "voices"
VOICES = [
    VOICES_DIR / "en_US-lessac-medium.onnx",
    VOICES_DIR / "en_US-amy-medium.onnx",
    VOICES_DIR / "en_US-ryan-medium.onnx",
    VOICES_DIR / "en_US-joe-medium.onnx",
    VOICES_DIR / "en_US-kusal-medium.onnx",
    VOICES_DIR / "en_GB-alan-medium.onnx",
]
VOICES = [v for v in VOICES if v.exists()]

POSITIVE_TEXTS = [
    "Zerotwo", "Zero two", "zero two", "Zerotwo!",
    "Hey Zerotwo", "zerotwo", "Zero-two", "ZERO TWO",
]

HARD_NEGATIVES = [
    "zero", "two", "hero", "zero to", "zero too",
    "serotwo", "arrow two", "bureau", "see row two",
    "zero gravity", "zero day", "zero shot", "zero sum",
    "true", "blue", "clue", "glue", "shoe", "who",
    "through", "threw", "brew", "dew", "few", "new",
    "caribou", "breakthrough", "tattoo", "voodoo", "shampoo",
    "one two three", "zero one", "two zero", "nine two zero",
]

RANDOM_SPEECH = [
    "turn off the lights", "play some music", "set a timer",
    "what is the weather today", "call my mom", "send a message",
    "open the app", "close the window", "take a photo",
    "hello there", "good morning", "good night", "thank you",
    "please wait", "no problem", "sounds good", "see you later",
    "I do not know", "what do you think", "never mind",
    "the quick brown fox jumps over the lazy dog",
    "she sells seashells by the seashore",
    "how much wood would a woodchuck chuck",
    "to be or not to be that is the question",
    "the weather is nice today",
    "let us go to the park this afternoon",
    "the meeting is scheduled for tomorrow morning",
    "artificial intelligence", "machine learning", "neural network",
    "deep learning", "voice assistant", "smart speaker",
    "monday tuesday wednesday thursday friday",
    "apple banana cherry mango orange pineapple",
    "cat dog bird fish rabbit hamster turtle",
]


def load_voice(idx):
    return PiperVoice.load(str(VOICES[idx % len(VOICES)]))


def synthesize_wav(voice, text, out_path):
    chunks = list(voice.synthesize(text))
    if not chunks:
        return False
    audio = chunks[0].audio_float_array
    sr = chunks[0].sample_rate
    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    sf.write(str(out_path), audio.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
    return True


# === POSITIVE GENERATION ===
print("=" * 55)
print("  GENERATING POSITIVE SAMPLES (1500)")
print("=" * 55)
voices = [load_voice(i) for i in range(len(VOICES))]
idx = 0
ok = 0
for _ in tqdm(range(1500), desc="Positive", unit="wav"):
    text = random.choice(POSITIVE_TEXTS)
    voice = random.choice(voices)
    out = POS_DIR / f"pos_{idx:05d}.wav"
    idx += 1
    if out.exists() and out.stat().st_size > 500:
        ok += 1; continue
    try:
        if synthesize_wav(voice, text, out):
            ok += 1
    except:
        pass
print(f"  Positive: {ok} generated")


# === NEGATIVE GENERATION ===
print("\n" + "=" * 55)
print("  GENERATING NEGATIVE SAMPLES (1500)")
print("=" * 55)
idx = 0; ok = 0
for _ in tqdm(range(1500), desc="Negative", unit="wav"):
    if random.random() < 0.4:
        text = random.choice(HARD_NEGATIVES)
    else:
        text = random.choice(RANDOM_SPEECH)
    voice = random.choice(voices)
    out = NEG_DIR / f"neg_{idx:05d}.wav"
    idx += 1
    if out.exists() and out.stat().st_size > 200:
        ok += 1; continue
    try:
        if synthesize_wav(voice, text, out):
            ok += 1
    except:
        pass
print(f"  Negative: {ok} generated")


# === NOISE GENERATION ===
print("\n" + "=" * 55)
print("  GENERATING NOISE SAMPLES (1500)")
print("=" * 55)

from scipy.signal import butter, filtfilt

def white_noise(n, amp=0.05):
    return (np.random.randn(n) * amp).astype(np.float32)

def pink_noise(n, amp=0.04):
    rows = 16
    arr = np.zeros((rows, n//rows+1)); cols = arr.shape[1]; power = np.zeros(cols)
    for i in range(rows):
        power += np.random.randn(cols); arr[i] = power
    p = arr.sum(axis=0)[:n]
    if len(p) < n: p = np.tile(p, n//len(p)+1)[:n]
    p /= (np.max(np.abs(p)) + 1e-9)
    return (p * amp).astype(np.float32)

def brown_noise(n, amp=0.06):
    bn = np.cumsum(np.random.randn(n)); bn -= bn.mean()
    bn /= (np.max(np.abs(bn)) + 1e-9)
    return (bn * amp).astype(np.float32)

def fan_hum(n, amp=0.03):
    t = np.linspace(0, n/SAMPLE_RATE, n)
    h = np.sin(2*np.pi*50*t)*0.5 + np.sin(2*np.pi*100*t)*0.25 + np.sin(2*np.pi*150*t)*0.12
    h *= (1.0 + 0.03*np.sin(2*np.pi*0.5*t))
    h += white_noise(n, 0.01); h /= (np.max(np.abs(h)) + 1e-9)
    return (h * amp).astype(np.float32)

def rain_noise(n, amp=0.04):
    wn = np.random.randn(n)
    b, a = butter(2, 1000/(SAMPLE_RATE/2), btype="high")
    r = filtfilt(b, a, wn)
    for _ in range(random.randint(5,30)):
        pos = random.randint(0, max(1, n-SAMPLE_RATE//10))
        drop = np.random.randn(SAMPLE_RATE//10)*0.5 * np.exp(-np.linspace(0,8,SAMPLE_RATE//10))
        end = min(pos+len(drop), n); r[pos:end] += drop[:end-pos]
    r /= (np.max(np.abs(r)) + 1e-9)
    return (r * amp).astype(np.float32)

def traffic_rumble(n, amp=0.05):
    bn = np.random.randn(n).cumsum(); bn -= bn.mean()
    b, a = butter(3, 400/(SAMPLE_RATE/2), btype="low")
    t = filtfilt(b, a, bn); t /= (np.max(np.abs(t)) + 1e-9)
    return (t * amp).astype(np.float32)

def mixed_noise(n, amp=0.05):
    s = white_noise(n)*random.uniform(0.3,0.7) + pink_noise(n)*random.uniform(0.3,0.7)
    s /= (np.max(np.abs(s)) + 1e-9)
    return (s * amp).astype(np.float32)

GENS = {
    "white": lambda n: white_noise(n, random.uniform(0.02,0.08)),
    "pink": lambda n: pink_noise(n, random.uniform(0.02,0.07)),
    "brown": lambda n: brown_noise(n, random.uniform(0.03,0.08)),
    "fan": lambda n: fan_hum(n, random.uniform(0.02,0.06)),
    "rain": lambda n: rain_noise(n, random.uniform(0.03,0.07)),
    "traffic": lambda n: traffic_rumble(n, random.uniform(0.03,0.07)),
    "mixed": lambda n: mixed_noise(n, random.uniform(0.03,0.07)),
}
WEIGHTS = {"white":10,"pink":10,"brown":10,"fan":15,"rain":15,"traffic":15,"mixed":15}
types = list(WEIGHTS.keys()); wts = list(WEIGHTS.values())

idx = 0; ok = 0
for _ in tqdm(range(1500), desc="Noise", unit="wav"):
    out = BG_DIR / f"noise_{idx:05d}.wav"
    idx += 1
    if out.exists() and out.stat().st_size > 200:
        ok += 1; continue
    try:
        ntype = random.choices(types, weights=wts, k=1)[0]
        dur = random.uniform(1.0, 3.0)
        nsamples = int(dur * SAMPLE_RATE)
        audio = GENS[ntype](nsamples)
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        sf.write(str(out), audio, SAMPLE_RATE, subtype="PCM_16")
        ok += 1
    except:
        pass
print(f"  Noise: {ok} generated")

print("\n" + "=" * 55)
print("  DATA GENERATION COMPLETE")
print(f"  Positive: {len(list(POS_DIR.glob('*.wav')))} files")
print(f"  Negative: {len(list(NEG_DIR.glob('*.wav')))} files")
print(f"  Background: {len(list(BG_DIR.glob('*.wav')))} files")
print("=" * 55)
print("\n  Next: python 04_clean_and_organize.py")
