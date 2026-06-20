"""
03_generate_negative.py
=======================
Generates 1500 negative speech samples using Piper TTS.
These are speech clips of words/phrases that are NOT "Zerotwo" --
including hard negatives (phonetically similar) and diverse speech.

Requires: Piper TTS (downloaded by 01_generate_positive.py)
          Voice models in tools/voices/

Output: zerotwo_wake/dataset_raw/negative/  (1500 WAV files, 16kHz mono)
"""

import os
import sys
import random
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm

# -----------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
PIPER_EXE   = PROJECT_DIR / "tools" / "piper" / "piper.exe"
VOICES_DIR  = PROJECT_DIR / "tools" / "voices"
ESPEAK_DIR  = PROJECT_DIR / "tools" / "piper" / "espeak-ng-data"
OUTPUT_DIR  = PROJECT_DIR / "dataset_raw" / "negative"
TARGET      = 2000
# -----------------------------------------------------------------

# -- HARD NEGATIVES (phonetically similar -- most important!) -------
HARD_NEGATIVES = [
    # Direct confusables
    "zero", "two", "hero", "zero to", "zero too",
    "serotwo", "zerotoo", "zero-two", "see row two",
    "zerotu", "zirotwo", "zero tow", "arrow two",
    "zero two three", "one two three",
    "see row", "row two", "arrow", "bureau",
    # Similar endings
    "caribou", "breakthrough", "tattoo", "voodoo", "shampoo",
    "igloo", "taboo", "bamboo", "kung fu", "debut",
    # Similar beginnings
    "zero gravity", "zero day", "zero shot", "zero sum",
    "zero waste", "zero point", "zero hour", "zero tolerance",
    # Rhymes with "two"
    "true", "blue", "clue", "glue", "shoe", "who",
    "through", "threw", "brew", "dew", "few", "new",
    # ADVERSARIAL: partial wake word + filler (strongest false trigger traps)
    "hey zero", "ok two", "listen two", "yo two",
    "zero and two", "zero two one", "zero two go",
    "set zero two", "call zero two", "open zero two",
    "zero to two", "near zero", "close to zero",
    "hero two", "nero two", "dero two",
    "let's go two", "go to two", "what two",
    "there are two", "only two", "just two",
    "we need two", "give me two", "number two",
]

# -- RANDOM SPEECH (diverse sentences to prevent false triggers) ---
RANDOM_SPEECH = [
    # Everyday commands
    "turn off the lights", "play some music", "set a timer",
    "what's the weather today", "call my mom", "send a message",
    "open the app", "close the window", "take a photo",
    "search the internet", "navigate to home", "go back",
    # Numbers and counting
    "one two three four five six seven eight nine ten",
    "first second third", "five minutes", "two hours",
    "thirty seconds", "ten thousand", "five hundred",
    # Common words + phrases
    "hello there", "good morning", "good night", "thank you",
    "please wait", "no problem", "sounds good", "see you later",
    "I don't know", "what do you think", "never mind",
    # Tech phrases
    "artificial intelligence", "machine learning", "neural network",
    "deep learning", "natural language processing",
    "voice assistant", "smart speaker", "wake word detection",
    # Random sentences
    "the quick brown fox jumps over the lazy dog",
    "she sells seashells by the seashore",
    "how much wood would a woodchuck chuck",
    "peter piper picked a peck of pickled peppers",
    "to be or not to be that is the question",
    "the weather is nice today isn't it",
    "I went to the store to buy some groceries",
    "can you help me with this problem please",
    "let's go to the park this afternoon",
    "the meeting is scheduled for tomorrow morning",
    # Numbers streamed together (confusable)
    "zero one", "zero zero", "one zero", "three two one",
    "two zero", "four two", "nine two zero",
    # Days / months
    "monday tuesday wednesday thursday friday saturday sunday",
    "january february march april may june",
    "july august september october november december",
    # Random nouns
    "apple banana cherry mango orange pineapple",
    "cat dog bird fish rabbit hamster turtle",
    "chair table desk lamp phone computer screen",
    # Filler speech
    "um uh let me think about that for a moment",
    "so yeah like I was saying earlier",
    "well actually I think that depends on",
    "hmm that's an interesting question",
    # Multi-lingual-sounding words (non-English phrases)
    "bonjour comment allez vous aujourd hui",
    "gracias por su ayuda me alegra mucho",
    "guten morgen wie geht es ihnen heute",
    "ciao come stai spero bene grazie mille",
    # Technical jargon
    "compile the source code and run the tests",
    "push to the main branch after code review",
    "the API endpoint returns a JSON response",
    "configure the environment variables first",
    # Sports / entertainment
    "the team scored three goals in the second half",
    "the movie starts at eight thirty tonight",
    "let's grab some food after the game",
    # Directions / navigation
    "turn left at the intersection ahead",
    "continue straight for two miles then exit",
    "your destination is on the right in fifty feet",
    # Weather
    "partly cloudy with a chance of rain tomorrow",
    "temperature will drop to sixty degrees tonight",
    "strong winds expected throughout the afternoon",
]

# -- Speed variations ----------------------------------------------
SPEED_VARIANTS = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]

ALL_PHRASES = HARD_NEGATIVES + RANDOM_SPEECH


def get_available_voices():
    """Return list of (voice_name, onnx_path) for installed voices."""
    voices = []
    for onnx_path in VOICES_DIR.glob("*.onnx"):
        voices.append(onnx_path)
    return voices


def generate_sample(voice_onnx: Path, text: str, output_wav: Path, speed: float) -> bool:
    """Piper writes to --output_dir as numbered files. Use temp dir to avoid collisions."""
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
            tmp_wavs = sorted(Path(tmpdir).glob("*.wav"))
            if tmp_wavs and tmp_wavs[0].stat().st_size > 500:
                output_wav.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(tmp_wavs[0], output_wav)
                return True
        return False
    except Exception:
        return False


def main():
    print("=" * 60)
    print("  ZEROTWO NEGATIVE SPEECH GENERATOR")
    print(f"  Target: {TARGET} negative samples")
    print(f"  Phrases pool: {len(ALL_PHRASES)} unique phrases")
    print("=" * 60)

    if not PIPER_EXE.exists():
        print(f"\n[!]  Piper not found at {PIPER_EXE}")
        print("   Please run 01_generate_positive.py first to download Piper.")
        sys.exit(1)

    voices = get_available_voices()
    if not voices:
        print(f"\n[!]  No voice models found in {VOICES_DIR}")
        print("   Please run 01_generate_positive.py first to download voices.")
        sys.exit(1)

    print(f"\n  Using {len(voices)} voice(s): {[v.stem for v in voices]}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Split: 40% hard negatives, 60% random speech
    n_hard   = int(TARGET * 0.40)
    n_random = TARGET - n_hard

    # Build generation plan: (phrase, voice, speed)
    plan = []

    # Hard negatives -- cycle through them
    for i in range(n_hard):
        phrase = HARD_NEGATIVES[i % len(HARD_NEGATIVES)]
        voice  = random.choice(voices)
        speed  = random.choice(SPEED_VARIANTS) + random.uniform(-0.02, 0.02)
        plan.append(("hard", phrase, voice, speed))

    # Random speech
    for i in range(n_random):
        phrase = RANDOM_SPEECH[i % len(RANDOM_SPEECH)]
        voice  = random.choice(voices)
        speed  = random.choice(SPEED_VARIANTS) + random.uniform(-0.02, 0.02)
        plan.append(("random", phrase, voice, speed))

    random.shuffle(plan)  # Mix hard + random

    total_ok  = 0
    total_fail = 0

    pbar = tqdm(enumerate(plan), total=len(plan), desc="Generating negatives", unit="file")
    for idx, (category, phrase, voice, speed) in pbar:
        out_wav = OUTPUT_DIR / f"neg_{idx:05d}_{category[:4]}.wav"

        if out_wav.exists() and out_wav.stat().st_size > 200:
            total_ok += 1
            continue

        ok = generate_sample(voice, phrase, out_wav, speed)
        if ok:
            total_ok += 1
        else:
            total_fail += 1

        pbar.set_postfix(ok=total_ok, fail=total_fail, cat=category[:4])

    print("\n" + "=" * 60)
    print(f"  [OK] Generated: {total_ok} negative samples")
    print(f"    Hard negatives:  ~{n_hard}")
    print(f"    Random speech:   ~{n_random}")
    print(f"  [X] Failed:    {total_fail}")
    print(f"  📁 Saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
