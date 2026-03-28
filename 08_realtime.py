"""
08_realtime.py
==============
Real-time wake word detection using the trained ONNX model.

Usage:
  python 08_realtime.py
  python 08_realtime.py --threshold 0.85
  python 08_realtime.py --model models/zerotwo_v2.onnx

Press Ctrl+C to stop.
"""

import sys
import os
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

try:
    import sounddevice as sd
except ImportError:
    os.system(f"{sys.executable} -m pip install sounddevice -q")
    import sounddevice as sd

try:
    import onnxruntime as ort
except ImportError:
    os.system(f"{sys.executable} -m pip install onnxruntime==1.17.0 -q")
    import onnxruntime as ort

try:
    import librosa
except ImportError:
    os.system(f"{sys.executable} -m pip install librosa -q")
    import librosa

# ─────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent
SAMPLE_RATE  = 16000
N_MELS       = 80
HOP_LENGTH   = 160
WIN_LENGTH   = 400
N_FFT        = 512
WINDOW_SEC   = 1.0     # 1 second sliding window
N_FRAMES     = 96       # Must match exported ONNX model (80x96 input)

# Sliding window buffer
BUFFER_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)
STEP_SAMPLES   = BUFFER_SAMPLES // 2   # 50% overlap

CONFIRMATION_FRAMES = 3     # Require 3 consecutive detections
COOLDOWN_SEC        = 1.5   # Ignore triggers for 1.5s after detection
# ─────────────────────────────────────────────────────────────────


def extract_melspec(audio: np.ndarray) -> np.ndarray:
    """Extract mel-spectrogram from audio numpy array."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
    )
    mel_db = librosa.power_to_db(mel + 1e-9, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    mel_db = mel_db[:, :N_FRAMES]
    return mel_db.astype(np.float32)[np.newaxis, np.newaxis, ...]  # (1,1,80,N_FRAMES)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser(description="Zerotwo Wake Word Real-Time Detector")
    parser.add_argument("--model", default="models/zerotwo_v1.onnx", help="ONNX model path")
    parser.add_argument("--threshold", type=float, default=0.80, help="Detection threshold")
    args = parser.parse_args()

    model_path = PROJECT_DIR / args.model
    if not model_path.exists():
        print(f"⚠  Model not found: {model_path}")
        print("   Run 06_train.py first to train the model.")
        sys.exit(1)

    print("=" * 55)
    print("  🎤 ZEROTWO WAKE WORD DETECTOR")
    print(f"  Model:     {model_path.name}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Confirm:   {CONFIRMATION_FRAMES} consecutive frames")
    print("=" * 55)
    print("\n  Loading model...")
    sess = ort.InferenceSession(str(model_path))
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print("  ✓ Model loaded")

    # Ring buffer for audio
    audio_buffer  = np.zeros(BUFFER_SAMPLES, dtype=np.float32)
    score_history = deque(maxlen=CONFIRMATION_FRAMES)
    last_trigger  = 0.0  # timestamp of last confirmed detection

    triggered_count = 0
    frame_count     = 0

    print("\n  Listening... Say 'Zerotwo' clearly.")
    print("  (Press Ctrl+C to stop)\n")

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_buffer, score_history, last_trigger, triggered_count, frame_count

        new_chunk = indata[:, 0].astype(np.float32)  # mono

        # Shift buffer and append new audio
        audio_buffer = np.roll(audio_buffer, -len(new_chunk))
        audio_buffer[-len(new_chunk):] = new_chunk

        frame_count += 1
        # Only run inference every STEP_SAMPLES
        if frame_count % 2 != 0:
            return

        try:
            mel   = extract_melspec(audio_buffer.copy())
            logits = sess.run([output_name], {input_name: mel})[0][0]
            probs  = softmax(logits)
            score  = float(probs[1])

            score_history.append(score)

            # Print score bar
            bar_len = int(score * 30)
            bar     = "█" * bar_len + "░" * (30 - bar_len)
            print(f"\r  Score: [{bar}] {score:.3f}", end="", flush=True)

            # Confirmation: require N consecutive frames above threshold
            now = time.time()
            if (
                len(score_history) == CONFIRMATION_FRAMES
                and all(s >= args.threshold for s in score_history)
                and (now - last_trigger) > COOLDOWN_SEC
            ):
                last_trigger = now
                triggered_count += 1
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"\n\n  🔔 WAKE WORD DETECTED! [{ts}] score={score:.3f} (#{triggered_count})")
                print("  Listening...\n")
                score_history.clear()

        except Exception as e:
            pass  # Silence audio callback errors

    # Start recording
    blocksize = STEP_SAMPLES  # 0.5 sec chunks
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n\n  Stopped. Total detections: {triggered_count}")
    except Exception as e:
        print(f"\n  Error: {e}")
        print("  Make sure your microphone is connected and accessible.")


if __name__ == "__main__":
    main()
