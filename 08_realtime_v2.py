"""
08_realtime_v2.py
=================
Real-time wake word detection with adaptive thresholding.
Supports both v1 (1-ch) and v2 (3-ch) ONNX models.
"""
import sys, os, time, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

import sounddevice as sd
import onnxruntime as ort
import librosa

PROJECT_DIR = Path(__file__).parent
SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 160
WIN_LENGTH = 400
N_FFT = 512
N_FRAMES = 100
BUFFER_SAMPLES = 16000
STEP_SAMPLES = 8000
CONFIRM_FRAMES = 3
COOLDOWN = 1.5


def extract_3ch(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    mel_db = librosa.power_to_db(mel+1e-9, ref=np.max)
    delta = librosa.feature.delta(mel_db); delta2 = librosa.feature.delta(mel_db, order=2)
    feat = np.stack([mel_db, delta, delta2], axis=0)
    for c in range(3):
        feat[c] = (feat[c]-feat[c].mean())/(feat[c].std()+1e-9)
    return feat[:,:N_FRAMES].astype(np.float32)[np.newaxis,...]


def extract_1ch(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    mel_db = librosa.power_to_db(mel+1e-9, ref=np.max)
    mel_db = (mel_db-mel_db.mean())/(mel_db.std()+1e-9)
    return mel_db[:,:N_FRAMES].astype(np.float32)[np.newaxis,np.newaxis,...]


def softmax(x):
    e = np.exp(x - x.max()); return e/e.sum()


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", default="models/zerotwo_v2.onnx")
    pa.add_argument("--threshold", type=float, default=0.75)
    pa.add_argument("--no-adaptive", action="store_true")
    a = pa.parse_args()

    mp = PROJECT_DIR / a.model
    if not mp.exists():
        mp = PROJECT_DIR / "models" / "zerotwo_v1.onnx"
        if not mp.exists():
            print("  No model found. Run 06_train.py first."); sys.exit(1)

    sess = ort.InferenceSession(str(mp))
    inp = sess.get_inputs()[0].name; out = sess.get_outputs()[0].name
    shape = sess.get_inputs()[0].shape
    is_v2 = isinstance(shape[1], int) and shape[1] == 3

    print("="*55)
    print("  ZEROTWO DETECTOR v2")
    print(f"  Model: {mp.name} ({'3-ch' if is_v2 else '1-ch'})")
    print(f"  Threshold: {a.threshold} | Adaptive: {not a.no_adaptive}")
    print("="*55)

    buf = np.zeros(BUFFER_SAMPLES, dtype=np.float32)
    hist = deque(maxlen=CONFIRM_FRAMES)
    ehist = deque(maxlen=30)
    last_trig = 0.0; trig_count = 0; frame = 0

    print("\n  Listening... Say 'Zerotwo'\n")

    def callback(indata, frames, time_info, status):
        nonlocal buf, hist, last_trig, trig_count, frame
        chunk = indata[:,0].astype(np.float32)
        buf = np.roll(buf, -len(chunk)); buf[-len(chunk):] = chunk
        frame += 1
        if frame % 2 != 0: return
        try:
            thr = a.threshold
            if not a.no_adaptive:
                rms = np.sqrt(np.mean(buf**2)+1e-9)
                ehist.append(rms)
                if len(ehist) > 5:
                    avg = np.mean(list(ehist))
                    if avg > 0.05: thr = min(a.threshold+0.1, 0.95)
                    elif avg > 0.02: thr = a.threshold+0.05

            mi = extract_3ch(buf.copy()) if is_v2 else extract_1ch(buf.copy())
            score = float(softmax(sess.run([out], {inp: mi})[0][0])[1])
            hist.append(score)

            bar = "#"*int(score*30) + "-"*(30-int(score*30))
            print(f"\r  [{bar}] {score:.3f} thr={thr:.2f}", end="", flush=True)

            now = time.time()
            if (len(hist)==CONFIRM_FRAMES and all(s>=thr for s in hist)
                    and (now-last_trig)>COOLDOWN):
                last_trig = now; trig_count += 1
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"\n\n  DETECTED! [{ts}] score={score:.3f} (#{trig_count})\n")
                hist.clear()
        except: pass

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=STEP_SAMPLES, callback=callback):
            while True: time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n  Stopped. Detections: {trig_count}")
    except Exception as e:
        print(f"\n  Error: {e}")


if __name__ == "__main__":
    main()
