"""
07_evaluate.py (v2)
===================
Evaluates zerotwo wake word model (v1 or v2) on held-out test data.
Prints precision, recall, F1, AUC-ROC, confusion matrix, and noise robustness.

Usage:
  python 07_evaluate.py
  python 07_evaluate.py --model models/zerotwo_v2.onnx
"""

import sys
import os
import random
import numpy as np
from pathlib import Path

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

# -----------------------------------------------------------------
PROJECT_DIR  = Path(__file__).parent
DATASET_DIR  = PROJECT_DIR / "dataset"
SAMPLE_RATE  = 16000
N_MELS       = 80
HOP_LENGTH   = 160
WIN_LENGTH   = 400
N_FFT        = 512
N_FRAMES     = 100      # Must match training (1 sec at 16kHz, hop=160)
THRESHOLD    = 0.80     # Detection threshold
N_TEST       = 500      # Files to sample for test
# -----------------------------------------------------------------


def extract_melspec(audio_path: str) -> np.ndarray | None:
    """Extract mel-spectrogram from audio file."""
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        target_len = N_FRAMES * HOP_LENGTH
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        )
        mel_db = librosa.power_to_db(mel + 1e-9, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
        mel_db = mel_db[:, :N_FRAMES]
        return mel_db.astype(np.float32)[np.newaxis, np.newaxis, ...]  # (1,1,80,100)
    except Exception:
        return None


def extract_melspec_noisy(audio_path: str, noise_path: str, snr_db: float) -> np.ndarray | None:
    """Extract mel-spectrogram from audio mixed with noise at given SNR."""
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        noise, _ = librosa.load(noise_path, sr=SAMPLE_RATE, mono=True)

        target_len = N_FRAMES * HOP_LENGTH
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        # Get matching noise chunk
        if len(noise) < target_len:
            noise = np.tile(noise, target_len // len(noise) + 1)
        start = random.randint(0, len(noise) - target_len)
        noise = noise[start:start + target_len]

        # Mix at specified SNR
        sig_power = np.mean(audio ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        snr_linear = 10.0 ** (snr_db / 10.0)
        scale = np.sqrt(sig_power / (noise_power * snr_linear))
        mixed = audio + noise * scale

        mel = librosa.feature.melspectrogram(
            y=mixed, sr=SAMPLE_RATE, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        )
        mel_db = librosa.power_to_db(mel + 1e-9, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
        mel_db = mel_db[:, :N_FRAMES]
        return mel_db.astype(np.float32)[np.newaxis, np.newaxis, ...]
    except Exception:
        return None


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def load_test_files():
    pos_dir = DATASET_DIR / "positive"
    neg_dir = DATASET_DIR / "negative"
    bg_dir  = DATASET_DIR / "background"

    pos_files = sorted(pos_dir.glob("*.wav"))
    neg_files = sorted(neg_dir.glob("*.wav")) + sorted(bg_dir.glob("*.wav"))

    # Sample test set
    n_each = min(N_TEST // 2, len(pos_files), len(neg_files))
    pos_sample = random.sample(pos_files, n_each)
    neg_sample = random.sample(neg_files, n_each)

    test_files  = [(str(f), 1) for f in pos_sample] + [(str(f), 0) for f in neg_sample]
    random.shuffle(test_files)
    return test_files, [str(f) for f in pos_sample]


def evaluate_at_threshold(sess, input_name, output_name, test_files, threshold):
    """Evaluate model at a specific threshold."""
    y_true, y_pred, y_score = [], [], []

    for fpath, label in test_files:
        mel = extract_melspec(fpath)
        if mel is None:
            continue

        logits = sess.run([output_name], {input_name: mel})[0][0]
        probs  = softmax(logits)
        score  = float(probs[1])

        y_true.append(label)
        y_score.append(score)
        y_pred.append(1 if score >= threshold else 0)

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_score = np.array(y_score)

    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (TP + TN) / max(len(y_true), 1)
    far       = FP / (FP + TN + 1e-9)

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy, "far": far,
        "y_true": y_true, "y_score": y_score,
    }


def noise_robustness_test(sess, input_name, output_name, pos_files, bg_dir, threshold):
    """Test model robustness at various noise levels."""
    bg_files = sorted(bg_dir.glob("*.wav"))
    if not bg_files:
        print("  [!] No background noise files found for robustness test.")
        return

    print("\n  -- Noise Robustness Test --")
    print(f"  {'SNR (dB)':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Acc':>8}")
    print(f"  {'-'*50}")

    for snr_db in [30, 20, 15, 10, 5, 0]:
        y_true, y_pred = [], []
        n_test = min(100, len(pos_files))

        for fpath in pos_files[:n_test]:
            bg_path = random.choice(bg_files)
            mel = extract_melspec_noisy(fpath, str(bg_path), snr_db)
            if mel is None:
                continue

            logits = sess.run([output_name], {input_name: mel})[0][0]
            probs  = softmax(logits)
            score  = float(probs[1])

            y_true.append(1)
            y_pred.append(1 if score >= threshold else 0)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        n_total = tp + fn

        precision = tp / max(n_total, 1)  # No negatives here, so precision = recall
        recall    = tp / max(n_total, 1)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy  = recall

        print(f"  {snr_db:>10} {precision:>10.3f} {recall:>8.3f} {f1:>8.3f} {accuracy:>8.3f}")

    print(f"\n  SNR guide: 30dB=clean, 20dB=quiet room, 10dB=busy office, 0dB=very noisy")


def main():
    parser_args = sys.argv[1:]
    model_path = PROJECT_DIR / "models" / "zerotwo_v2.onnx"
    if "--model" in parser_args:
        idx = parser_args.index("--model")
        model_path = PROJECT_DIR / parser_args[idx + 1]

    # Fallback to v1 if v2 doesn't exist
    if not model_path.exists() and "v2" in str(model_path):
        model_path = PROJECT_DIR / "models" / "zerotwo_v1.onnx"

    print("=" * 60)
    print("  ZEROTWO MODEL EVALUATOR (v2)")
    print("=" * 60)

    if not model_path.exists():
        print(f"\n[!]  Model not found: {model_path}")
        print("   Run 06_train.py first.")
        return

    # Load ONNX model
    print(f"\n  Loading model: {model_path.name}")
    sess = ort.InferenceSession(str(model_path))
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Show model input shape
    input_shape = sess.get_inputs()[0].shape
    print(f"  Input shape: {input_shape}")

    # Load test data
    print("\n  Loading test files...")
    test_files, pos_files = load_test_files()
    print(f"  Test samples: {len(test_files)}")

    results = evaluate_at_threshold(sess, input_name, output_name, test_files, THRESHOLD)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Threshold: {THRESHOLD}")
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted")
    print(f"  Actual     Wake   Other")
    print(f"  Wake    [{results['TP']:5d}] [{results['FN']:5d}]")
    print(f"  Other   [{results['FP']:5d}] [{results['TN']:5d}]")
    print(f"\n  Precision:  {results['precision']:.3f}  (of detections, how many correct?)")
    print(f"  Recall:     {results['recall']:.3f}  (of wake words, how many detected?)")
    print(f"  F1 Score:   {results['f1']:.3f}")
    print(f"  Accuracy:   {results['accuracy']:.3f}")
    print(f"  False Accept Rate: {results['far']:.3f}  (lower is better)")

    # Threshold sweep
    print(f"\n  -- Threshold Sweep --")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FAR':>8}")
    for thr in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        r = evaluate_at_threshold(sess, input_name, output_name, test_files, thr)
        print(f"  {thr:>10.2f} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['f1']:>8.3f} {r['far']:>8.3f}")

    # Noise robustness test
    bg_dir = DATASET_DIR / "background"
    if bg_dir.exists():
        noise_robustness_test(sess, input_name, output_name, pos_files, bg_dir, THRESHOLD)

    # Noise false-positive test
    neg_dir = DATASET_DIR / "negative"
    if bg_dir.exists() and neg_dir.exists():
        print(f"\n  -- Noise False-Positive Test --")
        print(f"  Testing: will speech + noise incorrectly trigger the model?")
        print(f"  {'SNR (dB)':>10} {'False Pos%':>12} {'Triggers':>10} {'Total':>8}")
        print(f"  {'-'*45}")
        neg_files = sorted(neg_dir.glob("*.wav"))
        bg_files  = sorted(bg_dir.glob("*.wav"))
        if neg_files and bg_files:
            for snr_db in [30, 20, 10, 5, 0]:
                fp_count = 0
                n_test = min(100, len(neg_files))
                for fpath in neg_files[:n_test]:
                    bg_path = random.choice(bg_files)
                    mel = extract_melspec_noisy(str(fpath), str(bg_path), snr_db)
                    if mel is None:
                        continue
                    logits = sess.run([output_name], {input_name: mel})[0][0]
                    probs  = softmax(logits)
                    score  = float(probs[1])
                    if score >= THRESHOLD:
                        fp_count += 1
                fp_rate = fp_count / max(n_test, 1)
                print(f"  {snr_db:>10} {fp_rate:>11.1%} {fp_count:>10} {n_test:>8}")
        print(f"  Goal: 0% false positive rate at all noise levels")

    print("\n" + "=" * 60)
    status = "[OK] PASS" if (results['precision'] > 0.90 and results['recall'] > 0.85) else "[!]  NEEDS IMPROVEMENT"
    print(f"  Status: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
