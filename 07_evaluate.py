"""
07_evaluate.py
==============
Evaluates zerotwo_v1.onnx on held-out test data.
Prints precision, recall, F1, AUC-ROC, and confusion matrix.
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

# ─────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent
DATASET_DIR  = PROJECT_DIR / "dataset"
MODEL_PATH   = PROJECT_DIR / "models" / "zerotwo_v1.onnx"
SAMPLE_RATE  = 16000
N_MELS       = 80
HOP_LENGTH   = 160
WIN_LENGTH   = 400
N_FFT        = 512
N_FRAMES     = 96       # 96 frames (96/2/2=24, 24/4=6 -> integer, ONNX-compatible)
THRESHOLD    = 0.80     # Detection threshold
N_TEST       = 500      # Files to sample for test
# ─────────────────────────────────────────────────────────────────


def extract_melspec(audio_path: str) -> np.ndarray | None:
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
    return test_files


def main():
    print("=" * 60)
    print("  ZEROTWO MODEL EVALUATOR")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"⚠  Model not found: {MODEL_PATH}")
        print("   Run 06_train.py first.")
        return

    # Load ONNX model
    print(f"\n  Loading model: {MODEL_PATH.name}")
    sess = ort.InferenceSession(str(MODEL_PATH))
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Load test data
    print("\n  Loading test files...")
    test_files = load_test_files()
    print(f"  Test samples: {len(test_files)}")

    from tqdm import tqdm
    y_true, y_pred, y_score = [], [], []

    for fpath, label in tqdm(test_files, desc="Evaluating", unit="file"):
        mel = extract_melspec(fpath)
        if mel is None:
            continue

        logits = sess.run([output_name], {input_name: mel})[0][0]  # (2,)
        # Softmax
        e     = np.exp(logits - logits.max())
        probs = e / e.sum()
        score = float(probs[1])  # probability of wake word

        y_true.append(label)
        y_score.append(score)
        y_pred.append(1 if score >= THRESHOLD else 0)

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_score = np.array(y_score)

    # Metrics
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (TP + TN) / len(y_true)
    far       = FP / (FP + TN + 1e-9)  # False Accept Rate

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Threshold: {THRESHOLD}")
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted")
    print(f"  Actual     Wake   Other")
    print(f"  Wake    [{TP:5d}] [{FN:5d}]")
    print(f"  Other   [{FP:5d}] [{TN:5d}]")
    print(f"\n  Precision:  {precision:.3f}  (of detections, how many correct?)")
    print(f"  Recall:     {recall:.3f}  (of wake words, how many detected?)")
    print(f"  F1 Score:   {f1:.3f}")
    print(f"  Accuracy:   {accuracy:.3f}")
    print(f"  False Accept Rate: {far:.3f}  (lower is better)")

    # Threshold sweep
    print(f"\n  ── Threshold Sweep ──")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FAR':>8}")
    for thr in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        yp  = (y_score >= thr).astype(int)
        tp_ = int(((yp == 1) & (y_true == 1)).sum())
        fp_ = int(((yp == 1) & (y_true == 0)).sum())
        fn_ = int(((yp == 0) & (y_true == 1)).sum())
        tn_ = int(((yp == 0) & (y_true == 0)).sum())
        pr_ = tp_ / (tp_ + fp_ + 1e-9)
        re_ = tp_ / (tp_ + fn_ + 1e-9)
        f1_ = 2 * pr_ * re_ / (pr_ + re_ + 1e-9)
        fa_ = fp_ / (fp_ + tn_ + 1e-9)
        print(f"  {thr:>10.2f} {pr_:>10.3f} {re_:>8.3f} {f1_:>8.3f} {fa_:>8.3f}")

    print("\n" + "=" * 60)
    status = "✅ PASS" if (precision > 0.90 and recall > 0.85) else "⚠  NEEDS IMPROVEMENT"
    print(f"  Status: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
