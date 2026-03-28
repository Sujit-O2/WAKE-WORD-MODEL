"""
06_train.py
===========
Trains a CNN wake word detector on mel-spectrograms.

Architecture:
  Input: 80 mel-bins × 98 frames (1 sec at 16kHz, hop=160)
  → Conv2D(32) → BN → ReLU → MaxPool
  → Conv2D(64) → BN → ReLU → MaxPool
  → Conv2D(128) → BN → ReLU → AdaptiveAvgPool
  → FC(256) → Dropout → FC(2) → Softmax

Labels: 1 = "Zerotwo", 0 = not wake word
Output: zerotwo_wake/models/zerotwo_v1.onnx
"""

import sys
import os
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("Installing PyTorch...")
    os.system(f"{sys.executable} -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu -q")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

try:
    import librosa
except ImportError:
    os.system(f"{sys.executable} -m pip install librosa -q")
    import librosa

# ─────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent
DATASET_DIR  = PROJECT_DIR / "dataset"
MODELS_DIR   = PROJECT_DIR / "models"
SAMPLE_RATE  = 16000
N_MELS       = 80
HOP_LENGTH   = 160     # 10ms
WIN_LENGTH   = 400     # 25ms
N_FFT        = 512
DURATION_SEC = 1.0     # fixed input length
N_FRAMES     = int(DURATION_SEC * SAMPLE_RATE / HOP_LENGTH)   # ~100 frames

EPOCHS       = 15
BATCH_SIZE   = 32
LEARNING_RATE = 1e-3
VAL_SPLIT    = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")
# ─────────────────────────────────────────────────────────────────


def extract_melspec(audio_path: str, target_frames: int = N_FRAMES) -> np.ndarray | None:
    """Load audio and extract mel-spectrogram."""
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        return None

    # Pad or trim to fixed length
    target_len = target_frames * HOP_LENGTH
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        # Random crop during training for augmentation
        start = random.randint(0, len(audio) - target_len)
        audio = audio[start: start + target_len]

    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
    )
    mel_db = librosa.power_to_db(mel + 1e-9, ref=np.max)

    # Normalize to [-1, 1]
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    mel_db = mel_db[:, :target_frames]  # ensure exact frame count

    return mel_db.astype(np.float32)  # shape: (N_MELS, N_FRAMES)


class WakeWordDataset(Dataset):
    def __init__(self, file_list: list, labels: list, augment: bool = False):
        self.file_list = file_list
        self.labels    = labels
        self.augment   = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path  = self.file_list[idx]
        label = self.labels[idx]

        mel = extract_melspec(path)
        if mel is None:
            mel = np.zeros((N_MELS, N_FRAMES), dtype=np.float32)

        # Add channel dim: (1, N_MELS, N_FRAMES)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return mel_tensor, label_tensor


class WakeWordCNN(nn.Module):
    def __init__(self, n_mels: int = N_MELS, n_frames: int = N_FRAMES):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)


def load_dataset():
    """Load all wav files with labels."""
    pos_dir = DATASET_DIR / "positive"
    neg_dir = DATASET_DIR / "negative"
    bg_dir  = DATASET_DIR / "background"

    files, labels = [], []

    # Positive samples (label = 1)
    for f in sorted(pos_dir.glob("*.wav")):
        files.append(str(f))
        labels.append(1)

    # Negative samples (label = 0)
    for f in sorted(neg_dir.glob("*.wav")):
        files.append(str(f))
        labels.append(0)

    # Background noise as negative (label = 0)
    for f in sorted(bg_dir.glob("*.wav")):
        files.append(str(f))
        labels.append(0)

    print(f"\n  Dataset: {labels.count(1)} positive + {labels.count(0)} negative")

    # Shuffle
    combined = list(zip(files, labels))
    random.shuffle(combined)
    files, labels = zip(*combined)
    return list(files), list(labels)


def split_dataset(files, labels, val_split=VAL_SPLIT):
    n     = len(files)
    split = int(n * (1 - val_split))
    return (files[:split], labels[:split]), (files[split:], labels[split:])


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for mel, label in loader:
        mel, label = mel.to(device), label.to(device)
        optimizer.zero_grad()
        out  = model(mel)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(label)
        correct    += (out.argmax(1) == label).sum().item()
        total      += len(label)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for mel, label in loader:
            mel, label = mel.to(device), label.to(device)
            out  = model(mel)
            loss = criterion(out, label)
            total_loss += loss.item() * len(label)
            correct    += (out.argmax(1) == label).sum().item()
            total      += len(label)
    return total_loss / total, correct / total


def export_onnx(model, save_path: Path):
    """Export model to ONNX format."""
    model.eval()
    dummy = torch.zeros(1, 1, N_MELS, N_FRAMES)
    torch.onnx.export(
        model, dummy, str(save_path),
        input_names=["melspectrogram"],
        output_names=["logits"],
        dynamic_axes={
            "melspectrogram": {0: "batch"},
            "logits":         {0: "batch"},
        },
        opset_version=11,
    )
    print(f"\n  ✓ ONNX model saved: {save_path}")


def main():
    print("=" * 60)
    print("  ZEROTWO WAKE WORD MODEL TRAINER")
    print(f"  Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────
    print("\n[1] Loading dataset...")
    files, labels = load_dataset()
    if len(files) < 100:
        print("⚠  Very few files found. Run data generation scripts first.")
        return

    (tr_f, tr_l), (va_f, va_l) = split_dataset(files, labels)
    print(f"  Train: {len(tr_f)} | Val: {len(va_f)}")

    tr_ds = WakeWordDataset(tr_f, tr_l, augment=True)
    va_ds = WakeWordDataset(va_f, va_l, augment=False)

    # Class weights for imbalanced data
    n_pos = tr_l.count(1)
    n_neg = tr_l.count(0)
    pos_weight = (n_neg / max(n_pos, 1))
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(DEVICE)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────
    print("\n[2] Building model...")
    model     = WakeWordCNN().to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ─────────────────────────────────────────
    print("\n[3] Training...")
    best_val_acc = 0.0
    best_model_path = MODELS_DIR / "zerotwo_best.pt"

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, criterion, DEVICE)
        va_loss, va_acc = eval_epoch(model, va_loader, criterion, DEVICE)
        scheduler.step()

        is_best = va_acc > best_val_acc
        if is_best:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_model_path)

        star = "★" if is_best else " "
        print(f"  {star} Epoch {epoch:2d}/{EPOCHS} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.3f}")

    # ── Export ────────────────────────────────────────────────
    print("\n[4] Exporting best model to ONNX...")
    model.load_state_dict(torch.load(best_model_path))
    onnx_path = MODELS_DIR / "zerotwo_v1.onnx"
    export_onnx(model, onnx_path)

    print("\n" + "=" * 60)
    print(f"  ✅ Training complete!")
    print(f"  Best val accuracy: {best_val_acc:.1%}")
    print(f"  Model: {onnx_path}")
    print("=" * 60)
    print("\n  Next: Run 07_evaluate.py to test the model")


if __name__ == "__main__":
    main()
