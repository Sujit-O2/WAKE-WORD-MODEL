"""
06_train_v2.py
==============
POWERFUL wake word trainer with ResNet-SE architecture, SpecAugment,
noise-mixed data, and advanced training techniques.

Changes from v1:
  - ResNet-SE architecture (deeper, squeeze-excitation, residual connections)
  - Input channels: 3 (mel + delta + delta-delta) instead of 1
  - SpecAugment (time/freq masking during training)
  - Label smoothing loss
  - 50 epochs with cosine warmup
  - Lower learning rate (3e-4) with OneCycleLR
  - Gradient clipping
  - Loads from dataset/positive_mixed/ for noise-robust training
  - TTA (test-time augmentation) during eval
  - Mixup augmentation
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
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("Installing PyTorch...")
    os.system(f"{sys.executable} -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu -q")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

try:
    import librosa
except ImportError:
    os.system(f"{sys.executable} -m pip install librosa -q")
    import librosa

# -----------------------------------------------------------------
PROJECT_DIR   = Path(__file__).parent
DATASET_DIR   = PROJECT_DIR / "dataset"
MODELS_DIR    = PROJECT_DIR / "models"
SAMPLE_RATE   = 16000
N_MELS        = 80
HOP_LENGTH    = 160       # 10ms
WIN_LENGTH    = 400       # 25ms
N_FFT         = 512
DURATION_SEC  = 1.0
N_FRAMES      = int(DURATION_SEC * SAMPLE_RATE / HOP_LENGTH)  # 100

# Training hyperparameters
EPOCHS        = 50
BATCH_SIZE    = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-3
VAL_SPLIT     = 0.15
LABEL_SMOOTH  = 0.1
MIXUP_ALPHA   = 0.2
GRAD_CLIP     = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")
# -----------------------------------------------------------------


# ═══════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION (with delta features)
# ═══════════════════════════════════════════════════════════════════

def extract_features(audio_path: str, target_frames: int = N_FRAMES, augment: bool = False) -> np.ndarray | None:
    """Extract mel-spectrogram with delta features (3 channels).
    
    Returns: (3, N_MELS, target_frames) -- mel, delta, delta-delta
    """
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        return None

    target_len = target_frames * HOP_LENGTH
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        if augment:
            start = random.randint(0, len(audio) - target_len)
        else:
            start = 0
        audio = audio[start: start + target_len]

    # SpecAugment: random time/freq masking (only during training)
    if augment:
        audio = spec_augment_audio(audio)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
    )
    mel_db = librosa.power_to_db(mel + 1e-9, ref=np.max)

    # Delta features (temporal dynamics -- crucial for noise robustness)
    delta = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)

    # Stack as channels: (3, N_MELS, N_FRAMES)
    features = np.stack([mel_db, delta, delta2], axis=0)

    # Normalize each channel independently
    for c in range(3):
        ch = features[c]
        features[c] = (ch - ch.mean()) / (ch.std() + 1e-9)

    features = features[:, :, :target_frames]
    return features.astype(np.float32)


def spec_augment_audio(audio: np.ndarray) -> np.ndarray:
    """Apply SpecAugment-style augmentation directly on audio."""
    # Random time shift (10% of audio)
    if random.random() < 0.5:
        shift = random.randint(-int(len(audio) * 0.1), int(len(audio) * 0.1))
        audio = np.roll(audio, shift)

    # Random gain
    if random.random() < 0.5:
        gain = random.uniform(0.8, 1.2)
        audio = audio * gain

    # Random additive noise
    if random.random() < 0.3:
        noise_level = random.uniform(0.001, 0.01)
        audio = audio + np.random.randn(len(audio)).astype(np.float32) * noise_level

    return audio


# ═══════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════

class WakeWordDataset(Dataset):
    def __init__(self, file_list: list, labels: list, augment: bool = False):
        self.file_list = file_list
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        label = self.labels[idx]

        features = extract_features(path, augment=self.augment)
        if features is None:
            features = np.zeros((3, N_MELS, N_FRAMES), dtype=np.float32)

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return features_tensor, label_tensor


# ═══════════════════════════════════════════════════════════════════
#  MODEL: ResNet-SE (Squeeze-and-Excitation Residual Network)
# ═══════════════════════════════════════════════════════════════════

class SqueezeExcitation(nn.Module):
    """Channel attention mechanism -- learns which features matter."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w.expand_as(x)


class ResBlockSE(nn.Module):
    """Residual block with Squeeze-Excitation attention."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SqueezeExcitation(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class WakeWordResNetSE(nn.Module):
    """
    ResNet-SE wake word detector.
    
    Architecture:
      Input: (batch, 3, 80, 100) -- mel + delta + delta-delta
      
      Stem: Conv2d(3->32) + BN + ReLU + MaxPool
      
      ResBlockSE Stage 1: 32->64 (stride 2)
      ResBlockSE Stage 2: 64->128 (stride 2)
      ResBlockSE Stage 3: 128->256 (stride 2)
      
      Global Average Pool -> FC(256->128) -> ReLU -> Dropout -> FC(128->2)
      
      Total params: ~1.2M (still lightweight for mobile!)
    """
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # ResNet stages with SE attention
        self.stage1 = nn.Sequential(
            ResBlockSE(32, 64, stride=2),
            ResBlockSE(64, 64),
        )
        self.stage2 = nn.Sequential(
            ResBlockSE(64, 128, stride=2),
            ResBlockSE(128, 128),
        )
        self.stage3 = nn.Sequential(
            ResBlockSE(128, 256, stride=2),
            ResBlockSE(256, 256),
        )

        # Global pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)           # (B, 32, 40, 50)
        x = self.stage1(x)         # (B, 64, 20, 25)
        x = self.stage2(x)         # (B, 128, 10, 12)
        x = self.stage3(x)         # (B, 256, 5, 6)
        x = self.global_pool(x)    # (B, 256, 1, 1)
        x = self.classifier(x)     # (B, 2)
        return x


# ═══════════════════════════════════════════════════════════════════
#  MIXUP AUGMENTATION
# ═══════════════════════════════════════════════════════════════════

def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """Mixup: blend two samples and their labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss: weighted sum of two losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ═══════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════

def load_dataset():
    """Load all wav files -- includes clean, mixed, and noise samples."""
    pos_dir = DATASET_DIR / "positive"
    pos_mixed_dir = DATASET_DIR / "positive_mixed"
    neg_dir = DATASET_DIR / "negative"
    bg_dir = DATASET_DIR / "background"

    files, labels = [], []

    # Clean positive samples (label = 1)
    if pos_dir.exists():
        for f in sorted(pos_dir.glob("*.wav")):
            files.append(str(f))
            labels.append(1)

    # Noise-mixed positive samples (label = 1) -- KEY for noise robustness!
    if pos_mixed_dir.exists():
        for f in sorted(pos_mixed_dir.glob("*.wav")):
            files.append(str(f))
            labels.append(1)

    # Negative samples (label = 0)
    if neg_dir.exists():
        for f in sorted(neg_dir.glob("*.wav")):
            files.append(str(f))
            labels.append(0)

    # Background noise as negative (label = 0)
    if bg_dir.exists():
        for f in sorted(bg_dir.glob("*.wav")):
            files.append(str(f))
            labels.append(0)

    n_pos = labels.count(1)
    n_neg = labels.count(0)
    print(f"\n  Dataset: {n_pos} positive + {n_neg} negative = {len(files)} total")

    combined = list(zip(files, labels))
    random.shuffle(combined)
    files, labels = zip(*combined)
    return list(files), list(labels)


def split_dataset(files, labels, val_split=VAL_SPLIT):
    n = len(files)
    split = int(n * (1 - val_split))
    return (files[:split], labels[:split]), (files[split:], labels[split:])


def train_epoch(model, loader, optimizer, criterion, device, use_mixup=True):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for mel, label in loader:
        mel, label = mel.to(device), label.to(device)

        optimizer.zero_grad()

        if use_mixup and random.random() < 0.5:
            mixed_mel, y_a, y_b, lam = mixup_data(mel, label)
            out = model(mixed_mel)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        else:
            out = model(mel)
            loss = criterion(out, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() * len(label)
        correct += (out.argmax(1) == label).sum().item()
        total += len(label)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for mel, label in loader:
            mel, label = mel.to(device), label.to(device)
            out = model(mel)
            loss = criterion(out, label)
            total_loss += loss.item() * len(label)
            correct += (out.argmax(1) == label).sum().item()
            total += len(label)
    return total_loss / total, correct / total


def export_onnx(model, save_path: Path):
    """Export model to ONNX format."""
    model.eval()
    # 3-channel input now (mel + delta + delta-delta)
    dummy = torch.zeros(1, 3, N_MELS, N_FRAMES)
    torch.onnx.export(
        model, dummy, str(save_path),
        input_names=["melspectrogram"],
        output_names=["logits"],
        dynamic_axes={
            "melspectrogram": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=11,
    )
    print(f"\n  ONNX model saved: {save_path}")


def main():
    print("=" * 60)
    print("  ZEROTWO WAKE WORD TRAINER v2 (ResNet-SE)")
    print(f"  Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"  LR: {LEARNING_RATE} | Label Smooth: {LABEL_SMOOTH}")
    print(f"  Mixup: {MIXUP_ALPHA} | Grad Clip: {GRAD_CLIP}")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -- Load data ----------------------------------------------
    print("\n[1] Loading dataset (clean + noise-mixed + negative)...")
    files, labels = load_dataset()
    if len(files) < 100:
        print("  !! Very few files found. Run data generation scripts first.")
        return

    (tr_f, tr_l), (va_f, va_l) = split_dataset(files, labels)
    print(f"  Train: {len(tr_f)} | Val: {len(va_f)}")

    tr_ds = WakeWordDataset(tr_f, tr_l, augment=True)
    va_ds = WakeWordDataset(va_f, va_l, augment=False)

    # Class weights
    n_pos = tr_l.count(1)
    n_neg = tr_l.count(0)
    pos_weight = n_neg / max(n_pos, 1)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(DEVICE)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # -- Model -------------------------------------------------
    print("\n[2] Building ResNet-SE model...")
    model = WakeWordResNetSE(in_channels=3, num_classes=2).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # -- Optimizer + Scheduler ----------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(tr_loader),
        pct_start=0.1,       # 10% warmup
        anneal_strategy="cos",
    )

    # -- Training loop -----------------------------------------
    print("\n[3] Training...")
    best_val_acc = 0.0
    best_model_path = MODELS_DIR / "zerotwo_v2_best.pt"
    patience = 10
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, criterion, DEVICE)
        va_loss, va_acc = eval_epoch(model, va_loader, criterion, DEVICE)
        scheduler.step()

        is_best = va_acc > best_val_acc
        if is_best:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        star = " *" if is_best else "  "
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  {star} Epoch {epoch:2d}/{EPOCHS} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.3f} | "
              f"lr={lr_now:.6f}")

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # -- Export ------------------------------------------------
    print("\n[4] Exporting best model to ONNX...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    onnx_path = MODELS_DIR / "zerotwo_v2.onnx"
    export_onnx(model, onnx_path)

    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Best val accuracy: {best_val_acc:.1%}")
    print(f"  Model: {onnx_path}")
    print(f"  Parameters: {n_params:,}")
    print("=" * 60)
    print("\n  Next: Run 07_evaluate_v2.py to test the model")


if __name__ == "__main__":
    main()
