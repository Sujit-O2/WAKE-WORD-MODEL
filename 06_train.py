"""
06_train.py
===========
Trains the Zerotwo wake word model (ResNet-SE, 3-channel).
Uses model.py for architecture -- no duplicated model code.

Usage:
  python 06_train.py
  python 06_train.py --epochs 100 --batch 64
"""

import sys, os, random, argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import librosa
from model import WakeWordResNetSE, N_MELS, N_FRAMES, IN_CHANNELS

PROJECT_DIR = Path(__file__).parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODELS_DIR = PROJECT_DIR / "models"
SAMPLE_RATE = 16000
HOP_LENGTH = 160
WIN_LENGTH = 400
N_FFT = 512

DEFAULTS = dict(epochs=80, batch=32, lr=3e-3, wd=1e-4, mixup=0.4,
                smooth=0.1, warmup=5, noise_prob=0.85, snr_range=(-5,30),
                focal_gamma=2.0, neg_noise_prob=0.5, neg_snr_range=(5,30))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")


class SpecAugment:
    def __init__(self, f_param=10, t_param=20, n_f=2, n_t=2, p=0.8):
        self.f_param, self.t_param = f_param, t_param
        self.n_f, self.n_t, self.p = n_f, n_t, p

    def __call__(self, mel):
        if random.random() > self.p:
            return mel
        m = mel.copy()
        H, W = m.shape
        for _ in range(self.n_f):
            f = random.randint(1, min(self.f_param, H-1))
            f0 = random.randint(0, max(0, H-f))
            m[f0:f0+f, :] = 0.0
        for _ in range(self.n_t):
            t = random.randint(1, min(self.t_param, W-1))
            t0 = random.randint(0, max(0, W-t))
            m[:, t0:t0+t] = 0.0
        return m


def extract_features(path, target_frames=N_FRAMES, noise_audio=None, snr_db=None):
    try:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        return None
    tgt = target_frames * HOP_LENGTH
    if len(audio) >= tgt:
        s = random.randint(0, len(audio) - tgt)
        audio = audio[s:s+tgt]
    else:
        audio = np.pad(audio, (0, tgt - len(audio)))
    if noise_audio is not None and snr_db is not None and len(noise_audio) > 0:
        if len(noise_audio) < tgt:
            nc = np.tile(noise_audio, tgt // len(noise_audio) + 1)[:tgt]
        else:
            s = random.randint(0, len(noise_audio) - tgt)
            nc = noise_audio[s:s+tgt]
        sp = np.mean(audio**2) + 1e-10
        np_ = np.mean(nc**2) + 1e-10
        sl = 10.0**(snr_db/10.0)
        audio = audio + nc * np.sqrt(sp / (np_ * sl))
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    mel_db = librosa.power_to_db(mel + 1e-9, ref=np.max)
    delta = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)
    feat = np.stack([mel_db, delta, delta2], axis=0)
    for c in range(3):
        feat[c] = (feat[c] - feat[c].mean()) / (feat[c].std() + 1e-9)
    return feat[:, :target_frames].astype(np.float32)


class FocalLoss(nn.Module):
    """Focal loss — focuses training on hard examples, reduces false positives."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class WakeWordDataset(Dataset):
    def __init__(self, files, labels, bg_dir, augment=False,
                 noise_prob=0.85, snr_range=(-5, 30),
                 neg_noise_prob=0.5, neg_snr_range=(5, 30)):
        self.files, self.labels = files, labels
        self.augment = augment
        self.noise_prob, self.snr_range = noise_prob, snr_range
        self.neg_noise_prob, self.neg_snr_range = neg_noise_prob, neg_snr_range
        self.spec_aug = SpecAugment() if augment else None
        self.bg_audios = []
        if augment and bg_dir.exists():
            for f in sorted(bg_dir.glob("*.wav"))[:600]:
                try:
                    a, _ = librosa.load(str(f), sr=SAMPLE_RATE, mono=True)
                    if len(a) > 500:
                        self.bg_audios.append(a)
                except:
                    pass
            print(f"  Loaded {len(self.bg_audios)} noise files for online mixing")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx], self.labels[idx]
        na, snr = None, None
        if self.augment and self.bg_audios:
            if label == 1 and random.random() < self.noise_prob:
                # Positive: wide SNR range (-5dB very noisy to 30dB clean)
                na = random.choice(self.bg_audios)
                snr = random.uniform(*self.snr_range)
            elif label == 0 and random.random() < self.neg_noise_prob:
                # Negative: also mix with noise to prevent false triggers in noisy envs
                na = random.choice(self.bg_audios)
                snr = random.uniform(*self.neg_snr_range)
        feat = extract_features(path, noise_audio=na, snr_db=snr)
        if feat is None:
            feat = np.zeros((3, N_MELS, N_FRAMES), dtype=np.float32)
        if self.spec_aug:
            feat[0] = self.spec_aug(feat[0])
        if self.augment and random.random() < 0.5:
            feat = feat * random.uniform(0.7, 1.3)
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def load_dataset():
    files, labels = [], []
    for d, lbl in [("positive",1), ("positive_mixed",1), ("negative",0), ("background",0)]:
        dp = DATASET_DIR / d
        if dp.exists():
            for f in sorted(dp.glob("*.wav")):
                files.append(str(f)); labels.append(lbl)
    n_pos, n_neg = labels.count(1), labels.count(0)
    print(f"\n  Dataset: {n_pos} pos + {n_neg} neg = {len(files)} total")
    combined = list(zip(files, labels))
    random.shuffle(combined)
    return [x[0] for x in combined], [x[1] for x in combined], DATASET_DIR/"background"


def stratified_split(files, labels, val=0.15):
    pos = [(f,l) for f,l in zip(files,labels) if l==1]
    neg = [(f,l) for f,l in zip(files,labels) if l==0]
    random.shuffle(pos); random.shuffle(neg)
    npv, nnv = max(1,int(len(pos)*val)), max(1,int(len(neg)*val))
    va = pos[:npv]+neg[:nnv]; tr = pos[npv:]+neg[nnv:]
    random.shuffle(tr); random.shuffle(va)
    tf, tl = zip(*tr) if tr else ([],[])
    vf, vl = zip(*va) if va else ([],[])
    return (list(tf),list(tl)), (list(vf),list(vl))


def mixup(x, y, alpha=0.4):
    if alpha <= 0: return x,y,y,1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam


def train_one_epoch(model, loader, optimizer, criterion, use_mixup=True):
    model.train()
    tl, correct, total = 0, 0, 0
    for feat, lbl in loader:
        feat, lbl = feat.to(DEVICE), lbl.to(DEVICE)
        optimizer.zero_grad()
        if use_mixup and random.random() < 0.5:
            mx, ya, yb, lam = mixup(feat, lbl)
            out = model(mx)
            loss = lam*criterion(out,ya) + (1-lam)*criterion(out,yb)
        else:
            out = model(feat)
            loss = criterion(out, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        tl += loss.item()*len(lbl); correct += (out.argmax(1)==lbl).sum().item(); total += len(lbl)
    return tl/total, correct/total


def eval_one_epoch(model, loader, criterion):
    model.eval()
    tl, correct, total = 0, 0, 0
    with torch.no_grad():
        for feat, lbl in loader:
            feat, lbl = feat.to(DEVICE), lbl.to(DEVICE)
            out = model(feat); loss = criterion(out, lbl)
            tl += loss.item()*len(lbl); correct += (out.argmax(1)==lbl).sum().item(); total += len(lbl)
    return tl/total, correct/total


def compute_f1(model, loader):
    model.eval()
    ap, al = [], []
    with torch.no_grad():
        for feat, lbl in loader:
            out = model(feat.to(DEVICE))
            ap.extend(out.argmax(1).cpu().numpy()); al.extend(lbl.numpy())
    ap, al = np.array(ap), np.array(al)
    tp = int(((ap==1)&(al==1)).sum()); fp = int(((ap==1)&(al==0)).sum()); fn = int(((ap==0)&(al==1)).sum())
    p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9)
    return 2*p*r/(p+r+1e-9), p, r


def export_onnx(model, path):
    model.eval()
    torch.onnx.export(model, torch.zeros(1,3,N_MELS,N_FRAMES), str(path),
        input_names=["melspectrogram"], output_names=["logits"],
        dynamic_axes={"melspectrogram":{0:"batch"},"logits":{0:"batch"}}, opset_version=11)
    print(f"  ONNX saved: {path} ({path.stat().st_size//1024} KB)")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    pa.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    pa.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    pa.add_argument("--warmup", type=int, default=DEFAULTS["warmup"])
    a = pa.parse_args()

    print("="*60)
    print("  ZEROTWO TRAINER (ResNet-SE, 3-ch, model.py)")
    print(f"  Device:{DEVICE} | Epochs:{a.epochs} | Batch:{a.batch}")
    print("="*60)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading data...")
    files, labels, bg_dir = load_dataset()
    if len(files) < 100:
        print("  Run data generation scripts first!"); return
    (trf,trl),(vaf,val) = stratified_split(files, labels)
    print(f"  Train:{len(trf)} | Val:{len(vaf)}")

    trds = WakeWordDataset(trf, trl, bg_dir, augment=True,
        noise_prob=DEFAULTS["noise_prob"], snr_range=DEFAULTS["snr_range"],
        neg_noise_prob=DEFAULTS["neg_noise_prob"], neg_snr_range=DEFAULTS["neg_snr_range"])
    vds = WakeWordDataset(vaf, val, bg_dir, augment=False)
    trl_loader = DataLoader(trds, batch_size=a.batch, shuffle=True, num_workers=0)
    va_loader = DataLoader(vds, batch_size=a.batch, shuffle=False, num_workers=0)

    nw = trl.count(1); nn_ = trl.count(0)
    cw = torch.tensor([1.0, nn_/max(nw,1)], dtype=torch.float32).to(DEVICE)

    print("\n[2] Building model...")
    model = WakeWordResNetSE().to(DEVICE)
    np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {np_:,}")

    criterion = FocalLoss(alpha=cw, gamma=DEFAULTS["focal_gamma"])
    optimizer = optim.AdamW(model.parameters(), lr=a.lr, weight_decay=DEFAULTS["wd"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    print(f"\n[3] Training ({a.epochs} epochs, {a.warmup} warmup)...")
    best_f1 = 0.0; best_acc = 0.0; no_imp = 0; patience = 15
    best_path = MODELS_DIR / "zerotwo_best.pt"

    for ep in range(1, a.epochs+1):
        if ep <= a.warmup:
            for pg in optimizer.param_groups:
                pg['lr'] = a.lr * ep / a.warmup
        else:
            scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        tl, ta = train_one_epoch(model, trl_loader, optimizer, criterion)
        vl, va = eval_one_epoch(model, va_loader, criterion)
        f1, prec, rec = compute_f1(model, va_loader)

        star = " " 
        if f1 > best_f1:
            best_f1, best_acc, no_imp = f1, va, 0
            torch.save(model.state_dict(), best_path)
            star = "*"
        else:
            no_imp += 1

        print(f"  {star} Ep{ep:2d}/{a.epochs} lr={lr:.2e} trL={tl:.4f} trA={ta:.3f} vlL={vl:.4f} vlA={va:.3f} f1={f1:.3f} p={prec:.3f} r={rec:.3f}")

        if no_imp >= patience:
            print(f"\n  Early stop at epoch {ep}"); break

    print("\n[4] Exporting ONNX...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    onnx_path = MODELS_DIR / "zerotwo_v2.onnx"
    export_onnx(model, onnx_path)

    print("\n"+"="*60)
    print(f"  DONE! Best F1: {best_f1:.3f} | Best Acc: {best_acc:.1%}")
    print(f"  Model: {onnx_path}")
    print("="*60)


if __name__ == "__main__":
    main()
