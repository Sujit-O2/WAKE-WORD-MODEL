"""
model.py
========
Single source of truth for the Zerotwo wake word model architecture.
All scripts (train, evaluate, export, realtime) import from here.

Architecture: ResNet-SE (Squeeze-and-Excitation Residual Network)
  - 3-channel input: mel + delta + delta-delta
  - 8 residual blocks with SE attention
  - ~1.2M parameters (lightweight for mobile)
  - Input shape: (batch, 3, 80, 100)
"""

import torch
import torch.nn as nn

N_MELS = 80
N_FRAMES = 100
IN_CHANNELS = 3


class SqueezeExcitation(nn.Module):
    """Channel attention -- learns which frequency channels matter."""
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
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SqueezeExcitation(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(self.bn2(self.conv2(out)))
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
      Stage 1: 2x ResBlockSE(32->64) + MaxPool
      Stage 2: 2x ResBlockSE(64->128) + MaxPool
      Stage 3: 2x ResBlockSE(128->256)
      GlobalAvgPool -> FC(256->128) -> Dropout -> FC(128->2)

    Total params: ~1.2M
    """
    def __init__(self, in_channels=IN_CHANNELS, num_classes=2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.stage1 = nn.Sequential(
            ResBlockSE(32, 64, stride=2),
            ResBlockSE(64, 64),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.stage2 = nn.Sequential(
            ResBlockSE(64, 128, stride=2),
            ResBlockSE(128, 128),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.stage3 = nn.Sequential(
            ResBlockSE(128, 256, stride=2),
            ResBlockSE(256, 256),
        )

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
        x = self.pool1(x)          # (B, 64, 10, 12)
        x = self.stage2(x)         # (B, 128, 5, 6)
        x = self.pool2(x)          # (B, 128, 2, 3)
        x = self.stage3(x)         # (B, 256, 1, 1)
        x = self.global_pool(x)    # (B, 256, 1, 1)
        x = self.classifier(x)     # (B, 2)
        return x


def load_model(weights_path: str = None, device: str = "cpu"):
    """Load model, optionally with pre-trained weights."""
    model = WakeWordResNetSE()
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)
