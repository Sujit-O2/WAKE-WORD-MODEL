"""
export_onnx_v2.py
=================
Exports the trained zerotwo_v2_best.pt model to zerotwo_v2.onnx.
Supports the new ResNet-SE architecture with 3-channel input.
"""

import sys
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
MODEL_PT = PROJECT_DIR / "models" / "zerotwo_v2_best.pt"
ONNX_PATH = PROJECT_DIR / "models" / "zerotwo_v2.onnx"
N_MELS, N_FRAMES = 80, 100

try:
    import onnxscript
except ImportError:
    print("Installing onnxscript...")
    os.system(f"{sys.executable} -m pip install onnxscript -q")

import torch
import torch.nn as nn


# -- Squeeze-Excitation ------------------------------------------
class SqueezeExcitation(nn.Module):
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
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.stage1 = nn.Sequential(ResBlockSE(32, 64, stride=2), ResBlockSE(64, 64))
        self.stage2 = nn.Sequential(ResBlockSE(64, 128, stride=2), ResBlockSE(128, 128))
        self.stage3 = nn.Sequential(ResBlockSE(128, 256, stride=2), ResBlockSE(256, 256))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


print("=" * 55)
print("  ZEROTWO MODEL v2 -> ONNX EXPORTER")
print("=" * 55)

if not MODEL_PT.exists():
    print(f"  Checkpoint not found: {MODEL_PT}")
    print("  Run 06_train_v2.py first.")
    sys.exit(1)

model = WakeWordResNetSE(in_channels=3, num_classes=2)
model.load_state_dict(torch.load(MODEL_PT, map_location="cpu"))
model.eval()
print(f"  Loaded: {MODEL_PT.name}")

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters: {n_params:,}")

dummy = torch.zeros(1, 3, N_MELS, N_FRAMES)

try:
    torch.onnx.export(
        model, dummy, str(ONNX_PATH),
        input_names=["melspectrogram"],
        output_names=["logits"],
        dynamic_axes={"melspectrogram": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=11,
    )
    print(f"  Exported: {ONNX_PATH.name}")
except Exception as e1:
    print(f"  Method 1 failed: {e1}")
    try:
        export_output = torch.onnx.dynamo_export(model, dummy)
        export_output.save(str(ONNX_PATH))
        print(f"  Exported (dynamo): {ONNX_PATH.name}")
    except Exception as e2:
        print(f"  Method 2 failed: {e2}")
        ts_path = ONNX_PATH.with_suffix(".torchscript.pt")
        scripted = torch.jit.trace(model, dummy)
        torch.jit.save(scripted, str(ts_path))
        print(f"  Saved as TorchScript: {ts_path.name}")

if ONNX_PATH.exists():
    size_kb = ONNX_PATH.stat().st_size // 1024
    print(f"\n  Model size: {size_kb} KB")
    print(f"  Path: {ONNX_PATH}")

print("\n" + "=" * 55)
print("  Export complete!")
print("=" * 55)
