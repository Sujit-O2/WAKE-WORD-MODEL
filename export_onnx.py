"""
export_onnx.py
==============
Exports the trained zerotwo_best.pt model to zerotwo_v1.onnx.
Handles both old and new PyTorch ONNX export paths.
"""
import sys
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
MODEL_PT    = PROJECT_DIR / "models" / "zerotwo_best.pt"
ONNX_PATH   = PROJECT_DIR / "models" / "zerotwo_v1.onnx"
N_MELS, N_FRAMES = 80, 100

# ── Install onnxscript if missing (required by PyTorch 2.x) ───────
try:
    import onnxscript
except ImportError:
    print("Installing onnxscript (required by PyTorch 2.x ONNX exporter)...")
    os.system(f"{sys.executable} -m pip install onnxscript -q")

import torch
import torch.nn as nn

# ── Recreate model architecture ───────────────────────────────────
class WakeWordCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )
    def forward(self, x):
        return self.classifier(self.conv3(self.conv2(self.conv1(x))))

print("=" * 55)
print("  ZEROTWO MODEL → ONNX EXPORTER")
print("=" * 55)

if not MODEL_PT.exists():
    print(f"✗ Checkpoint not found: {MODEL_PT}")
    sys.exit(1)

# Load weights
model = WakeWordCNN()
model.load_state_dict(torch.load(MODEL_PT, map_location="cpu"))
model.eval()
print(f"  ✓ Loaded: {MODEL_PT.name}")

dummy = torch.zeros(1, 1, N_MELS, N_FRAMES)

# Try export — Method 1: standard torch.onnx.export
try:
    torch.onnx.export(
        model, dummy, str(ONNX_PATH),
        input_names=["melspectrogram"],
        output_names=["logits"],
        dynamic_axes={"melspectrogram": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=11,
    )
    print(f"  ✓ Exported (method 1): {ONNX_PATH.name}")
except Exception as e1:
    print(f"  Method 1 failed: {e1}")
    # Method 2: dynamo export (PyTorch 2.x)
    try:
        export_output = torch.onnx.dynamo_export(model, dummy)
        export_output.save(str(ONNX_PATH))
        print(f"  ✓ Exported (method 2/dynamo): {ONNX_PATH.name}")
    except Exception as e2:
        print(f"  Method 2 failed: {e2}")
        # Method 3: TorchScript fallback
        ts_path = ONNX_PATH.with_suffix(".torchscript.pt")
        scripted = torch.jit.trace(model, dummy)
        torch.jit.save(scripted, str(ts_path))
        print(f"  ✓ Saved as TorchScript: {ts_path.name}")
        print("  (Use TorchScript on Android with PyTorch Mobile)")

if ONNX_PATH.exists():
    size_kb = ONNX_PATH.stat().st_size // 1024
    print(f"\n  📦 Model size: {size_kb} KB")
    print(f"  📁 Path: {ONNX_PATH}")

print("\n" + "=" * 55)
print("  ✅ Export complete! Model ready for Android.")
print("=" * 55)
print("\n  Next steps:")
print("  1. Copy models/zerotwo_v1.onnx to your Android app/assets/")
print("  2. Run: .\\wake_env\\Scripts\\python.exe 07_evaluate.py")
print("  3. Run: .\\wake_env\\Scripts\\python.exe 08_realtime.py")
