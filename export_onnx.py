"""
export_onnx.py
==============
Exports trained zerotwo_best.pt to zerotwo_v2.onnx using model.py.
"""
import sys, os
from pathlib import Path
import torch
from model import WakeWordResNetSE, N_MELS, N_FRAMES, IN_CHANNELS

PROJECT_DIR = Path(__file__).parent
MODEL_PT = PROJECT_DIR / "models" / "zerotwo_best.pt"
ONNX_PATH = PROJECT_DIR / "models" / "zerotwo_v2.onnx"

print("="*55)
print("  ZEROTWO MODEL -> ONNX EXPORT")
print("="*55)

if not MODEL_PT.exists():
    print(f"  Checkpoint not found: {MODEL_PT}")
    print("  Run 06_train.py first."); sys.exit(1)

model = WakeWordResNetSE()
model.load_state_dict(torch.load(MODEL_PT, map_location="cpu"))
model.eval()
np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Loaded: {MODEL_PT.name} | Params: {np_:,}")

dummy = torch.zeros(1, IN_CHANNELS, N_MELS, N_FRAMES)
try:
    torch.onnx.export(model, dummy, str(ONNX_PATH),
        input_names=["melspectrogram"], output_names=["logits"],
        dynamic_axes={"melspectrogram":{0:"batch"},"logits":{0:"batch"}},
        opset_version=11)
    print(f"  Exported: {ONNX_PATH.name} ({ONNX_PATH.stat().st_size//1024} KB)")
except Exception as e:
    print(f"  Export failed: {e}")
    try:
        out = torch.onnx.dynamo_export(model, dummy)
        out.save(str(ONNX_PATH))
        print(f"  Exported (dynamo): {ONNX_PATH.name}")
    except Exception as e2:
        print(f"  Dynamo failed: {e2}")
        ts = ONNX_PATH.with_suffix(".torchscript.pt")
        torch.jit.save(torch.jit.trace(model, dummy), str(ts))
        print(f"  Saved TorchScript: {ts.name}")

print("\n" + "="*55)
print("  Export complete!")
print("="*55)
