import sys, io, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import torch
from model import WakeWordResNetSE

model = WakeWordResNetSE()
ckpt = torch.load("P:/WAKE-WORD-MODEL/models/zerotwo_best.pt", map_location="cpu", weights_only=False)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()

dummy = torch.zeros(1, 3, 80, 100)
torch.onnx.export(
    model, dummy, "P:/WAKE-WORD-MODEL/models/zerotwo_v2.onnx",
    opset_version=14,
    input_names=["melspectrogram"],
    output_names=["logits"],
    dynamic_axes={"melspectrogram": {0: "batch"}, "logits": {0: "batch"}}
)
print("Exported: models/zerotwo_v2.onnx")
if isinstance(ckpt, dict) and "best_val_acc" in ckpt:
    print("Best val acc:", ckpt["best_val_acc"])
    print("Best val f1:", ckpt["best_val_f1"])
    print("Epoch:", ckpt["epoch"])
else:
    print("Checkpoint is raw state_dict (no metrics saved)")
