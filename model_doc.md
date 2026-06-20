# zerotwo_v2.onnx — Model Technical Documentation

## Identity
| Property | Value |
|----------|-------|
| File | `models/zerotwo_v2.onnx` |
| Size | ~3.6 MB (float32) |
| IR Version | 6 |
| Opset | 11 |
| Trainable parameters | ~910K |
| Architecture | ResNet + Squeeze-and-Excitation Attention |

## Architecture
Deep CNN binary classifier with residual blocks and channel attention, trained on a mel-spectrogram frontend with noise-robust augmentation.

```
Input: melspectrogram [1, 1, 80, 100]
  ↓ Stem: Conv2D(1→16, 3×3)
  ↓ Stage 1: 2× ResBlock(16) + MaxPool(2)
  ↓ Stage 2: 2× ResBlock(32) + MaxPool(2)
  ↓ Stage 3: 2× ResBlock(64) + MaxPool(2)
  ↓ Stage 4: 2× ResBlock(128)
  ↓ GlobalAvgPool → [1, 128]
  ↓ FC(128→128) → ReLU → Dropout(0.4) → FC(128→2)
  ↓ softmax(logits)[1] = P(wake="Zerotwo")
```

- **Class 0** = background / non-wake
- **Class 1** = "Zerotwo" detected

### ResBlock Detail
Each ResBlock contains:
- Two 3×3 Conv2D + BatchNorm + ReLU layers
- Dropout2D(0.1) for regularization
- Squeeze-and-Excitation (SE) attention: learns channel-wise importance weights
- Residual shortcut connection (identity or 1×1 projection)

### Squeeze-and-Excitation (SE) Block
- Global Average Pool → FC(channels → channels/4) → ReLU → FC(channels/4 → channels) → Sigmoid
- Recalibrates channel features: amplifies important frequency bands, suppresses noise

## Input Specification

| Field | Value |
|-------|-------|
| Name | `melspectrogram` |
| Shape | `[1, 1, 80, 100]` |
| Type | float32 |
| Layout | `[batch=1, channels=1, mel_bins=80, time_frames=100]` |

### How to compute the input tensor

```
Audio: 16000 Hz, mono, float32 (16-bit PCM / 32768)
Window: 1.0 seconds = 16,000 samples
n_fft: 512
hop_length: 160 → 100 frames/sec
n_mels: 80
target_frames: 100  (= 1.0 sec of audio at 100 fps)
```

**Steps:**
1. Capture 16 kHz mono PCM audio
2. Normalize: divide by 32768.0 to get [-1, 1] float range
3. Compute mel-spectrogram:
   - n_fft = 512, hop_length = 160, n_mels = 80
   - Hann window, center=True
   - fmin=0 Hz, fmax=8000 Hz (sr/2)
4. Convert to dB: `10 * log10(max(S, 1e-10))`  ref = max value
5. Normalize to [-1, 1]: `(S_db - mean) / std`
6. Take last 100 frames (sliding window)
7. Reshape to `[1, 1, 80, 100]` float32

## Output Specification

| Field | Value |
|-------|-------|
| Name | `logits` |
| Shape | `[1, 2]` |
| Type | float32 |

**Usage:**
```dart
// Apply softmax manually (model outputs raw logits)
final exp0 = math.exp(logits[0]);
final exp1 = math.exp(logits[1]);
final wakeProb = exp1 / (exp0 + exp1);
```

## Thresholding Strategy

| Tier | Condition | Action |
|------|-----------|--------|
| Fast-Pass | `wakeProb >= 0.90` | Immediate trigger |
| Confirmation | `wakeProb >= 0.65` | Add to 5-frame window, trigger on 3/5 |
| Reject | `wakeProb < 0.65` | Clear confirmation buffer |

## Training Data
- ~9,000 Piper TTS synthetic "Zerotwo" samples (6 voices)
- ~270 augmented real recordings (27 M4A × 10 augmentations)
- ~1,500 negative speech samples (hard negatives + random speech)
- ~1,500 background noise files (white, pink, brown, fan, rain, traffic, etc.)
- **Online noise mixing**: positive samples randomly mixed with background noise at SNR 0-25 dB
- **SpecAugment**: random frequency + time masking during training
- **Mixup augmentation**: blended sample pairs for regularization

## Key Improvements From v1
| | v1 (`zerotwo_v1.onnx`) | v2 (`zerotwo_v2.onnx`) |
|-|------------------------|------------------------|
| Architecture | 3-layer CNN (shallow) | 8 ResBlocks + SE attention (deep) |
| Parameters | ~617K (~2.4MB) | ~910K (~3.6MB) |
| Attention | None | Squeeze-and-Excitation |
| Residuals | None | Residual connections |
| Pooling | AdaptiveAvgPool(4,4) | Global Average Pool |
| Input shape | `[1, 1, 80, 100]` | `[1, 1, 80, 100]` |
| Epochs | 15 | 50 with warmup |
| Noise robustness | Low (clean training data) | High (online noise mixing) |
| Augmentation | Offline only | SpecAugment + Mixup + NoiseMix |
| Loss | CrossEntropy | CrossEntropy + Label Smoothing |
| Optimizer | Adam | AdamW |
| Scheduler | CosineAnnealing | CosineAnnealingWarmRestarts |
| N_FRAMES | 96-100 (inconsistent) | 100 (consistent) |
