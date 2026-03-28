# zerotwo_v1.onnx — Model Technical Documentation

## Identity
| Property | Value |
|----------|-------|
| File | `assets/wakeword/zerotwo_v1.onnx` |
| Size | 2.4 MB |
| IR Version | 6 |
| Opset | 11 |
| Graph nodes | 13 |
| Trainable parameters | 10 initializers |

## Architecture
Lightweight CNN binary classifier trained on top of a mel-spectrogram frontend.

```
Input: melspectrogram [1, 1, 80, 96]
  ↓ Conv2D layers (13 graph nodes)
Output: logits [1, 2]
  ↓ softmax(logits)[1] = P(wake="Zerotwo")
```

- **Class 0** = background / non-wake
- **Class 1** = "Zerotwo" detected

## Input Specification

| Field | Value |
|-------|-------|
| Name | `melspectrogram` |
| Shape | `[1, 1, 80, 96]` |
| Type | float32 |
| Layout | `[batch=1, channels=1, mel_bins=80, time_frames=96]` |

### How to compute the input tensor

```
Audio: 16000 Hz, mono, float32 (16-bit PCM / 32768)
Window: 1.5 seconds = 24,000 samples
n_fft: 512
hop_length: 160 → 100 frames/sec
n_mels: 80
target_frames: 96  (= 0.96 sec of audio at 100 fps)
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
6. Take last 96 frames (sliding window)
7. Reshape to `[1, 1, 80, 96]` float32

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
- ~9,000 Piper TTS synthetic "Zerotwo" samples (3 voices)
- ~270 augmented real recordings (27 M4A × 10 augmentations)
- ~1,500 LibriSpeech negative samples
- ~500 hard negatives (phonetically similar words)

## Key Differences From Previous Model
| | Old (`production_zero_two.onnx`) | New (`zerotwo_v1.onnx`) |
|-|-------------------------------|------------------------|
| Input name | `input` | `melspectrogram` |
| Input shape | `[1, 40, 32, 1]` (MFCC) | `[1, 1, 80, 96]` (Mel-spec) |
| Output name | `Identity:0` / `dense_7` | `logits` |
| Output shape | `[1, N]` | `[1, 2]` |
| Feature type | MFCCs (40 bins) | Mel-spectrogram (80 bins) |
| Time frames | 32 frames | 96 frames |
