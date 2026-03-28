# 🎙️ ZeroTwo Offline Wake Word System 🤖

![Wake Word Detection](https://img.shields.io/badge/Status-Fully%20Functional-brightgreen?style=for-the-badge)
![ONNX](https://img.shields.io/badge/Model-ONNX-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10-yellow?style=for-the-badge)
![Android](https://img.shields.io/badge/Platform-Android%20%7C%20Windows-green?style=for-the-badge)

> **"Zerotwo"** - A high-performance, fully offline wake word detection engine designed for privacy, speed, and reliability. Built with PyTorch and optimized for mobile deployment via ONNX Runtime.

---

## 📑 Table of Contents
1. [Overview](#-overview)
2. [Project Philosophy](#-project-philosophy)
3. [System Architecture](#-system-architecture)
4. [Performance Metrics](#-performance-metrics)
5. [Installation & Setup](#-installation--setup)
6. [Data Generation Pipeline](#-data-generation-pipeline)
7. [Training Methodology](#-training-methodology)
8. [Evaluation & Validation](#-evaluation--validation)
9. [Real-time Inference](#-real-time-inference)
10. [Android Integration](#-android-integration)
11. [Troubleshooting](#-troubleshooting)
12. [Future Roadmap](#-future-roadmap)
13. [License](#-license)

---

## 🔍 Overview

The ZeroTwo Wake Word System is a robust, end-to-end solution for detecting the specific phrase **"Zerotwo"** locally on edge devices. Unlike cloud-based solutions, this system processes audio streams entirely on-device, ensuring zero latency and absolute privacy.

This project covers the entire lifecycle of a wake word engine:
- Synthetic Data Generation (TTS)
- Real Recording Processing & Augmentation
- CNN Model Architecture Design
- High-Performance Training Pipeline
- ONNX Model Export & Quantization
- Real-time Micro-Inference Engine

---

## 🧠 Project Philosophy

### 1. Privacy First
In an era of constant connectivity, the most sensitive data—your voice—should never leave your device. ZeroTwo is built to run 100% offline.

### 2. Efficiency & Speed
Wake word detection must be "always-on" without draining battery. By using a lightweight CNN architecture and ONNX Runtime, we achieve sub-10ms inference times.

### 3. Robustness
A wake word engine is only as good as its false-trigger rate. We utilize a diverse dataset of 1500+ synthetic samples across 6 different voices, combined with background noise and "hard negatives" (words that sound similar but aren't the wake word).

---

## 🏗️ System Architecture

### Audio Pre-processing
The system transforms raw 1D audio into a 2D "image" for the CNN to process:
- **Sample Rate**: 16,000 Hz (Standard for speech)
- **Feature Extraction**: Mel-spectrogram
- **N_FFT**: 512
- **Hop Length**: 160 (10ms steps)
- **Window Length**: 400 (25ms windows)
- **N_Mels**: 80 frequency bins
- **Input Shape**: (1, 80, 96) — representing ~1 second of audio.

### CNN Model Structure
Our model uses a custom 3-layer Convolutional Neural Network:
1.  **Conv Block 1**: 32 filters, 3x3 kernel, BatchNorm, ReLU, MaxPool.
2.  **Conv Block 2**: 64 filters, 3x3 kernel, BatchNorm, ReLU, MaxPool.
3.  **Conv Block 3**: 128 filters, 3x3 kernel, BatchNorm, ReLU, AdaptiveAvgPool.
4.  **Classifier**: Flatten -> Linear(2048) -> ReLU -> Dropout(0.4) -> Linear(2).

---

## 📊 Performance Metrics

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **97.4%** | Overall correctness on the test set. |
| **Precision** | **97.2%** | Ratio of true positives to all positive predictions. |
| **Recall** | **97.6%** | Ratio of true positives to all actual wake words. |
| **F1 Score** | **97.4%** | Harmonic mean of precision and recall. |
| **FAR (False Accept Rate)** | **2.8%** | Rate of incorrect triggers on non-wake words. |
| **Model Size** | **2.4 MB** | Size of the exported `.onnx` file. |

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10
- Git
- Speaker/Microphone for testing

### Quick Start
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Sujit-O2/WAKE-WORD-MODEL.git
    cd WAKE-WORD-MODEL
    ```

2.  **Initialize Environment**:
    ```bash
    .\setup_env.bat
    ```

3.  **Download Voices & Tools**:
    The setup script will automatically download Piper TTS and the required voice models.

---

## 🧪 Data Generation Pipeline

The data pipeline is designed to be fully automated and parallelizable.

### 1. Synthetic Positive Generation (`01_generate_positive.py`)
Generates 1500 versions of "Zerotwo" using 6 distinct voices:
- `en_US-lessac-medium`
- `en_US-amy-medium`
- `en_US-ryan-medium`
- `en_US-joe-medium`
- `en_US-kusal-medium`
- `en_GB-alan-medium`

### 2. Negative Speech Generation (`03_generate_negative.py`)
Generates 1500 samples of random phrases and "hard negative" words like "Zero", "Two", "Hero", "To", etc., to prevent accidental triggers.

### 3. Noise Synthesis (`02_generate_noise.py`)
Creates 1500 background noise files (white noise, pink noise, low-frequency hums) to simulate real-world environments.

### 4. Real Recording Integration (`00_process_real.py`)
If you have real-world recordings (`.m4a`), this script converts them to 16kHz mono WAV and normalizes the volume.

---

## 🚀 Training Methodology

The training script (`06_train.py`) uses PyTorch and follows these steps:
1.  **Dataset Loading**: Loads all clean files from `dataset/`.
2.  **Split**: 85% Training, 15% Validation.
3.  **Optimizer**: Adam with weight decay.
4.  **Loss Function**: Negative Log Likelihood (NLLLoss) with LogSoftmax output.
5.  **Schedulers**: StepLR to reduce learning rate as training converges.
6.  **Checkpointing**: Saves the best performing model as `zerotwo_best.pt`.

---

## 📐 Evaluation & Validation

Run `07_evaluate.py` to see a detailed breakdown of your model's performance. It generates a confusion matrix and a threshold sweep:

```text
Threshold  Precision   Recall       F1      FAR
  0.70      0.969    0.984    0.976    0.032
  0.80      0.972    0.976    0.974    0.028
  0.85      0.980    0.972    0.976    0.020
```

We recommend a **threshold of 0.85** for most users.

---

## 🎤 Real-time Inference

The `08_realtime.py` script provides a live testing environment. It uses a sliding window (1.0s window, 0.5s step) to process audio from your microphone.

### Confirmation Logic
To prevent "blip" triggers, the script requires **3 consecutive detections** above the threshold before firing a notification. This significantly reduces false positives from sharp environmental noises.

---

## 📱 Android Integration

For mobile developers, we provide an optimized `.onnx` model.

### Steps:
1.  **Assets**: Place `zerotwo_v1.onnx` in `src/main/assets/`.
2.  **Library**: Add `com.microsoft.onnxruntime:onnxruntime-android:1.15.1` to your `build.gradle`.
3.  **Capture**: Capture 16kHz PCM audio.
4.  **Pre-process**: Convert audio to Mel-spectrogram in Kotlin/Java.
5.  **Predict**: Run the ONNX session with the spectrogram as input.

Full details in [ANDROID_GUIDE.md](ANDROID_GUIDE.md).

---

## 🛠️ Troubleshooting

### Piper TTS Crashes
- Ensure `espeak-ng-data` is not flattened. Run `fix_piper.py` if you get espeak errors.

### No Microphone Input
- Check your OS permissions for the terminal/IDE.
- Ensure `sounddevice` is correctly installed.

### High False Triggers
- Increase the `THRESHOLD` in the realtime script or Android app.
- Check the `CONFIRMATION_FRAMES` setting.

---

## 🗺️ Future Roadmap

- [ ] **Quantization**: Reduce model size by 4x using INT8 quantization.
- [ ] **More Accents**: Add Indian, Australian, and Scottish English variants.
- [ ] **Background Model**: Implement a separate "is_speech" model to save power.
- [ ] **On-device Training**: Allow users to calibrate the model with 5 local samples.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

### Developed with ❤️ by Sujit-O2
### Support
If you find this project useful, please give it a ⭐ on GitHub!
