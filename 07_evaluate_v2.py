"""
07_evaluate_v2.py
=================
Evaluates zerotwo_v2.onnx with noise robustness testing.
Tests performance at multiple SNR levels.
"""
import sys, os, random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import onnxruntime as ort
import librosa

PROJECT_DIR = Path(__file__).parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_PATH = PROJECT_DIR / "models" / "zerotwo_v2.onnx"
SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 160
WIN_LENGTH = 400
N_FFT = 512
N_FRAMES = 100
THRESHOLD = 0.50
N_TEST = 500
SNR_LEVELS = [0, 5, 10, 15, 20, 999]


def extract_features(audio_path):
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        tgt = N_FRAMES * HOP_LENGTH
        if len(audio) < tgt:
            audio = np.pad(audio, (0, tgt - len(audio)))
        else:
            audio = audio[:tgt]
        mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        mel_db = librosa.power_to_db(mel + 1e-9, ref=np.max)
        delta = librosa.feature.delta(mel_db)
        delta2 = librosa.feature.delta(mel_db, order=2)
        feat = np.stack([mel_db, delta, delta2], axis=0)
        for c in range(3):
            feat[c] = (feat[c] - feat[c].mean()) / (feat[c].std() + 1e-9)
        return feat[:, :, :N_FRAMES].astype(np.float32)[np.newaxis, ...]
    except:
        return None


def mix_with_noise(clean, noise_audios, snr_db):
    if snr_db >= 100: return clean
    noise = random.choice(noise_audios)
    if len(noise) < len(clean):
        noise = np.tile(noise, len(clean)//len(noise)+1)
    s = random.randint(0, len(noise)-len(clean))
    noise = noise[s:s+len(clean)]
    cp = np.mean(clean**2)+1e-9; np_ = np.mean(noise**2)+1e-9
    sl = 10**(snr_db/10.0)
    mixed = clean + noise * np.sqrt(cp/(np_*sl))
    pk = np.max(np.abs(mixed))
    if pk > 0.95: mixed = mixed/pk*0.9
    return mixed.astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max()); return e/e.sum()


def load_test_files():
    pos = sorted((DATASET_DIR/"positive").glob("*.wav"))
    neg = sorted((DATASET_DIR/"negative").glob("*.wav")) + sorted((DATASET_DIR/"background").glob("*.wav"))
    n = min(N_TEST//2, len(pos), len(neg))
    ps = random.sample(pos, n); ns = random.sample(neg, n)
    tf = [(str(f),1) for f in ps] + [(str(f),0) for f in ns]
    random.shuffle(tf); return tf


def load_noise_files():
    audios = []
    for f in sorted((DATASET_DIR/"background").glob("*.wav")):
        try:
            a, _ = librosa.load(str(f), sr=SAMPLE_RATE, mono=True)
            if len(a) > 100: audios.append(a)
        except: pass
    return audios


def evaluate_at_snr(sess, inp, out, test_files, noise_audios, snr_db, threshold):
    yt, ys = [], []
    for fp, lab in test_files:
        try:
            audio, _ = librosa.load(fp, sr=SAMPLE_RATE, mono=True)
            tgt = N_FRAMES*HOP_LENGTH
            if len(audio) < tgt: audio = np.pad(audio, (0, tgt-len(audio)))
            else:
                s = random.randint(0, len(audio)-tgt); audio = audio[s:s+tgt]
            if snr_db < 100 and noise_audios:
                audio = mix_with_noise(audio, noise_audios, snr_db)
            mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS,
                n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
            mel_db = librosa.power_to_db(mel+1e-9, ref=np.max)
            delta = librosa.feature.delta(mel_db); delta2 = librosa.feature.delta(mel_db, order=2)
            feat = np.stack([mel_db, delta, delta2], axis=0)
            for c in range(3):
                feat[c] = (feat[c]-feat[c].mean())/(feat[c].std()+1e-9)
            mi = feat[:,:,:N_FRAMES].astype(np.float32)[np.newaxis,...]
            logits = sess.run([out], {inp: mi})[0][0]
            score = float(softmax(logits)[1])
            yt.append(lab); ys.append(score)
        except: pass
    yt, ys = np.array(yt), np.array(ys)
    yp = (ys >= threshold).astype(int)
    tp = int(((yp==1)&(yt==1)).sum()); fp = int(((yp==1)&(yt==0)).sum())
    tn = int(((yp==0)&(yt==0)).sum()); fn = int(((yp==0)&(yt==1)).sum())
    p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9); f1 = 2*p*r/(p+r+1e-9)
    acc = (tp+tn)/max(len(yt),1); far = fp/(fp+tn+1e-9)
    return dict(precision=p, recall=r, f1=f1, accuracy=acc, far=far)


def main():
    print("="*65)
    print("  ZEROTWO EVALUATOR v2 (Noise Robustness)")
    print("="*65)
    if not MODEL_PATH.exists():
        print(f"  Model not found: {MODEL_PATH}\n  Run 06_train.py first."); return

    sess = ort.InferenceSession(str(MODEL_PATH))
    inp = sess.get_inputs()[0].name; out = sess.get_outputs()[0].name
    print(f"  Model: {MODEL_PATH.name}")

    tf = load_test_files(); print(f"  Test: {len(tf)} files")
    na = load_noise_files(); print(f"  Noise: {len(na)} files")

    # Clean eval
    print("\n  CLEAN THRESHOLD SWEEP:")
    yt, ys = [], []
    for fp, lab in tqdm(tf, desc="  Clean"):
        m = extract_features(fp)
        if m is None: continue
        score = float(softmax(sess.run([out], {inp: m})[0][0])[1])
        yt.append(lab); ys.append(score)
    yt, ys = np.array(yt), np.array(ys)
    print(f"\n  {'Thr':>6} {'Prec':>8} {'Rec':>8} {'F1':>8} {'FAR':>8}")
    for thr in [0.5,0.6,0.7,0.8,0.85,0.9,0.95]:
        yp = (ys>=thr).astype(int)
        tp_=int(((yp==1)&(yt==1)).sum()); fp_=int(((yp==1)&(yt==0)).sum())
        fn_=int(((yp==0)&(yt==1)).sum()); tn_=int(((yp==0)&(yt==0)).sum())
        p_=tp_/(tp_+fp_+1e-9); r_=tp_/(tp_+fn_+1e-9)
        f1_=2*p_*r_/(p_+r_+1e-9); fa_=fp_/(fp_+tn_+1e-9)
        print(f"  {thr:>6.2f} {p_:>8.3f} {r_:>8.3f} {f1_:>8.3f} {fa_:>8.3f}")

    # Noise robustness
    print("\n  NOISE ROBUSTNESS:")
    print(f"  {'SNR':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Acc':>8} {'FAR':>8}")
    for snr in SNR_LEVELS:
        r = evaluate_at_snr(sess, inp, out, tf, na, snr, THRESHOLD)
        lbl = "clean" if snr>=100 else f"{snr}dB"
        print(f"  {lbl:>8} {r['precision']:>8.3f} {r['recall']:>8.3f} {r['f1']:>8.3f} {r['accuracy']:>8.3f} {r['far']:>8.3f}")

    c = evaluate_at_snr(sess, inp, out, tf, na, 999, THRESHOLD)
    n = evaluate_at_snr(sess, inp, out, tf, na, 5, THRESHOLD)
    print(f"\n  Clean F1: {c['f1']:.3f} | Noisy F1: {n['f1']:.3f} | Drop: {c['f1']-n['f1']:.3f}")
    print("="*65)


if __name__ == "__main__":
    main()
