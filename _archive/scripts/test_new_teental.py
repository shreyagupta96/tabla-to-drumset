"""
Test with new Teental file from Tabla Player
"""

import torch
import sys
sys.path.insert(0, "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG Copy 3")
from lstm_model import create_model as create_model_copy3
sys.path.pop(0)
from lstm_model import create_model as create_model_current
import librosa
import numpy as np
from scipy.signal import find_peaks

# Import from api (classification functions)
import pickle

def load_model(model_path, create_fn):
    checkpoint = torch.load(model_path, map_location='cpu')
    vocab_size = checkpoint['vocab_size']
    model = create_fn(vocab_size=vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    note_to_idx = checkpoint['note_to_idx']
    idx_to_note = {int(k): v for k, v in checkpoint['idx_to_note'].items()}
    return model, note_to_idx, idx_to_note

def compute_rcd_onsets(y, sr, n_fft=2048, hop_length=512, threshold=0.3):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(S), np.angle(S)
    phase_diff = np.diff(phase, axis=1)
    phase_diff = np.pad(phase_diff, ((0, 0), (1, 0)), mode='constant')
    pred = mag[:, :-1] * np.exp(1j * (phase[:, :-1] + phase_diff[:, :-1]))
    error = np.abs(S[:, 1:] - pred)
    rcd = np.where(mag[:, 1:] >= mag[:, :-1], error, 0)
    onset_env = np.sum(rcd, axis=0)
    onset_env = onset_env - np.mean(onset_env)
    onset_env /= np.std(onset_env) + 1e-8
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=threshold, wait=5)
    onset_samples = librosa.frames_to_samples(peaks, hop_length=hop_length)
    if len(onset_samples) == 0 or onset_samples[-1] < len(y):
        onset_samples = np.append(onset_samples, len(y))
    return onset_samples, onset_env

def preprocess(audio_data, sample_rate):
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate, n_chroma=13)
    min_frames = min(mfccs.shape[1], chroma.shape[1])
    mfccs = mfccs[:, :min_frames]
    chroma = chroma[:, :min_frames]
    features = np.stack([mfccs, chroma], axis=2)
    return features

def adjust_length(audio_data, target_length):
    if len(audio_data) < target_length:
        return np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
    else:
        return audio_data

# Simple classification without full CNN import
print("=" * 100)
print("TESTING WITH NEW TEENTAL FILE")
print("=" * 100)

audio_file = "/Users/shreyagupta/Desktop/AI_Research_Data/From Tabla Player/Taal Variations � 12-01-2025/Bounces/Teental Dugun Variation#02.wav"

print(f"\n📂 Loading: {audio_file}")
y, sr = librosa.load(audio_file, sr=None)

print(f"✅ Audio loaded: {len(y)/sr:.2f}s duration, {sr}Hz sample rate")

print("\n🔍 Detecting onsets...")
onset_samples, onset_env = compute_rcd_onsets(y, sr)

print(f"✅ Detected {len(onset_samples)-1} tabla strokes")

# Calculate durations
durations = []
for i in range(len(onset_samples) - 1):
    start = onset_samples[i]
    end = onset_samples[i + 1]
    duration_sec = (end - start) / sr
    durations.append(duration_sec)

print(f"\n📊 Stroke durations (first 20):")
for i in range(min(20, len(durations))):
    print(f"  {i+1}. {durations[i]:.3f}s")

print(f"\n📊 Total duration: {sum(durations):.2f}s")
print(f"📊 Average stroke duration: {np.mean(durations):.3f}s")
print(f"📊 Median stroke duration: {np.median(durations):.3f}s")

print("\n" + "=" * 100)
print("NOTE: This file needs full CNN classification to get note predictions.")
print("The classify_terminal.py has a Flask import issue preventing direct use.")
print("Would you like me to:")
print("  1. Create a Flask-free classifier script")
print("  2. Use one of the existing input files instead")
print("=" * 100)
