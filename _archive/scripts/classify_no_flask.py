"""
Classification without Flask import
"""

import torch
import torch.nn as nn
import librosa
import numpy as np
from scipy.signal import find_peaks

# CNN Model
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvNet(torch.nn.Module):
    def __init__(self, input_channels, num_classes=12, num_channels=16, kernel_size=5):
        super(ConvNet, self).__init__()
        self.layers = torch.nn.Sequential(
            ConvBlock(input_channels, num_channels, kernel_size),
            ConvBlock(num_channels, num_channels, kernel_size),
            ConvBlock(num_channels, num_channels, kernel_size),
            ConvBlock(num_channels, num_channels, kernel_size),
            ConvBlock(num_channels, num_channels, kernel_size),
            ConvBlock(num_channels, num_channels, kernel_size),
            ConvBlock(num_channels, num_channels, kernel_size),
            ConvBlock(num_channels, num_channels, kernel_size),
        )
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = self.global_avg_pool(x).view(x.size(0), -1)
        network_output = self.fc(x)
        return network_output

note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]

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

def predict_tabla_bols(file_path, model, target_length, db_threshold=-30, pre_onset_samples=1000):
    y, sr = librosa.load(file_path, sr=None)
    onset_samples, onset_env = compute_rcd_onsets(y, sr)

    if 0 not in onset_samples:
        onset_samples = np.insert(onset_samples, 0, 0)

    results = []
    duration = []

    for i in range(len(onset_samples) - 1):
        start = max(onset_samples[i] - pre_onset_samples, 0)
        end = onset_samples[i + 1]
        duration_samples = end - start
        duration_sec = duration_samples / sr
        duration.append(duration_sec)
        stroke = y[start:end]

        if stroke.shape[0] == 0:
            continue

        stroke_db = librosa.amplitude_to_db([np.max(np.abs(stroke))])[0]
        if stroke_db < db_threshold:
            continue

        adjusted = adjust_length(stroke, target_length)
        features = preprocess(adjusted, sr)
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            pred_index = torch.argmax(output, dim=1).item()
            predicted_bol = note_labels[pred_index]

        results.append(predicted_bol)

    return results, duration

# Load model
print("Loading CNN model...")
model_CNN = ConvNet(input_channels=13, num_classes=12)
model_CNN.load_state_dict(torch.load("ConvNet_SNFPR_model.pth"))

# Classify
audio_file = "/Users/shreyagupta/Desktop/AI_Research_Data/From Tabla Player/Taal Variations � 12-01-2025/Bounces/Teental Dugun Variation#02.wav"
print(f"Classifying: {audio_file}")

predicted_notes, durations = predict_tabla_bols(
    file_path=audio_file,
    model=model_CNN,
    target_length=72000
)

print(f"\n✅ Classified {len(predicted_notes)} strokes")
print(f"\nFirst 32 notes: {' '.join(predicted_notes[:32])}")
print(f"Total duration: {sum(durations):.2f}s")

# Save for later use
import json
result = {
    "file": audio_file,
    "notes": predicted_notes,
    "durations": durations
}

with open("teental_dugun_classification.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\n💾 Saved classification to teental_dugun_classification.json")
