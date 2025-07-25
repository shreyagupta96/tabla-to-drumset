import json
import os
import torch
import librosa
import pickle
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# CNN Module Block
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


# Define a simple CNN model for classification
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
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))  # Change to 2d pooling
        self.fc = torch.nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = self.global_avg_pool(x).view(x.size(0), -1)  # Flatten to 2d
        network_output = self.fc(x)
        return network_output

# RCD Onset Detection
def compute_rcd_onsets(y, sr, n_fft=2048, hop_length=512, threshold=0.3):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(S), np.angle(S)

    # Estimate phase prediction
    phase_diff = np.diff(phase, axis=1)
    phase_diff = np.pad(phase_diff, ((0, 0), (1, 0)), mode='constant')
    pred = mag[:, :-1] * np.exp(1j * (phase[:, :-1] + phase_diff[:, :-1]))
    error = np.abs(S[:, 1:] - pred)

    # Rectified Complex Domain
    rcd = np.where(mag[:, 1:] >= mag[:, :-1], error, 0)
    onset_env = np.sum(rcd, axis=0)

    # Normalize and peak pick
    onset_env = onset_env - np.mean(onset_env)
    onset_env /= np.std(onset_env) + 1e-8
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3,delta=threshold, wait=5)

    # Convert frames to samples
    onset_samples = librosa.frames_to_samples(peaks, hop_length=hop_length)

    # Append the end of audio to capture the last stroke
    if onset_samples[-1] < len(y):
        onset_samples = np.append(onset_samples, len(y))

    return onset_samples, onset_env

# Function to preprocess audio data
def preprocess(audio_data, sample_rate):
    # Normalize the audio data
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    # Extract MFCCs (13 coefficients)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)

    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate, n_chroma=13)

    # Ensure both features have the same number of time frames
    min_frames = min(mfccs.shape[1], chroma.shape[1])
    mfccs = mfccs[:, :min_frames]
    chroma = chroma[:, :min_frames]

    # Stack features along a new dimension to create a 3D array
    # Shape: (n_features, n_time_frames, n_feature_types)
    features = np.stack([mfccs, chroma], axis=2)
    return features

note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]

# Function to adjust the length of audio data
def Adjust_Length(audio_data, target_length):
    # Right zero-padding
    if len(audio_data) < target_length:
        padded_audio = np.pad(audio_data, (0, target_length - len(audio_data)), mode = 'constant')
        return padded_audio
    else:
        return audio_data

def predict_tabla_bols(
    file_path,
    model,
    adjust_length_fn,
    target_length,
    db_threshold=-30,
    pre_onset_samples=1000
):
    # Load audio
    y, sr = librosa.load(file_path, sr=None)

    # Compute RCD-based onsets
    onset_samples, onset_env = compute_rcd_onsets(y, sr)

    # Manually add 0 if missing at start
    if 0 not in onset_samples:
        initial_db = librosa.amplitude_to_db([np.max(np.abs(y[:2048]))])[0]
        print(f"Added onset at 0s manually (initial dB: {initial_db:.2f})")
        onset_samples = np.insert(onset_samples, 0, 0)

    results = []
    duration = []
    t_axis = np.linspace(0, len(y) / sr, len(y))

    print(f"Detected {len(onset_samples) - 1} strokes")

    for i in range(len(onset_samples) - 1):
        start = max(onset_samples[i] - pre_onset_samples, 0)
        end = onset_samples[i + 1]
        duration_samples = end - start
        duration_sec = duration_samples / sr
        duration.append(duration_sec)
        stroke = y[start:end]

        if stroke.shape[0] == 0:
            print(f"Skipping stroke {i+1}: zero-length.")
            continue

        stroke_db = librosa.amplitude_to_db([np.max(np.abs(stroke))])[0]
        if stroke_db < db_threshold:
            print(f"Skipping stroke {i+1}: below dB threshold ({stroke_db:.2f} dB).")
            continue

        # Preprocess
        adjusted = adjust_length_fn(stroke, target_length)
        features = preprocess(adjusted, sr)
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Predict
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            pred_index = torch.argmax(output, dim=1).item()
            predicted_bol = note_labels[pred_index]

        results.append(predicted_bol)
        print(f"Stroke {i+1}: Predicted → {predicted_bol} (dB: {stroke_db:.2f})")
    return results, duration

def preprocess_generate (predicted_notes, durations):
    merge_list = []
    for i in range(len(predicted_notes)):
        merge_list.append(note_labels.index(predicted_notes[i]))
        merge_list.append(durations[i])
    return merge_list[len(merge_list) - 6:], len(predicted_notes)

def generate_notes (predicted_normalised, durations):
    predicted_notes = []
    predicted_duration = []
    x_test_sample, x_test_length = preprocess_generate(predicted_notes=predicted_normalised, durations=durations)
    for i in range(0,x_test_length):
        pred_note = model.predict([x_test_sample])
        pred_duration = model_2.predict([x_test_sample])
        print(note_labels[pred_note[0]], pred_duration[0])
        predicted_notes.append(note_labels[pred_note[0]])
        predicted_duration.append(pred_duration[0])
        x_test_sample = x_test_sample[2:]
        x_test_sample.append(pred_note[0])
        x_test_sample.append(pred_duration[0])
    return predicted_notes, predicted_duration

def lambda_handler(file):
    predicted_normalised, durations = predict_tabla_bols(file_path=file,
                                                         model=model_CNN, adjust_length_fn=Adjust_Length,
                                                         target_length=72000)
    return predicted_normalised, durations


app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        if 'file' not in request.files:
            return {'error': 'No file part'}, 400
        file = request.files['file']
        predicted_normalised, durations = lambda_handler(file)
        # Fetch user logic here
        return jsonify({'notes': predicted_normalised, 'duration': durations})

@app.route('/nextgen', methods=['GET', 'POST'])
def nextgen():
    if request.method == 'POST':
        if 'file' not in request.files:
            return {'error': 'No file part'}, 400
        file = request.files['file']
        predicted_normalised, durations = lambda_handler(file)
        next_gen_notes, next_gen_duration = generate_notes (predicted_normalised, durations)
        # Fetch user logic here
        return jsonify({'notes': next_gen_notes, 'duration': next_gen_duration})

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_2.pkl', 'rb') as f:
    model_2 = pickle.load(f)

model_path = "ConvNet_SNFPR_model.pth"  # Enclose the filename in quotes
model_CNN = ConvNet(input_channels=13, num_classes=12)  # Ensure input_channels and num_classes match your model
model_CNN.load_state_dict(torch.load(model_path))

app.run(host="0.0.0.0", port=5010)


