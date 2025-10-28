"""
Phase 1: Batch Classification Pipeline
Classify all 12 long tabla files from Taal Variations folder
"""

import torch
import torch.nn as nn
import librosa
import numpy as np
import json
import os
from scipy.signal import find_peaks

# CNN Model (from classify_no_flask.py)
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
    """Detect onsets using RCD method"""
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
    """Extract MFCC and Chroma features"""
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
    """Pad or truncate audio to target length"""
    if len(audio_data) < target_length:
        return np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
    else:
        return audio_data

def classify_tabla_file(file_path, model, target_length, db_threshold=-30, pre_onset_samples=1000):
    """
    Classify all tabla strokes in a file
    Returns: notes, durations, onset_samples
    """
    print(f"\n📂 Loading: {os.path.basename(file_path)}")
    y, sr = librosa.load(file_path, sr=None)
    print(f"   Duration: {len(y)/sr:.1f}s, Sample rate: {sr}Hz")

    print(f"   Detecting onsets...")
    onset_samples, onset_env = compute_rcd_onsets(y, sr)

    if 0 not in onset_samples:
        onset_samples = np.insert(onset_samples, 0, 0)

    print(f"   Found {len(onset_samples)-1} tabla strokes")
    print(f"   Classifying strokes...")

    results = []
    durations = []
    valid_onsets = []

    for i in range(len(onset_samples) - 1):
        start = max(onset_samples[i] - pre_onset_samples, 0)
        end = onset_samples[i + 1]
        duration_samples = end - start
        duration_sec = duration_samples / sr
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

        # Only append if classification succeeded
        results.append(predicted_bol)
        durations.append(duration_sec)
        valid_onsets.append(int(onset_samples[i]))  # Convert to Python int

    print(f"   ✅ Classified {len(results)} notes")

    # Apply "Ti Re Ki T" rule correction
    from tabla_rule_correction import apply_corrections_and_report
    corrected_notes, corrected_durations, num_corrections = apply_corrections_and_report(
        results, durations, os.path.basename(file_path)
    )

    return corrected_notes, corrected_durations, valid_onsets

def batch_classify_files(input_dir, output_dir, model_path):
    """
    Batch classify all wav files in input directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load CNN model
    print("=" * 100)
    print(" " * 30 + "PHASE 1: BATCH CLASSIFICATION")
    print("=" * 100)
    print("\n📦 Loading CNN model...")
    model = ConvNet(input_channels=13, num_classes=12)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("✅ Model loaded")

    # Find all wav files
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    wav_files.sort()

    print(f"\n📋 Found {len(wav_files)} files to classify")

    # Process each file
    results_summary = []

    for i, filename in enumerate(wav_files, 1):
        filepath = os.path.join(input_dir, filename)

        print(f"\n{'=' * 100}")
        print(f"FILE {i}/{len(wav_files)}: {filename}")
        print(f"{'=' * 100}")

        try:
            # Check if already classified
            output_file = os.path.join(output_dir, filename.replace('.wav', '_classified.json'))
            if os.path.exists(output_file):
                print(f"⏭️  Already classified, skipping...")
                with open(output_file, 'r') as f:
                    data = json.load(f)
                results_summary.append({
                    'file': filename,
                    'status': 'already_done',
                    'num_notes': len(data['notes'])
                })
                continue

            # Classify
            notes, durations, onset_samples = classify_tabla_file(
                filepath, model, target_length=72000
            )

            # Save result
            result = {
                'file': filename,
                'filepath': filepath,
                'notes': notes,
                'durations': durations,
                'onset_samples': onset_samples,
                'num_notes': len(notes),
                'total_duration_sec': sum(durations)
            }

            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"   💾 Saved to: {os.path.basename(output_file)}")

            results_summary.append({
                'file': filename,
                'status': 'success',
                'num_notes': len(notes)
            })

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                'file': filename,
                'status': 'error',
                'error': str(e)
            })

    # Print summary
    print(f"\n\n{'=' * 100}")
    print(" " * 35 + "CLASSIFICATION SUMMARY")
    print(f"{'=' * 100}")

    success = sum(1 for r in results_summary if r['status'] in ['success', 'already_done'])
    errors = sum(1 for r in results_summary if r['status'] == 'error')

    print(f"\n✅ Successfully classified: {success}/{len(wav_files)}")
    print(f"❌ Errors: {errors}/{len(wav_files)}")

    if success > 0:
        print(f"\n📊 Classified Files:")
        total_notes = 0
        for r in results_summary:
            if r['status'] in ['success', 'already_done']:
                status_icon = '✅' if r['status'] == 'success' else '⏭️'
                print(f"   {status_icon} {r['file']:50} → {r['num_notes']:4} notes")
                total_notes += r['num_notes']

        print(f"\n📈 Total notes classified: {total_notes}")

    print(f"\n{'=' * 100}")
    print("✅ Phase 1 Complete! Ready for Phase 2 (Bar Segmentation)")
    print(f"{'=' * 100}")

    return results_summary

if __name__ == "__main__":
    # Auto-detect input directory (handle special characters)
    base = "/Users/shreyagupta/Desktop/AI_Research_Data/From Tabla Player"
    taal_dirs = [d for d in os.listdir(base) if 'Taal Variations' in d]

    INPUT_DIR = None
    for d in taal_dirs:
        bounces_path = os.path.join(base, d, 'Bounces')
        if os.path.exists(bounces_path):
            INPUT_DIR = bounces_path
            break

    if INPUT_DIR is None:
        print("❌ Error: Could not find Taal Variations Bounces directory")
        exit(1)

    OUTPUT_DIR = "classified_long_files"
    MODEL_PATH = "ConvNet_SNFPR_model.pth"

    print("\n📌 Configuration:")
    print(f"   Input directory: {INPUT_DIR}")
    print(f"   Output directory: {OUTPUT_DIR}")
    print(f"   CNN model: {MODEL_PATH}")

    # Run batch classification
    results = batch_classify_files(INPUT_DIR, OUTPUT_DIR, MODEL_PATH)
