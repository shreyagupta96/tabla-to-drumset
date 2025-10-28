"""
Standalone Batch Classification Script
Doesn't import api.py to avoid Flask server starting
"""

import os
import json
import torch
import librosa
import numpy as np
from pathlib import Path

# Copy necessary classes and functions from api.py
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

    if onset_samples[-1] < len(y):
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

note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]

def Adjust_Length(audio_data, target_length):
    if len(audio_data) < target_length:
        padded_audio = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
        return padded_audio
    else:
        return audio_data

def predict_tabla_bols(file_path, model, adjust_length_fn, target_length, db_threshold=-30, pre_onset_samples=1000):
    y, sr = librosa.load(file_path, sr=None)

    onset_samples, onset_env = compute_rcd_onsets(y, sr)

    if 0 not in onset_samples:
        initial_db = librosa.amplitude_to_db([np.max(np.abs(y[:2048]))])[0]
        print(f"Added onset at 0s manually (initial dB: {initial_db:.2f})")
        onset_samples = np.insert(onset_samples, 0, 0)

    results = []
    duration = []

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

        adjusted = adjust_length_fn(stroke, target_length)
        features = preprocess(adjusted, sr)
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            pred_index = torch.argmax(output, dim=1).item()
            predicted_bol = note_labels[pred_index]

        results.append(predicted_bol)
        print(f"Stroke {i+1}: Predicted → {predicted_bol} (dB: {stroke_db:.2f})")

    return results, duration

def batch_classify_tabla_files(data_dir, output_file, model_path="ConvNet_SNFPR_model.pth"):
    """Classify all tabla files in the dataset directory"""

    print("=" * 70)
    print("BATCH TABLA CLASSIFICATION")
    print("=" * 70)
    print(f"\n📦 Loading CNN model from {model_path}...")

    model_CNN = ConvNet(input_channels=13, num_classes=12)
    model_CNN.load_state_dict(torch.load(model_path))
    model_CNN.eval()

    print("✅ Model loaded successfully!\n")

    # Find all .wav files recursively
    wav_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    wav_files.sort()
    print(f"🎵 Found {len(wav_files)} WAV files to classify\n")
    print("-" * 70)

    # Store all results
    all_results = {
        "metadata": {
            "total_files": len(wav_files),
            "model_path": model_path,
            "note_vocabulary": note_labels
        },
        "classifications": []
    }

    # Process each file
    for idx, wav_file in enumerate(wav_files, 1):
        relative_path = os.path.relpath(wav_file, data_dir)
        print(f"\n[{idx}/{len(wav_files)}] Processing: {relative_path}")

        try:
            predicted_notes, durations = predict_tabla_bols(
                file_path=wav_file,
                model=model_CNN,
                adjust_length_fn=Adjust_Length,
                target_length=72000
            )

            result = {
                "file_path": wav_file,
                "relative_path": relative_path,
                "filename": os.path.basename(wav_file),
                "num_notes": len(predicted_notes),
                "notes": predicted_notes,
                "durations": durations,
                "total_duration": sum(durations),
                "status": "success"
            }

            print(f"   ✅ Detected {len(predicted_notes)} notes")
            print(f"   🎵 Sequence: {' '.join(predicted_notes[:10])}" +
                  (" ..." if len(predicted_notes) > 10 else ""))
            print(f"   ⏱️  Total duration: {sum(durations):.2f}s")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            result = {
                "file_path": wav_file,
                "relative_path": relative_path,
                "filename": os.path.basename(wav_file),
                "status": "error",
                "error_message": str(e)
            }

        all_results["classifications"].append(result)

    # Save results
    print("\n" + "=" * 70)
    print(f"💾 Saving results to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("✅ Results saved successfully!")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    successful = [r for r in all_results["classifications"] if r["status"] == "success"]
    failed = [r for r in all_results["classifications"] if r["status"] == "error"]

    print(f"\n✅ Successfully classified: {len(successful)} files")
    print(f"❌ Failed: {len(failed)} files")

    if successful:
        total_notes = sum(r["num_notes"] for r in successful)
        avg_notes = total_notes / len(successful)
        total_duration = sum(r["total_duration"] for r in successful)

        print(f"\n📊 Total notes across all files: {total_notes}")
        print(f"📊 Average notes per file: {avg_notes:.1f}")
        print(f"📊 Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")

        from collections import Counter
        all_notes = []
        for r in successful:
            all_notes.extend(r["notes"])

        note_counts = Counter(all_notes)
        print(f"\n🎵 Note Distribution (top 5):")
        for note, count in note_counts.most_common(5):
            percentage = (count / total_notes) * 100
            print(f"   {note:8s}: {count:5d} ({percentage:5.1f}%)")

    if failed:
        print(f"\n❌ Failed files:")
        for r in failed:
            print(f"   - {r['relative_path']}: {r['error_message']}")

    print("\n" + "=" * 70)
    print("🎉 Batch classification complete!")
    print(f"📝 Next step: Review and edit {output_file} for accuracy")
    print("=" * 70)

    return all_results

if __name__ == "__main__":
    DATA_DIR = "/Users/shreyagupta/Desktop/Machine_Learning/Group_Project/Project - Tabla/DataSet/For_Generation"
    OUTPUT_FILE = "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/training_data_classified.json"

    results = batch_classify_tabla_files(DATA_DIR, OUTPUT_FILE)
