"""
Classify and Generate using Model C (4-Taal Bar-Aware LSTM)
Automatic processing without interactive prompts
"""

import sys
import torch
import librosa
import numpy as np
import os
import soundfile as sf
from meter_conditional_lstm import create_model

# CNN Model Architecture
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

        adjusted = adjust_length_fn(stroke, target_length)
        features = preprocess(adjusted, sr)
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            pred_index = torch.argmax(output, dim=1).item()
            predicted_bol = note_labels[pred_index]

        results.append(predicted_bol)

    return results, duration

def quantize_duration(duration, bins=[0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]):
    """Quantize continuous duration into discrete bins"""
    for i, edge in enumerate(bins[1:]):
        if duration < edge:
            return i
    return len(bins) - 2

def duration_bins_to_seconds(duration_bins):
    """Convert duration bin indices to approximate seconds"""
    bin_midpoints = [0.15, 0.4, 0.65, 1.15, 2.0]
    return [bin_midpoints[idx] for idx in duration_bins]

def export_audio(notes, durations, output_file, use_drums=True, fade_ms=5, ghost_notes=True, ghost_volume=0.3):
    """Export audio sequence to WAV file"""
    folder = "drums" if use_drums else "tabla"

    print(f"🎵 Exporting {len(notes)} notes to {output_file}...")

    # Calculate total duration
    total_duration = sum(durations)
    sr = 44100  # Standard sample rate

    # Create empty array for final audio
    output_audio = np.zeros(int(sr * (total_duration + 2)))  # Add 2s buffer

    current_sample = 0

    for note, duration in zip(notes, durations):
        audio_file = f"{folder}/{note}.wav"

        if not os.path.exists(audio_file):
            print(f"⚠️  Audio file not found: {audio_file}")
            current_sample += int(sr * duration)
            continue

        audio_data, file_sr = librosa.load(audio_file, sr=sr)

        # Ghost notes
        if ghost_notes and note == "T":
            audio_data = audio_data * ghost_volume

        # Fade in/out
        fade_samples = int(sr * fade_ms / 1000)
        fade_samples = min(fade_samples, len(audio_data) // 4)

        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            audio_data[:fade_samples] *= fade_in
            fade_out = np.linspace(1, 0, fade_samples)
            audio_data[-fade_samples:] *= fade_out

        # Add to output
        end_sample = current_sample + len(audio_data)
        if end_sample > len(output_audio):
            output_audio = np.pad(output_audio, (0, end_sample - len(output_audio)), mode='constant')

        output_audio[current_sample:end_sample] += audio_data
        current_sample += int(sr * duration)

    # Normalize
    max_val = np.max(np.abs(output_audio))
    if max_val > 0:
        output_audio = output_audio / max_val * 0.9

    # Save
    sf.write(output_file, output_audio, sr)
    print(f"✅ Saved: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_and_generate_modelc.py <audio_file.wav> [temperature]")
        print("Example: python classify_and_generate_modelc.py input/my_tabla.wav 1.0")
        sys.exit(1)

    audio_file = sys.argv[1]
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    # Get base filename for outputs
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_dir = "generated_audio"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("🎵 TABLA CLASSIFICATION & GENERATION (Model C - 4 Taals)")
    print("="*80)
    print(f"\n📂 Input file: {audio_file}")
    print(f"🌡️  Temperature: {temperature}")

    # Step 1: Load CNN model
    print("\n📦 Loading CNN classification model...")
    model_path = "ConvNet_SNFPR_model.pth"
    model_CNN = ConvNet(input_channels=13, num_classes=12)
    model_CNN.load_state_dict(torch.load(model_path))
    print("✅ CNN model loaded")

    # Step 2: Classify
    print("\n🔍 Classifying tabla notes...")
    predicted_notes, durations = predict_tabla_bols(
        file_path=audio_file,
        model=model_CNN,
        adjust_length_fn=Adjust_Length,
        target_length=72000
    )

    print("\n" + "="*80)
    print("📊 CLASSIFICATION RESULTS (INPUT)")
    print("="*80)
    print(f"\nDetected {len(predicted_notes)} tabla strokes:")
    print(f"Notes:     {' '.join(predicted_notes)}")
    print(f"Durations: {' '.join([f'{d:.2f}' for d in durations])}")
    print(f"Total duration: {sum(durations):.2f}s")
    print("="*80)

    # Step 3: Load Model C (4-Taal Bar-Aware LSTM)
    print("\n📦 Loading Model C (4-Taal Bar-Aware LSTM)...")
    model_path_lstm = "models/best_bar_aware_lstm.pth"

    checkpoint = torch.load(model_path_lstm, map_location='cpu')
    metadata = checkpoint['metadata']

    lstm_model, regularizer = create_model(
        vocab_size=metadata['num_classes'],
        num_duration_bins=metadata['num_duration_bins'],
        num_taals=len(metadata['taal_mapping']),
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        note_labels=metadata['note_labels']
    )

    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()
    print(f"✅ Model C loaded (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f})")

    # Get taal mapping
    taal_mapping = {v: k for k, v in metadata['taal_mapping'].items()}
    print(f"🎼 Supported taals: {list(taal_mapping.values())}")

    # Step 4: Generate using Model C
    print("\n🤖 Generating AI response with Model C...")

    # Convert notes to indices
    note_to_idx = {note: idx for idx, note in enumerate(metadata['note_labels'])}

    # Take last 16 notes as seed (one Teental bar)
    seed_window = min(16, len(predicted_notes))
    seed_notes_subset = predicted_notes[-seed_window:]
    seed_durations_subset = durations[-seed_window:]

    # Convert to indices
    seed_note_indices = [note_to_idx[note] for note in seed_notes_subset]
    seed_duration_bins = [quantize_duration(d) for d in seed_durations_subset]

    # Generate (for Teental, taal_id=0)
    gen_note_indices, gen_dur_indices = lstm_model.generate(
        seed_notes=seed_note_indices,
        seed_durations=seed_duration_bins,
        taal_id=0,  # Teental
        num_generate=len(predicted_notes),
        temperature=temperature,
        device='cpu'
    )

    # Convert back to note names
    idx_to_note = metadata['note_labels']
    response_notes = [idx_to_note[idx] for idx in gen_note_indices]
    response_durations = duration_bins_to_seconds(gen_dur_indices)

    print("\n" + "="*80)
    print("🤖 AI GENERATED RESPONSE (Model C)")
    print("="*80)
    print(f"\nGenerated {len(response_notes)} tabla strokes:")
    print(f"Notes:     {' '.join(response_notes)}")
    print(f"Durations: {' '.join([f'{d:.2f}' for d in response_durations])}")
    print(f"Total duration: {sum(response_durations):.2f}s")
    print("="*80)

    # Step 5: Export audio files
    print("\n💾 Exporting audio files...")

    # Export input as tabla
    input_tabla_file = f"{output_dir}/{base_name}_input_tabla.wav"
    export_audio(predicted_notes, durations, input_tabla_file, use_drums=False)

    # Export input as drums
    input_drums_file = f"{output_dir}/{base_name}_input_drums.wav"
    export_audio(predicted_notes, durations, input_drums_file, use_drums=True)

    # Export response as tabla
    response_tabla_file = f"{output_dir}/{base_name}_response_tabla_modelc.wav"
    export_audio(response_notes, response_durations, response_tabla_file, use_drums=False)

    # Export response as drums
    response_drums_file = f"{output_dir}/{base_name}_response_drums_modelc.wav"
    export_audio(response_notes, response_durations, response_drums_file, use_drums=True)

    print("\n" + "="*80)
    print("✨ COMPLETE!")
    print("="*80)
    print("\n📁 Output files:")
    print(f"   INPUT (Tabla):    {input_tabla_file}")
    print(f"   INPUT (Drums):    {input_drums_file}")
    print(f"   RESPONSE (Tabla): {response_tabla_file}")
    print(f"   RESPONSE (Drums): {response_drums_file}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()