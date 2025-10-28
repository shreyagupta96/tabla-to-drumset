"""
Standalone Jugalbandi Terminal Script
Classification + Generation without importing api.py (to avoid Flask server)
"""

import sys
import torch
import librosa
import numpy as np
from lstm_model import create_model
import os
import time
import subprocess
import tempfile
import soundfile as sf

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

# LSTM generation functions
def load_lstm_model(model_path):
    """Load trained LSTM model for generation"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        vocab_size = checkpoint['vocab_size']

        model = create_model(vocab_size=vocab_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        note_to_idx = checkpoint['note_to_idx']
        idx_to_note = {int(k): v for k, v in checkpoint['idx_to_note'].items()}

        return model, note_to_idx, idx_to_note
    except Exception as e:
        print(f"⚠️  Could not load LSTM model: {e}")
        return None, None, None

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

def generate_response(lstm_model, seed_notes, seed_durations, note_to_idx, idx_to_note,
                     num_generate=None, temperature=1.0):
    """Generate tabla response using LSTM"""
    if num_generate is None:
        num_generate = len(seed_notes)

    # Take last 8 notes as seed
    seed_window = min(8, len(seed_notes))
    seed_notes_subset = seed_notes[-seed_window:]
    seed_durations_subset = seed_durations[-seed_window:]

    # Convert to indices
    seed_note_indices = [note_to_idx[note] for note in seed_notes_subset]
    seed_duration_bins = [quantize_duration(d) for d in seed_durations_subset]

    # Generate
    gen_note_indices, gen_dur_indices = lstm_model.generate(
        seed_notes=seed_note_indices,
        seed_durations=seed_duration_bins,
        num_generate=num_generate,
        temperature=temperature,
        device='cpu'
    )

    # Convert back
    gen_note_names = [idx_to_note[idx] for idx in gen_note_indices]
    gen_durations = duration_bins_to_seconds(gen_dur_indices)

    return gen_note_names, gen_durations

def play_audio_terminal(notes, durations, use_drums=True, fade_ms=5, ghost_notes=True,
                       ghost_volume=0.3, label=""):
    """Play audio files in terminal with fade-in/fade-out"""
    folder = "drums" if use_drums else "tabla"

    if label:
        print(f"\n{label}")
    print(f"🎵 Playing {len(notes)} notes...")
    if ghost_notes:
        print(f"👻 Ghost notes enabled (T notes at {int(ghost_volume*100)}% volume)")
    print("📦 Pre-processing audio files...")

    temp_files = []
    processed_notes = []

    for i, (note, duration) in enumerate(zip(notes, durations)):
        audio_file = f"{folder}/{note}.wav"

        if not os.path.exists(audio_file):
            print(f"⚠️  Audio file not found: {audio_file}")
            continue

        audio_data, sr = librosa.load(audio_file, sr=None)

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

        # Save temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, audio_data, sr)
        temp_files.append(temp_file.name)
        temp_file.close()

        is_ghost = ghost_notes and note == "T"
        processed_notes.append((note, duration, i, is_ghost))

    print("-" * 50)

    # Play with accurate timing
    start_time = time.time()

    for note, duration, index, is_ghost in processed_notes:
        temp_file_path = temp_files[index]

        target_time = start_time + sum(durations[:index])
        current_time = time.time()
        wait_time = target_time - current_time

        if wait_time > 0:
            time.sleep(wait_time)

        ghost_indicator = " 👻" if is_ghost else ""
        print(f"Playing: {note} ({duration:.2f}s){ghost_indicator}")

        if sys.platform == "darwin":
            subprocess.Popen(['afplay', temp_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith("linux"):
            subprocess.Popen(['aplay', temp_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "win32":
            subprocess.Popen(['powershell', '-c', f'(New-Object Media.SoundPlayer "{temp_file_path}").PlaySync()'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    time.sleep(2)

    # Cleanup
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except:
            pass

    print("-" * 50)
    print("✅ Playback complete!\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python jugalbandi_standalone.py <audio_file.wav> [--generate]")
        print("Example: python jugalbandi_standalone.py input/my_tabla.wav")
        print("         python jugalbandi_standalone.py input/my_tabla.wav --generate")
        sys.exit(1)

    audio_file = sys.argv[1]
    enable_generation = "--generate" in sys.argv

    print("\n" + "="*80)
    if enable_generation:
        print("🎵 TABLA JUGALBANDI SYSTEM (Classification + Generation)")
    else:
        print("🎵 TABLA CLASSIFICATION SYSTEM")
    print("="*80)
    print(f"\n📂 Input file: {audio_file}")

    # Load CNN model
    print("\n📦 Loading classification model...")
    model_path = "ConvNet_SNFPR_model.pth"
    model_CNN = ConvNet(input_channels=13, num_classes=12)
    model_CNN.load_state_dict(torch.load(model_path))

    # Load LSTM if needed
    lstm_model, note_to_idx, idx_to_note = None, None, None
    if enable_generation:
        print("📦 Loading generation model...")
        lstm_path = "tabla_lstm_model.pth"
        lstm_model, note_to_idx, idx_to_note = load_lstm_model(lstm_path)
        if lstm_model is None:
            print("⚠️  Generation disabled (LSTM model not available)")
            enable_generation = False

    # Classify
    print("\n🔍 Classifying tabla notes...")
    predicted_notes, durations = predict_tabla_bols(
        file_path=audio_file,
        model=model_CNN,
        adjust_length_fn=Adjust_Length,
        target_length=72000
    )

    # Display results
    print("\n" + "="*80)
    if enable_generation:
        print("📊 INPUT PHRASE (CALL)")
    else:
        print("📊 CLASSIFICATION RESULTS")
    print("="*80)
    print(f"\nDetected {len(predicted_notes)} tabla strokes:")
    print(f"Notes:     {' '.join(predicted_notes)}")
    print(f"Durations: {' '.join([f'{d:.2f}' for d in durations])}")
    print(f"Total duration: {sum(durations):.2f}s")
    print("="*80)

    # Generate if enabled
    response_notes, response_durations = None, None
    if enable_generation and lstm_model is not None:
        print("\n🤖 Generate AI response?")
        print("   1. Conservative (temp=0.8) - Structured, safe")
        print("   2. Balanced (temp=1.0) - Musical, varied")
        print("   3. Creative (temp=1.2) - Improvisational, diverse")
        print("   4. Custom temperature")
        print("   5. Skip generation")

        gen_choice = input("\nChoose option (1-5, default=2): ").strip()

        if gen_choice != "5":
            if gen_choice == "1":
                temperature = 0.8
            elif gen_choice == "3":
                temperature = 1.2
            elif gen_choice == "4":
                try:
                    temperature = float(input("Enter temperature (0.5-2.0): ").strip())
                    temperature = max(0.5, min(2.0, temperature))
                except:
                    temperature = 1.0
                    print("Invalid input, using default 1.0")
            else:
                temperature = 1.0

            print(f"\n🌡️  Generating with temperature: {temperature}")

            response_notes, response_durations = generate_response(
                lstm_model, predicted_notes, durations, note_to_idx, idx_to_note,
                num_generate=len(predicted_notes),
                temperature=temperature
            )

            # Display response
            print("\n" + "="*80)
            print("🤖 AI RESPONSE")
            print("="*80)
            print(f"\nGenerated {len(response_notes)} tabla strokes:")
            print(f"Notes:     {' '.join(response_notes)}")
            print(f"Durations: {' '.join([f'{d:.2f}' for d in response_durations])}")
            print(f"Total duration: {sum(response_durations):.2f}s")
            print("="*80)

    # Playback
    print("\n🔊 Playback options:")
    if response_notes is not None:
        print("  1. Play CALL only (input)")
        print("  2. Play RESPONSE only (AI)")
        print("  3. Play both (CALL → RESPONSE) as TABLA")
        print("  4. Play both as DRUMS")
        print("  5. Skip playback")
        choice_prompt = "\nEnter choice (1-5, default=3): "
    else:
        print("  1. Play as DRUMS")
        print("  2. Play as TABLA")
        print("  3. Skip playback")
        choice_prompt = "\nEnter choice (1-3, default=1): "

    choice = input(choice_prompt).strip()

    if response_notes is not None:
        if choice == "1":
            play_audio_terminal(predicted_notes, durations, use_drums=False, label="🎵 CALL (Input Phrase)")
        elif choice == "2":
            play_audio_terminal(response_notes, response_durations, use_drums=False, label="🤖 RESPONSE (AI Generated)")
        elif choice == "4":
            print("\n🥁 Playing as DRUMS:")
            play_audio_terminal(predicted_notes, durations, use_drums=True, label="🎵 CALL (Drums)")
            time.sleep(1)
            play_audio_terminal(response_notes, response_durations, use_drums=True, label="🤖 RESPONSE (Drums)")
        elif choice == "5":
            print("⏭️  Skipping playback")
        else:
            print("\n🎵 Playing JUGALBANDI (Call-and-Response):")
            play_audio_terminal(predicted_notes, durations, use_drums=False, label="🎵 CALL (Input Phrase)")
            time.sleep(1)
            play_audio_terminal(response_notes, response_durations, use_drums=False, label="🤖 RESPONSE (AI Generated)")
    else:
        if choice == "1" or choice == "":
            play_audio_terminal(predicted_notes, durations, use_drums=True)
        elif choice == "2":
            play_audio_terminal(predicted_notes, durations, use_drums=False)
        else:
            print("⏭️  Skipping playback")

    print("\n" + "="*80)
    print("✨ Done!")
    print("="*80)
    print()

if __name__ == "__main__":
    main()
