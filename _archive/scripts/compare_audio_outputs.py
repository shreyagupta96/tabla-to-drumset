"""
Compare audio outputs from Copy 3 (no embeddings) vs Current (with embeddings)
Generate and play responses from both models with same input
"""

import torch
import sys
import os
import time
import subprocess
import tempfile
import soundfile as sf
import librosa
import numpy as np

# Add Copy 3 to path to import its LSTM model
sys.path.insert(0, "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG Copy 3")
from lstm_model import create_model as create_model_copy3

# Remove Copy 3 from path and import current LSTM model
sys.path.pop(0)
from lstm_model import create_model as create_model_current

def load_model(model_path, create_fn):
    """Load a trained LSTM model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    vocab_size = checkpoint['vocab_size']

    model = create_fn(vocab_size=vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    note_to_idx = checkpoint['note_to_idx']
    idx_to_note = {int(k): v for k, v in checkpoint['idx_to_note'].items()}

    return model, note_to_idx, idx_to_note

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

def generate_response(model, seed_notes, seed_durations, note_to_idx, idx_to_note,
                     num_generate, temperature=1.0):
    """Generate tabla response using LSTM"""
    # Take last 8 notes as seed
    seed_window = min(8, len(seed_notes))
    seed_notes_subset = seed_notes[-seed_window:]
    seed_durations_subset = seed_durations[-seed_window:]

    # Convert to indices
    seed_note_indices = [note_to_idx[note] for note in seed_notes_subset]
    seed_duration_bins = [quantize_duration(d) for d in seed_durations_subset]

    # Generate
    gen_note_indices, gen_dur_indices = model.generate(
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

def play_audio_terminal(notes, durations, use_drums=False, label=""):
    """Play audio files in terminal"""
    folder = "drums" if use_drums else "tabla"

    if label:
        print(f"\n{label}")
    print(f"🎵 Playing: {' '.join(notes)}")
    print("📦 Pre-processing audio files...")

    # Pre-process all audio files
    temp_files = []
    processed_notes = []

    for i, (note, duration) in enumerate(zip(notes, durations)):
        audio_file = f"{folder}/{note}.wav"

        if not os.path.exists(audio_file):
            print(f"⚠️  Audio file not found: {audio_file}")
            continue

        # Load audio and apply fade
        audio_data, sr = librosa.load(audio_file, sr=None)

        # Apply fade
        fade_samples = int(sr * 5 / 1000)  # 5ms fade
        fade_samples = min(fade_samples, len(audio_data) // 4)

        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            audio_data[:fade_samples] *= fade_in
            fade_out = np.linspace(1, 0, fade_samples)
            audio_data[-fade_samples:] *= fade_out

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, audio_data, sr)
        temp_files.append(temp_file.name)
        temp_file.close()

        processed_notes.append((note, duration, i))

    print("-" * 50)

    # Play all pre-processed files
    start_time = time.time()

    for note, duration, index in processed_notes:
        temp_file_path = temp_files[index]

        # Calculate timing
        target_time = start_time + sum(durations[:index])
        current_time = time.time()
        wait_time = target_time - current_time

        if wait_time > 0:
            time.sleep(wait_time)

        print(f"Playing: {note} ({duration:.2f}s)")

        # Play audio
        if sys.platform == "darwin":  # macOS
            subprocess.Popen(['afplay', temp_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith("linux"):
            subprocess.Popen(['aplay', temp_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for last note
    time.sleep(2)

    # Clean up
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except:
            pass

    print("-" * 50)
    print("✅ Playback complete!\n")

def main():
    print("=" * 100)
    print(" " * 25 + "AUDIO COMPARISON: COPY 3 vs CURRENT")
    print("=" * 100)
    print("\n📊 Copy 3: LSTM without embedding regularization (Oct 22)")
    print("📊 Current: LSTM WITH embedding regularization (Oct 25)")
    print("-" * 100)

    # Load both models
    print("\n📦 Loading Copy 3 model (no embeddings)...")
    copy3_model, copy3_note_to_idx, copy3_idx_to_note = load_model(
        "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG Copy 3/tabla_lstm_model.pth",
        create_model_copy3
    )
    print("✅ Copy 3 model loaded")

    print("\n📦 Loading Current model (with embeddings)...")
    current_model, current_note_to_idx, current_idx_to_note = load_model(
        "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/tabla_lstm_model.pth",
        create_model_current
    )
    print("✅ Current model loaded")

    # Test examples
    test_cases = [
        {
            "name": "Rupak Taal (7 beats)",
            "notes": ["Tin", "Tin", "Ta", "Dhin", "Na", "Dhin", "Na"],
            "durations": [0.73, 0.77, 0.80, 0.71, 0.75, 0.80, 0.76]
        },
        {
            "name": "Jhaptaal (10 beats)",
            "notes": ["Dhin", "Na", "Dhin", "Dhin", "Ta", "Tin", "Ta", "Dhin", "Dhin", "Na"],
            "durations": [0.73, 0.64, 0.72, 0.75, 0.70, 0.72, 0.75, 0.68, 0.72, 0.75]
        }
    ]

    print("\n🎵 Choose a test:")
    print("  1. Rupak Taal (7 beats)")
    print("  2. Jhaptaal (10 beats)")

    choice = input("\nEnter choice (1/2, default=1): ").strip()
    test_idx = 1 if choice == "2" else 0
    test = test_cases[test_idx]

    temperature = 1.0

    print(f"\n{'=' * 100}")
    print(f"🎵 {test['name']}")
    print(f"{'=' * 100}")
    print(f"\nINPUT: {' '.join(test['notes'])}")
    print(f"Temperature: {temperature}")

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate with Copy 3 model
    print(f"\n🔄 Generating with Copy 3 model (no embeddings)...")
    copy3_notes, copy3_durs = generate_response(
        copy3_model, test['notes'], test['durations'],
        copy3_note_to_idx, copy3_idx_to_note,
        num_generate=len(test['notes']),
        temperature=temperature
    )

    # Reset seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate with Current model
    print(f"🔄 Generating with Current model (with embeddings)...")
    current_notes, current_durs = generate_response(
        current_model, test['notes'], test['durations'],
        current_note_to_idx, current_idx_to_note,
        num_generate=len(test['notes']),
        temperature=temperature
    )

    print(f"\n{'=' * 100}")
    print("📊 GENERATED RESPONSES")
    print(f"{'=' * 100}")
    print(f"\nCopy 3 (no embeddings):  {' '.join(copy3_notes)}")
    print(f"Current (with embeddings): {' '.join(current_notes)}")

    # Analyze differences
    print(f"\n{'=' * 100}")
    print("📊 ANALYSIS")
    print(f"{'=' * 100}")

    copy3_unique = len(set(copy3_notes))
    current_unique = len(set(current_notes))

    copy3_has_na_ta = 'Na' in copy3_notes and 'Ta' in copy3_notes
    current_has_na_ta = 'Na' in current_notes and 'Ta' in current_notes

    copy3_has_tin_tun = 'Tin' in copy3_notes and 'Tun' in copy3_notes
    current_has_tin_tun = 'Tin' in current_notes and 'Tun' in current_notes

    print(f"\nCopy 3:  Unique notes: {copy3_unique}/{len(copy3_notes)}")
    print(f"Current: Unique notes: {current_unique}/{len(current_notes)}")

    print(f"\nCopy 3:  Na+Ta variation: {'✅ Yes' if copy3_has_na_ta else '❌ No'}")
    print(f"Current: Na+Ta variation: {'✅ Yes' if current_has_na_ta else '❌ No'}")

    print(f"\nCopy 3:  Tin+Tun variation: {'✅ Yes' if copy3_has_tin_tun else '❌ No'}")
    print(f"Current: Tin+Tun variation: {'✅ Yes' if current_has_tin_tun else '❌ No'}")

    print(f"\n{'=' * 100}")
    print("🔊 AUDIO PLAYBACK")
    print(f"{'=' * 100}")

    # Play input phrase first
    print("\nPlaying INPUT phrase...")
    play_audio_terminal(test['notes'], test['durations'], label="🎵 INPUT (Original)")
    time.sleep(1)

    # Play Copy 3 output
    print("\nPlaying Copy 3 OUTPUT (no embeddings)...")
    play_audio_terminal(copy3_notes, copy3_durs, label="📊 COPY 3 OUTPUT (No Embedding Regularization)")
    time.sleep(1)

    # Play Current output
    print("\nPlaying Current OUTPUT (with embeddings)...")
    play_audio_terminal(current_notes, current_durs, label="✨ CURRENT OUTPUT (WITH Embedding Regularization)")

    print(f"\n{'=' * 100}")
    print("✅ Comparison complete!")
    print(f"{'=' * 100}")
    print("\n🎯 Key Observation:")
    print("   The model WITH embedding regularization should show:")
    print("   - More Na/Ta and Tin/Tun variations")
    print("   - Better musical flow")
    print("   - Less repetitive patterns")
    print(f"{'=' * 100}")

if __name__ == "__main__":
    main()
