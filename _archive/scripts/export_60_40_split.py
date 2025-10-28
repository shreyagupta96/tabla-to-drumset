"""
Export audio with 60% INPUT + 40% GENERATION split
Shows more context from the input before the generated continuation
"""

import os
import numpy as np
import soundfile as sf
import librosa
import torch
import sys

# Add meter detection path
sys.path.insert(0, "/Users/shreyagupta/Desktop/Research MFP/B264026_Masters_Final_Project_SG/Submission/Code/HybridApproach")

from meter_conditional_lstm import create_model
from hybrid_meter_pipeline import hybrid_meter
from batch_classify_long_files import classify_tabla_file, ConvNet
from segment_by_bars import segment_notes_by_bars

note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]
note_to_idx = {note: idx for idx, note in enumerate(note_labels)}

DURATION_BINS = [0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]

def quantize_duration(duration):
    """Quantize duration into bins"""
    for i, threshold in enumerate(DURATION_BINS[1:]):
        if duration < threshold:
            return i
    return len(DURATION_BINS) - 2

def duration_bins_to_seconds(duration_bins):
    """Convert duration bin indices to actual seconds"""
    bin_midpoints = [0.15, 0.4, 0.65, 1.15, 2.0]
    return [bin_midpoints[idx] for idx in duration_bins]

def synthesize_audio_seamless(notes, durations, sample_rate=44100, crossfade_ms=10):
    """
    Synthesize audio with SEAMLESS note transitions
    """
    tabla_folder = 'tabla'
    crossfade_samples = int(sample_rate * crossfade_ms / 1000)

    audio_segments = []

    for i, (note, duration) in enumerate(zip(notes, durations)):
        audio_file = f"{tabla_folder}/{note}.wav"

        if not os.path.exists(audio_file):
            continue

        try:
            audio_data, sr = librosa.load(audio_file, sr=sample_rate)
            target_samples = int(sample_rate * duration)

            if len(audio_data) > target_samples:
                audio_data = audio_data[:target_samples]

            # Apply fade out to end
            fade_samples = min(crossfade_samples, len(audio_data) // 4)
            if fade_samples > 0:
                fade_out = np.linspace(1, 0, fade_samples)
                audio_data[-fade_samples:] *= fade_out

            audio_segments.append(audio_data)

        except Exception as e:
            pass

    if not audio_segments:
        return np.array([]), sample_rate

    # Concatenate with crossfade
    full_audio = audio_segments[0]

    for segment in audio_segments[1:]:
        overlap_samples = min(crossfade_samples, len(full_audio), len(segment))

        if overlap_samples > 0:
            fade_out = np.linspace(1, 0, overlap_samples)
            fade_in = np.linspace(0, 1, overlap_samples)

            full_audio[-overlap_samples:] *= fade_out
            full_audio[-overlap_samples:] += segment[:overlap_samples] * fade_in
            full_audio = np.concatenate([full_audio, segment[overlap_samples:]])
        else:
            full_audio = np.concatenate([full_audio, segment])

    # Normalize
    max_val = np.abs(full_audio).max()
    if max_val > 0:
        full_audio = full_audio / max_val * 0.9

    return full_audio, sample_rate


def generate_60_40_split(audio_file, model_path, target_bars=10, temperature=1.0):
    """
    Generate with 60% input + 40% generation split

    Args:
        audio_file: Input tabla audio
        model_path: Path to trained LSTM model
        target_bars: Target number of bars total (input + generated)
        temperature: Sampling temperature

    Returns:
        input_notes, input_durations, gen_notes, gen_durations, num_input_bars, num_gen_bars
    """
    # Step 1: Classify notes
    cnn_model = ConvNet(input_channels=13, num_classes=12)
    cnn_model.load_state_dict(torch.load('ConvNet_SNFPR_model.pth'))
    cnn_model.eval()

    notes, durations, onset_samples = classify_tabla_file(audio_file, cnn_model, target_length=72000)

    # Step 2: Meter detection and segmentation
    meter_result = hybrid_meter(audio_file)
    meter = meter_result.get('final_meter')
    bar_start_samples = meter_result.get('bar_start_samples', [])

    if meter == 16:
        taal_id = 0
    elif meter == 12:
        taal_id = 1
    else:
        taal_id = 2

    bars = segment_notes_by_bars(notes, durations, onset_samples, bar_start_samples)

    if len(bars) < 3:
        return None, None, None, None, 0, 0

    # Step 3: Calculate 60-40 split
    # 60% of target_bars should be from input, 40% should be generated
    num_input_bars = int(target_bars * 0.6)  # e.g., 6 bars
    num_gen_bars = target_bars - num_input_bars  # e.g., 4 bars

    # Estimate notes per bar from the input
    avg_notes_per_bar = np.mean([len(bar['notes']) for bar in bars])
    num_generate = int(num_gen_bars * avg_notes_per_bar)  # Estimate how many notes to generate

    # Make sure we have enough input bars
    num_input_bars = min(num_input_bars, len(bars))

    # Step 4: Get input bars (last N bars for context)
    input_bars = bars[-num_input_bars:]
    input_notes = []
    input_durations = []

    for bar in input_bars:
        input_notes.extend(bar['notes'])
        input_durations.extend(bar['durations'])

    # Step 5: Generate with LSTM (use last 2 bars as seed)
    checkpoint = torch.load(model_path, map_location='cpu')
    metadata = checkpoint['metadata']

    model, _ = create_model(
        vocab_size=metadata['num_classes'],
        num_duration_bins=metadata['num_duration_bins'],
        num_taals=len(metadata['taal_mapping']),
        hidden_size=checkpoint['hyperparameters']['hidden_size'],
        num_layers=checkpoint['hyperparameters']['num_layers'],
        dropout=checkpoint['hyperparameters']['dropout'],
        note_labels=metadata['note_labels']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Use last 2 bars as seed
    seed_bars = bars[-2:]
    seed_notes = []
    seed_durations = []

    for bar in seed_bars:
        seed_notes.extend([note_to_idx[n] for n in bar['notes']])
        seed_durations.extend([quantize_duration(d) for d in bar['durations']])

    # Generate
    gen_note_indices, gen_duration_indices = model.generate(
        seed_notes=seed_notes,
        seed_durations=seed_durations,
        taal_id=taal_id,
        num_generate=num_generate,
        temperature=temperature,
        device='cpu'
    )

    gen_note_labels = [note_labels[idx] for idx in gen_note_indices]
    gen_durations_seconds = duration_bins_to_seconds(gen_duration_indices)

    return input_notes, input_durations, gen_note_labels, gen_durations_seconds, num_input_bars, num_gen_bars


def export_60_40_split(output_dir='generated_audio_60_40_split', target_bars=10):
    """Export audio with 60% input + 40% generation"""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print(" " * 25 + "EXPORTING AUDIO: 60% INPUT + 40% GENERATION")
    print("=" * 100)
    print(f"\nOutput directory: {output_dir}")
    print(f"Target total bars: {target_bars}")
    print(f"  → Input bars (60%): ~{int(target_bars * 0.6)} bars")
    print(f"  → Generated bars (40%): ~{int(target_bars * 0.4)} bars")
    print("\nEach file contains:")
    print("  1. Last N bars of input (60% of content)")
    print("  2. 🔹 GENERATION MARKER 🔹")
    print("  3. Generated continuation (40% of content)")
    print()

    ektaal_file = "/Users/shreyagupta/Desktop/AI_Research_Data/Tabla_files/Ektaal.wav"

    models = {
        'Model_A_Original_Reg': {
            'path': 'models/best_bar_aware_lstm.pth',
            'description': 'Original Data + Regularization',
            'val_loss': 3.0473
        },
        'Model_B_Original_NoReg': {
            'path': 'models/best_bar_aware_lstm_no_reg.pth',
            'description': 'Original Data + No Regularization',
            'val_loss': 3.0094
        },
        'Model_C_Corrected_Reg': {
            'path': 'models/best_bar_aware_lstm_corrected.pth',
            'description': 'Corrected Data + Regularization',
            'val_loss': 3.0360
        },
        'Model_D_Corrected_NoReg': {
            'path': 'models/best_bar_aware_lstm_corrected_no_reg.pth',
            'description': 'Corrected Data + No Regularization',
            'val_loss': 3.0375
        }
    }

    exported_files = []

    for model_name, model_info in models.items():
        print(f"\n{'=' * 100}")
        print(f"🎵 {model_name}")
        print(f"{'=' * 100}")
        print(f"   Description: {model_info['description']}")
        print(f"   Val Loss: {model_info['val_loss']}")

        # Generate with 60-40 split
        input_notes, input_durations, gen_notes, gen_durations, num_input_bars, num_gen_bars = generate_60_40_split(
            ektaal_file,
            model_info['path'],
            target_bars=target_bars,
            temperature=1.0
        )

        if input_notes is None:
            print(f"   ❌ Generation failed")
            continue

        print(f"\n   📥 INPUT ({num_input_bars} bars - 60% of content):")
        print(f"      Notes: {' '.join(input_notes)}")
        print(f"      Total notes: {len(input_notes)}")
        print(f"      Duration: {sum(input_durations):.2f}s")

        print(f"\n   📤 GENERATED ({num_gen_bars} bars estimate - 40% of content):")
        print(f"      Notes: {' '.join(gen_notes)}")
        print(f"      Total notes: {len(gen_notes)}")
        print(f"      Duration: {sum(gen_durations):.2f}s")

        # Synthesize input audio
        print(f"\n   🔨 Synthesizing input audio...")
        input_audio, sr = synthesize_audio_seamless(input_notes, input_durations)

        # Synthesize generated audio
        print(f"   🔨 Synthesizing generated audio...")
        gen_audio, sr = synthesize_audio_seamless(gen_notes, gen_durations)

        if len(input_audio) == 0 or len(gen_audio) == 0:
            print(f"   ❌ Synthesis failed")
            continue

        # Add silence gap with marker tone
        marker_duration_s = 0.3
        marker_freq = 800  # Hz
        marker_samples = int(sr * marker_duration_s)
        t = np.linspace(0, marker_duration_s, marker_samples)

        # Create beep with fade in/out
        marker_beep = 0.3 * np.sin(2 * np.pi * marker_freq * t)
        fade_samples = int(sr * 0.05)  # 50ms fade
        marker_beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
        marker_beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        # Small silence before and after marker
        silence = np.zeros(int(sr * 0.1))  # 100ms silence

        # Concatenate: input + silence + marker + silence + generation
        full_audio = np.concatenate([
            input_audio,
            silence,
            marker_beep,
            silence,
            gen_audio
        ])

        # Normalize
        max_val = np.abs(full_audio).max()
        if max_val > 0:
            full_audio = full_audio / max_val * 0.9

        # Save audio
        output_file = os.path.join(output_dir, f"{model_name}_60_40_SPLIT.wav")
        sf.write(output_file, full_audio, sr)
        exported_files.append(output_file)

        total_duration = len(full_audio) / sr
        input_percentage = (len(input_audio) / len(full_audio)) * 100
        gen_percentage = (len(gen_audio) / len(full_audio)) * 100

        print(f"   ✅ Exported: {output_file}")
        print(f"   Total duration: {total_duration:.2f}s")
        print(f"   Actual split: {input_percentage:.1f}% input / {gen_percentage:.1f}% generated")

    print(f"\n{'=' * 100}")
    print("✅ EXPORT COMPLETE!")
    print(f"{'=' * 100}")
    print(f"\n📁 Exported {len(exported_files)} audio files with 60-40 split:")
    for f in exported_files:
        print(f"   - {os.path.basename(f)}")

    print(f"\n💡 Listen for:")
    print(f"   🎵 First part (60%): Last {int(target_bars * 0.6)} bars of input")
    print(f"   📢 Beep tone: Marks the transition point")
    print(f"   ✨ Second part (40%): Model's generated continuation")
    print()

    return exported_files


if __name__ == "__main__":
    files = export_60_40_split(target_bars=10)

    # Automatically open the folder
    import subprocess
    subprocess.call(['open', 'generated_audio_60_40_split'])
