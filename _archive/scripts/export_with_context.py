"""
Export audio with CONTEXT: Last bar of input + Generated continuation
This shows what the model "heard" and how it responded
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


def generate_with_context(audio_file, model_path, num_generate=32, temperature=1.0):
    """
    Run the full generation pipeline and return both:
    1. Last bar notes/durations (context)
    2. Generated notes/durations (continuation)
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
        return None, None, None, None

    # Step 3: Generate with LSTM
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

    # Return last bar (context) and generated (continuation)
    last_bar = bars[-1]
    context_notes = last_bar['notes']
    context_durations = last_bar['durations']

    return context_notes, context_durations, gen_note_labels, gen_durations_seconds


def export_with_context(output_dir='generated_audio_with_context'):
    """Export audio with context (last bar + generation)"""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print(" " * 20 + "EXPORTING AUDIO WITH CONTEXT (INPUT + GENERATION)")
    print("=" * 100)
    print(f"\nOutput directory: {output_dir}")
    print("\nEach file contains:")
    print("  1. Last bar of input (what the model 'heard')")
    print("  2. 🔹 GENERATION MARKER 🔹")
    print("  3. Generated continuation (what the model created)")
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

        # Generate with context
        context_notes, context_durations, gen_notes, gen_durations = generate_with_context(
            ektaal_file,
            model_info['path'],
            num_generate=32,
            temperature=1.0
        )

        if context_notes is None:
            print(f"   ❌ Generation failed")
            continue

        print(f"\n   📥 CONTEXT (last bar of input):")
        print(f"      Notes: {' '.join(context_notes)}")
        print(f"      Duration: {sum(context_durations):.2f}s")

        print(f"\n   📤 GENERATED (model continuation):")
        print(f"      Notes: {' '.join(gen_notes)}")
        print(f"      Duration: {sum(gen_durations):.2f}s")

        # Synthesize context audio
        print(f"\n   🔨 Synthesizing context audio...")
        context_audio, sr = synthesize_audio_seamless(context_notes, context_durations)

        # Synthesize generated audio
        print(f"   🔨 Synthesizing generated audio...")
        gen_audio, sr = synthesize_audio_seamless(gen_notes, gen_durations)

        if len(context_audio) == 0 or len(gen_audio) == 0:
            print(f"   ❌ Synthesis failed")
            continue

        # Add silence gap with marker tone to indicate transition
        # Create a short "beep" to mark the boundary
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

        # Concatenate: context + silence + marker + silence + generation
        full_audio = np.concatenate([
            context_audio,
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
        output_file = os.path.join(output_dir, f"{model_name}_WITH_CONTEXT.wav")
        sf.write(output_file, full_audio, sr)
        exported_files.append(output_file)

        total_duration = len(full_audio) / sr
        print(f"   ✅ Exported: {output_file}")
        print(f"   Total duration: {total_duration:.2f}s")

    print(f"\n{'=' * 100}")
    print("✅ EXPORT COMPLETE!")
    print(f"{'=' * 100}")
    print(f"\n📁 Exported {len(exported_files)} audio files with context:")
    for f in exported_files:
        print(f"   - {os.path.basename(f)}")

    print(f"\n💡 Listen for:")
    print(f"   🎵 First part: Last bar of input (context)")
    print(f"   📢 Beep tone: Marks the transition point")
    print(f"   ✨ Second part: Model's generated continuation")
    print()

    return exported_files


if __name__ == "__main__":
    files = export_with_context()

    # Automatically open the folder
    import subprocess
    subprocess.call(['open', 'generated_audio_with_context'])
