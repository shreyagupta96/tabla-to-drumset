"""
Export with BOTH tabla and drums in the same file
Format: Context (tabla) + Marker + Generation (tabla) + Marker + Generation (drums)
This allows direct comparison of the same generated sequence with both sound sets
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


def extract_swing_template(swing_result, input_durations):
    """Extract inter-onset intervals (IOIs) from swing-adjusted timeline"""
    adjusted_beats = swing_result['adjusted_beats']
    iois = np.diff(adjusted_beats)

    duration_clusters = {i: [] for i in range(5)}

    for duration, ioi in zip(input_durations, iois):
        bin_idx = quantize_duration(duration)
        duration_clusters[bin_idx].append(ioi)

    for i in range(5):
        if len(duration_clusters[i]) == 0:
            bin_midpoints = [0.15, 0.4, 0.65, 1.15, 2.0]
            duration_clusters[i] = [bin_midpoints[i]]

    return duration_clusters


def generate_swing_adjusted_durations(gen_duration_indices, duration_clusters, add_variation=True):
    """Map duration bins to actual IOIs from swing template"""
    swing_durations = []

    for bin_idx in gen_duration_indices:
        candidates = duration_clusters[bin_idx]

        if len(candidates) > 0:
            base_duration = np.random.choice(candidates)

            if add_variation:
                variation = np.random.uniform(0.95, 1.05)
                duration = base_duration * variation
            else:
                duration = base_duration
        else:
            bin_midpoints = [0.15, 0.4, 0.65, 1.15, 2.0]
            duration = bin_midpoints[bin_idx]

        swing_durations.append(duration)

    return swing_durations


def synthesize_audio_seamless(notes, durations, folder='tabla', sample_rate=44100, crossfade_ms=10):
    """
    Synthesize audio with EXACT swing timing preservation

    This function places each sample at the EXACT time specified by the swing durations,
    allowing natural overlaps and ensuring timing accuracy.

    Args:
        notes: List of note names
        durations: List of inter-onset intervals (IOIs) in seconds
        folder: Sample folder ('tabla' or drums path)
        sample_rate: Output sample rate
        crossfade_ms: Crossfade duration (for overlapping samples)
    """
    if len(notes) == 0:
        return np.array([]), sample_rate

    # Calculate exact onset times from durations (cumulative sum)
    onset_times = np.insert(np.cumsum(durations), 0, 0.0)  # [0, dur0, dur0+dur1, ...]

    # Load all samples
    samples = []
    for note in notes:
        audio_file = f"{folder}/{note}.wav"
        if os.path.exists(audio_file):
            try:
                audio_data, sr = librosa.load(audio_file, sr=sample_rate)
                samples.append(audio_data)
            except:
                samples.append(np.array([]))
        else:
            samples.append(np.array([]))

    # Calculate total length needed
    last_onset_samples = int(onset_times[-1] * sample_rate)
    max_sample_length = max([len(s) for s in samples] + [0])
    total_length = last_onset_samples + max_sample_length

    # Create output buffer
    full_audio = np.zeros(total_length)

    # Place each sample at its exact onset time
    crossfade_samples = int(sample_rate * crossfade_ms / 1000)

    for i, (sample, onset_time) in enumerate(zip(samples, onset_times)):
        if len(sample) == 0:
            continue

        onset_sample = int(onset_time * sample_rate)

        # Check if we overlap with previous sample
        if i > 0 and onset_sample < int(onset_times[i-1] * sample_rate) + len(samples[i-1]):
            # We have overlap - apply crossfade
            prev_onset = int(onset_times[i-1] * sample_rate)
            prev_end = prev_onset + len(samples[i-1])

            overlap_start = onset_sample
            overlap_end = min(onset_sample + len(sample), prev_end)
            overlap_length = overlap_end - overlap_start

            if overlap_length > 0:
                # Create crossfade for overlapping region
                fade_length = min(crossfade_samples, overlap_length)
                fade_out = np.linspace(1, 0, fade_length)
                fade_in = np.linspace(0, 1, fade_length)

                # Apply crossfade to overlapping region
                overlap_offset_prev = overlap_start - prev_onset
                overlap_offset_curr = 0

                # Fade out the previous sample
                full_audio[overlap_start:overlap_start+fade_length] *= fade_out
                # Fade in the current sample
                crossfaded = sample[overlap_offset_curr:overlap_offset_curr+fade_length] * fade_in
                full_audio[overlap_start:overlap_start+fade_length] += crossfaded

                # Add rest of current sample (non-overlapping part)
                rest_start = overlap_start + fade_length
                rest_samples = sample[fade_length:]
                full_audio[rest_start:rest_start+len(rest_samples)] += rest_samples
            else:
                # No significant overlap, just add
                full_audio[onset_sample:onset_sample+len(sample)] += sample
        else:
            # No overlap, just place the sample
            full_audio[onset_sample:onset_sample+len(sample)] += sample

    # Trim silence at end
    # Find last non-zero sample
    non_zero = np.where(np.abs(full_audio) > 1e-6)[0]
    if len(non_zero) > 0:
        full_audio = full_audio[:non_zero[-1] + int(sample_rate * 0.1)]  # Add 100ms tail

    # Normalize
    max_val = np.abs(full_audio).max()
    if max_val > 0:
        full_audio = full_audio / max_val * 0.9

    return full_audio, sample_rate


def generate_with_swing(audio_file, model_path, num_generate=32, temperature=1.0):
    """Generate with swing-adjusted timing"""
    # Step 1: Classify notes
    cnn_model = ConvNet(input_channels=13, num_classes=12)
    cnn_model.load_state_dict(torch.load('ConvNet_SNFPR_model.pth'))
    cnn_model.eval()

    notes, durations, onset_samples = classify_tabla_file(audio_file, cnn_model, target_length=72000)

    # Step 2: Meter detection and segmentation
    meter_result = hybrid_meter(audio_file)
    meter = meter_result.get('final_meter')
    bar_start_samples = meter_result.get('bar_start_samples', [])
    swing_result = meter_result.get('swing_result')

    if meter == 16:
        taal_id = 0
    elif meter == 12:
        taal_id = 1
    else:
        taal_id = 2

    bars = segment_notes_by_bars(notes, durations, onset_samples, bar_start_samples)

    if len(bars) < 3:
        return None, None, None, None, None

    # Step 3: Extract swing template
    duration_clusters = extract_swing_template(swing_result, durations)

    swing_stats = {
        'total_clusters': sum(len(v) for v in duration_clusters.values()),
        'cluster_sizes': {f'bin_{i}': len(v) for i, v in duration_clusters.items()},
        'cluster_means': {f'bin_{i}': np.mean(v) if len(v) > 0 else 0 for i, v in duration_clusters.items()}
    }

    # Step 4: Generate with LSTM
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
    gen_durations_swing = generate_swing_adjusted_durations(gen_duration_indices, duration_clusters)

    # Get context (last bar)
    last_bar = bars[-1]
    context_notes = last_bar['notes']
    context_durations = last_bar['durations']

    return context_notes, context_durations, gen_note_labels, gen_durations_swing, swing_stats


def create_marker_beep(sample_rate=44100, duration_s=0.3, freq=800):
    """Create transition marker beep"""
    marker_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, marker_samples)

    # Create beep with fade in/out
    marker_beep = 0.3 * np.sin(2 * np.pi * freq * t)
    fade_samples = int(sample_rate * 0.05)  # 50ms fade
    marker_beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
    marker_beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    return marker_beep


def export_tabla_and_drums(output_dir='generated_tabla_and_drums'):
    """Export audio with BOTH tabla and drums for comparison"""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print(" " * 20 + "EXPORTING AUDIO WITH TABLA AND DRUMS")
    print("=" * 100)
    print(f"\nOutput directory: {output_dir}")
    print("\nFile structure:")
    print("  1. 🎵 CONTEXT: Last bar of input (tabla)")
    print("  2. 📢 MARKER: Transition beep")
    print("  3. 🎵 GENERATION: Model output (tabla)")
    print("  4. 📢 MARKER: Transition beep")
    print("  5. 🥁 GENERATION: Same output (drums)")
    print("\nThis allows direct A/B comparison of tabla vs drums with identical timing!")
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
        print(f"🎼 {model_name}")
        print(f"{'=' * 100}")
        print(f"   Description: {model_info['description']}")
        print(f"   Val Loss: {model_info['val_loss']}")

        # Generate with swing
        context_notes, context_durations, gen_notes, gen_durations, swing_stats = generate_with_swing(
            ektaal_file,
            model_info['path'],
            num_generate=32,
            temperature=1.0
        )

        if context_notes is None:
            print(f"   ❌ Generation failed")
            continue

        print(f"\n   📥 CONTEXT (tabla):")
        print(f"      Notes: {' '.join(context_notes[:10])}...")
        print(f"      Duration: {sum(context_durations):.2f}s")

        print(f"\n   📤 GENERATED:")
        print(f"      Notes: {' '.join(gen_notes[:10])}...")
        print(f"      Duration: {sum(gen_durations):.2f}s")

        # Synthesize all sections
        print(f"\n   🔨 Synthesizing context (tabla)...")
        context_audio, sr = synthesize_audio_seamless(context_notes, context_durations, folder='tabla')

        print(f"   🔨 Synthesizing generation (tabla)...")
        gen_tabla_audio, sr = synthesize_audio_seamless(gen_notes, gen_durations, folder='tabla')

        print(f"   🔨 Synthesizing generation (drums)...")
        gen_drums_audio, sr = synthesize_audio_seamless(gen_notes, gen_durations, folder='/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/drums')

        if len(context_audio) == 0 or len(gen_tabla_audio) == 0 or len(gen_drums_audio) == 0:
            print(f"   ❌ Synthesis failed")
            continue

        # Create markers and silence
        marker_beep = create_marker_beep(sr)
        silence = np.zeros(int(sr * 0.1))  # 100ms silence

        # Concatenate: context + marker + tabla gen + marker + drums gen
        full_audio = np.concatenate([
            context_audio,
            silence,
            marker_beep,
            silence,
            gen_tabla_audio,
            silence,
            marker_beep,
            silence,
            gen_drums_audio
        ])

        # Normalize
        max_val = np.abs(full_audio).max()
        if max_val > 0:
            full_audio = full_audio / max_val * 0.9

        # Save
        output_file = os.path.join(output_dir, f"{model_name}_TABLA_AND_DRUMS.wav")
        sf.write(output_file, full_audio, sr)
        exported_files.append(output_file)

        total_duration = len(full_audio) / sr
        context_duration = len(context_audio) / sr
        gen_duration = len(gen_tabla_audio) / sr

        print(f"   ✅ Exported: {output_file}")
        print(f"   Total duration: {total_duration:.2f}s")
        print(f"      - Context: {context_duration:.2f}s")
        print(f"      - Generation (tabla): {gen_duration:.2f}s")
        print(f"      - Generation (drums): {len(gen_drums_audio)/sr:.2f}s")

    print(f"\n{'=' * 100}")
    print("✅ EXPORT COMPLETE!")
    print(f"{'=' * 100}")
    print(f"\n📁 Exported {len(exported_files)} audio files with tabla AND drums:")
    for f in exported_files:
        print(f"   - {os.path.basename(f)}")

    print(f"\n💡 Listen for:")
    print(f"   🎵 Section 1: Tabla context (input)")
    print(f"   📢 Beep: Transition marker")
    print(f"   🎵 Section 2: Tabla generation (same timing as drums)")
    print(f"   📢 Beep: Transition marker")
    print(f"   🥁 Section 3: Drums generation (IDENTICAL timing to tabla)")
    print(f"\n   You can directly compare how the same generated sequence sounds with both instruments!")
    print()

    return exported_files


if __name__ == "__main__":
    files = export_tabla_and_drums()

    # Automatically open the folder
    import subprocess
    subprocess.call(['open', 'generated_tabla_and_drums'])
