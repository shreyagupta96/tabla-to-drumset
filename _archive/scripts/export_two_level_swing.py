"""
Two-Level Swing Timing System
==============================
MACRO: Swing-adjusted beat timing (groove/feel)
MICRO: Duration-pattern-aware subdivisions within beats (rhythmic vocabulary)

This preserves both the natural swing feel AND the rhythmic relationships
between notes (e.g., "Ti Re Ki T" stays fast within one beat).
"""

import os
import numpy as np
import soundfile as sf
import librosa
import torch
import sys
from collections import defaultdict

sys.path.insert(0, "/Users/shreyagupta/Desktop/Research MFP/B264026_Masters_Final_Project_SG/Submission/Code/HybridApproach")

from meter_conditional_lstm import create_model
from hybrid_meter_pipeline import hybrid_meter
from batch_classify_long_files import classify_tabla_file, ConvNet
from segment_by_bars import segment_notes_by_bars

note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]
note_to_idx = {note: idx for idx, note in enumerate(note_labels)}

DURATION_BINS = [0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]
BIN_MIDPOINTS = [0.15, 0.4, 0.65, 1.15, 2.0]

def quantize_duration(duration):
    """Quantize duration into bins"""
    for i, threshold in enumerate(DURATION_BINS[1:]):
        if duration < threshold:
            return i
    return len(DURATION_BINS) - 2


def extract_beat_patterns(swing_result, notes, durations, onset_samples, sample_rate):
    """
    Extract beat-level patterns from input with micro-timing

    Returns:
        beat_patterns: dict mapping (num_notes, duration_tuple) -> list of patterns
        swing_beat_iois: list of beat IOIs from swing template
    """
    adjusted_beats = swing_result['adjusted_beats']

    # Convert beat times to samples
    adjusted_beat_samples = (np.array(adjusted_beats) * sample_rate).astype(int)

    # Swing beat IOIs (MACRO timing)
    swing_beat_iois = np.diff(adjusted_beats)

    # Organize patterns by beat
    beat_patterns = defaultdict(list)

    # Convert to numpy arrays for indexing
    onset_samples = np.array(onset_samples)
    notes = np.array(notes)
    durations = np.array(durations)

    for i in range(len(adjusted_beat_samples) - 1):
        beat_start = adjusted_beat_samples[i]
        beat_end = adjusted_beat_samples[i + 1]
        beat_duration = swing_beat_iois[i]

        # Find notes within this beat
        beat_note_mask = (onset_samples >= beat_start) & (onset_samples < beat_end)
        beat_notes = notes[beat_note_mask]
        beat_onsets = onset_samples[beat_note_mask]
        beat_durations = durations[beat_note_mask]

        if len(beat_notes) == 0:
            continue

        # Calculate relative timing within beat (MICRO timing)
        beat_onsets_relative = (beat_onsets - beat_start) / sample_rate
        beat_iois = np.diff(np.append(beat_onsets_relative, beat_duration))

        # Normalize to proportions
        relative_iois = beat_iois / beat_duration

        # Get duration bins
        duration_bins = tuple([quantize_duration(d) for d in beat_durations])

        # Store pattern
        pattern_key = (len(beat_notes), duration_bins)
        pattern = {
            'beat_duration': beat_duration,
            'num_notes': len(beat_notes),
            'duration_bins': duration_bins,
            'actual_iois': beat_iois,
            'relative_iois': relative_iois
        }

        beat_patterns[pattern_key].append(pattern)

    print(f"\n📊 Extracted Beat Pattern Library:")
    print(f"   Total unique patterns: {len(beat_patterns)}")
    for key, patterns in sorted(beat_patterns.items(), key=lambda x: -len(x[1]))[:10]:
        num_notes, bins = key
        print(f"   {num_notes} notes {bins}: {len(patterns)} examples")

    return beat_patterns, swing_beat_iois


def group_notes_into_beats(gen_duration_indices, avg_beat_duration=0.6):
    """
    Group generated notes into beat chunks based on duration bins

    Args:
        gen_duration_indices: Model's predicted duration bins
        avg_beat_duration: Average beat duration for estimation

    Returns:
        List of beat chunks (indices into gen_duration_indices)
    """
    beats = []
    current_beat = []
    accumulated_duration = 0.0

    for i, bin_idx in enumerate(gen_duration_indices):
        current_beat.append(i)
        accumulated_duration += BIN_MIDPOINTS[bin_idx]

        # Check if we've accumulated ~1 beat worth
        if accumulated_duration >= avg_beat_duration * 0.8:  # Allow some flexibility
            beats.append(current_beat)
            current_beat = []
            accumulated_duration = 0.0

    # Add remaining notes as final beat
    if current_beat:
        beats.append(current_beat)

    return beats


def match_and_apply_pattern(gen_chunk_bins, beat_patterns, swing_beat_duration, pattern_stats):
    """
    Match generated beat chunk to input pattern and apply swing timing

    Args:
        gen_chunk_bins: Tuple of duration bins for this beat chunk
        beat_patterns: Library of input beat patterns
        swing_beat_duration: Swing-adjusted duration for this beat
        pattern_stats: Dict to track statistics

    Returns:
        List of actual durations for this beat chunk
    """
    num_notes = len(gen_chunk_bins)
    pattern_key = (num_notes, gen_chunk_bins)

    # Try exact match first
    if pattern_key in beat_patterns:
        candidates = beat_patterns[pattern_key]
        chosen_idx = np.random.randint(0, len(candidates))
        chosen = candidates[chosen_idx]
        pattern_stats['exact_match'] += 1
    else:
        # Fallback 1: Match by number of notes only
        num_notes_matches = [k for k in beat_patterns.keys() if k[0] == num_notes]

        if num_notes_matches:
            # Randomly choose one of the matching keys
            fallback_key = num_notes_matches[np.random.randint(0, len(num_notes_matches))]
            candidates = beat_patterns[fallback_key]
            chosen_idx = np.random.randint(0, len(candidates))
            chosen = candidates[chosen_idx]
            pattern_stats['num_notes_match'] += 1
        else:
            # Fallback 2: Use bin midpoints as equal subdivisions
            relative_iois = np.array([BIN_MIDPOINTS[b] for b in gen_chunk_bins])
            relative_iois = relative_iois / relative_iois.sum()
            chosen = {
                'relative_iois': relative_iois
            }
            pattern_stats['synthesized'] += 1

    # Scale to swing-adjusted beat duration
    actual_durations = chosen['relative_iois'] * swing_beat_duration

    return actual_durations


def generate_with_two_level_swing(audio_file, model_path, num_generate=32, temperature=1.0):
    """
    Generate with two-level swing timing:
    - MACRO: Swing-adjusted beat spacing
    - MICRO: Pattern-matched subdivision timing
    """
    # Step 1: Classify notes
    print("📊 Step 1: Classifying input notes...")
    cnn_model = ConvNet(input_channels=13, num_classes=12)
    cnn_model.load_state_dict(torch.load('ConvNet_SNFPR_model.pth'))
    cnn_model.eval()

    notes, durations, onset_samples = classify_tabla_file(audio_file, cnn_model, target_length=72000)

    # Step 2: Meter detection and swing analysis
    print("📊 Step 2: Analyzing meter and swing...")
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
        return None

    # Step 3: Extract beat patterns from input
    print("📊 Step 3: Extracting beat patterns...")
    y, sr = librosa.load(audio_file, sr=44100)
    beat_patterns, swing_beat_iois = extract_beat_patterns(
        swing_result, notes, durations, onset_samples, sr
    )

    # Step 4: Load model and generate
    print("📊 Step 4: Generating with LSTM...")
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

    # Step 5: Apply two-level timing
    print("📊 Step 5: Applying two-level swing timing...")

    # Group into beats
    avg_beat_duration = np.mean(swing_beat_iois)
    beat_chunks = group_notes_into_beats(gen_duration_indices, avg_beat_duration)

    print(f"   Grouped {len(gen_note_labels)} notes into {len(beat_chunks)} beat chunks")

    # Match and apply patterns
    pattern_stats = {'exact_match': 0, 'num_notes_match': 0, 'synthesized': 0}
    gen_durations_final = []

    for beat_idx, chunk_indices in enumerate(beat_chunks):
        chunk_bins = tuple([gen_duration_indices[i] for i in chunk_indices])

        # Get swing beat duration (cycle through if needed)
        swing_beat_idx = beat_idx % len(swing_beat_iois)
        swing_duration = swing_beat_iois[swing_beat_idx]

        # Match and apply
        beat_durations = match_and_apply_pattern(
            chunk_bins, beat_patterns, swing_duration, pattern_stats
        )

        gen_durations_final.extend(beat_durations)

    print(f"   Pattern matching stats:")
    print(f"      Exact matches: {pattern_stats['exact_match']}")
    print(f"      Note-count matches: {pattern_stats['num_notes_match']}")
    print(f"      Synthesized: {pattern_stats['synthesized']}")

    # Get context (last bar)
    last_bar = bars[-1]
    context_notes = last_bar['notes']
    context_durations = last_bar['durations']

    return context_notes, context_durations, gen_note_labels, gen_durations_final, pattern_stats


def synthesize_audio_seamless(notes, durations, folder='tabla', sample_rate=44100, fade_ms=5):
    """
    Synthesize audio with EXACT timing preservation and smooth fades

    Args:
        notes: List of note names
        durations: List of IOIs in seconds
        folder: Sample folder
        sample_rate: Output sample rate
        fade_ms: Fade duration in milliseconds for smoothing
    """
    if len(notes) == 0:
        return np.array([]), sample_rate

    # Calculate exact onset times
    onset_times = np.insert(np.cumsum(durations), 0, 0.0)

    # Load all samples and apply gentle fade-in to avoid clicks
    samples = []
    fade_samples = int(sample_rate * fade_ms / 1000)

    for note in notes:
        audio_file = f"{folder}/{note}.wav"
        if os.path.exists(audio_file):
            try:
                audio_data, sr = librosa.load(audio_file, sr=sample_rate)

                # Apply gentle fade-in to avoid attack clicks
                fade_in_length = min(fade_samples, len(audio_data) // 10)  # Max 10% of sample
                if fade_in_length > 0:
                    # Use cosine fade for smoother transition
                    fade_in = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, fade_in_length)))
                    audio_data[:fade_in_length] *= fade_in

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

    # Place each sample at its exact onset time with intelligent fading
    for i, (sample, onset_time) in enumerate(zip(samples, onset_times)):
        if len(sample) == 0:
            continue

        onset_sample = int(onset_time * sample_rate)
        next_onset = int(onset_times[i+1] * sample_rate) if i < len(onset_times) - 1 else total_length

        # Check if we need to truncate this sample before next onset
        available_space = next_onset - onset_sample

        if len(sample) > available_space:
            # Need to truncate - apply fade-out
            fade_out_length = min(fade_samples, available_space // 4, len(sample) // 4)
            if fade_out_length > 0:
                # Cosine fade-out
                fade_out = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, fade_out_length)))
                sample = sample[:available_space].copy()
                sample[-fade_out_length:] *= fade_out
            else:
                sample = sample[:available_space]
        else:
            # Sample fits - apply gentle fade-out at natural ending
            fade_out_length = min(fade_samples, len(sample) // 5)
            if fade_out_length > 0:
                fade_out = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, fade_out_length)))
                sample = sample.copy()
                sample[-fade_out_length:] *= fade_out

        # Check for overlap with previous sample
        if i > 0:
            prev_onset = int(onset_times[i-1] * sample_rate)
            prev_sample_len = len(samples[i-1])
            prev_end = prev_onset + min(prev_sample_len, available_space)

            if onset_sample < prev_end:
                # Overlap detected - use crossfade
                overlap_length = prev_end - onset_sample
                crossfade_length = min(int(sample_rate * fade_ms / 1000), overlap_length, len(sample) // 2)

                if crossfade_length > 0:
                    # Cosine crossfade
                    fade_out_cf = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, crossfade_length)))
                    fade_in_cf = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, crossfade_length)))

                    # Apply crossfade
                    full_audio[onset_sample:onset_sample+crossfade_length] *= fade_out_cf
                    full_audio[onset_sample:onset_sample+crossfade_length] += sample[:crossfade_length] * fade_in_cf

                    # Add rest of sample
                    if len(sample) > crossfade_length:
                        full_audio[onset_sample+crossfade_length:onset_sample+len(sample)] += sample[crossfade_length:]
                else:
                    full_audio[onset_sample:onset_sample+len(sample)] += sample
            else:
                # No overlap
                full_audio[onset_sample:onset_sample+len(sample)] += sample
        else:
            # First sample
            full_audio[onset_sample:onset_sample+len(sample)] += sample

    # Trim silence at end
    non_zero = np.where(np.abs(full_audio) > 1e-6)[0]
    if len(non_zero) > 0:
        full_audio = full_audio[:non_zero[-1] + int(sample_rate * 0.1)]

    # Apply gentle compression to avoid clipping
    max_val = np.abs(full_audio).max()
    if max_val > 0:
        # Soft clipping for more musical sound
        full_audio = full_audio / max_val
        full_audio = np.tanh(full_audio * 0.9) * 0.9

    return full_audio, sample_rate


def export_two_level_swing(output_dir='generated_two_level_swing'):
    """Export audio with two-level swing timing - BOTH tabla and drums"""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print(" " * 20 + "TWO-LEVEL SWING TIMING SYSTEM")
    print("=" * 100)
    print(f"\nOutput directory: {output_dir}")
    print("\nFile structure:")
    print("  1. 🎵 CONTEXT: Last bar of input (tabla)")
    print("  2. 📢 MARKER: Transition beep")
    print("  3. 🎵 GENERATION: Model output (tabla)")
    print("  4. 📢 MARKER: Transition beep")
    print("  5. 🥁 GENERATION: Same output (drums)")
    print("\nFeatures:")
    print("  ✓ MACRO: Swing-adjusted beat timing (groove/feel)")
    print("  ✓ MICRO: Pattern-matched subdivisions (rhythmic vocabulary)")
    print("  ✓ Preserves both swing feel AND note relationships")
    print("  ✓ Fast patterns stay fast, slow patterns stay slow")
    print("  ✓ Same timing for tabla AND drums - direct A/B comparison!")
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

        # Generate with two-level swing
        result = generate_with_two_level_swing(
            ektaal_file,
            model_info['path'],
            num_generate=32,
            temperature=1.0
        )

        if result is None:
            print(f"   ❌ Generation failed")
            continue

        context_notes, context_durations, gen_notes, gen_durations, pattern_stats = result

        print(f"\n   📥 CONTEXT (last bar):")
        print(f"      Notes: {' '.join(context_notes[:10])}...")
        print(f"      Duration: {sum(context_durations):.2f}s")

        print(f"\n   📤 GENERATED (two-level swing):")
        print(f"      Notes: {' '.join(gen_notes[:10])}...")
        print(f"      Duration: {sum(gen_durations):.2f}s")
        print(f"      Duration range: {min(gen_durations):.3f}s - {max(gen_durations):.3f}s")

        # Synthesize
        print(f"\n   🔨 Synthesizing context (tabla)...")
        context_audio, sr = synthesize_audio_seamless(context_notes, context_durations, folder='tabla')

        print(f"   🔨 Synthesizing generation (tabla)...")
        gen_tabla_audio, sr = synthesize_audio_seamless(gen_notes, gen_durations, folder='tabla')

        print(f"   🔨 Synthesizing generation (drums)...")
        gen_drums_audio, sr = synthesize_audio_seamless(gen_notes, gen_durations, folder='/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/drums')

        if len(context_audio) == 0 or len(gen_tabla_audio) == 0 or len(gen_drums_audio) == 0:
            print(f"   ❌ Synthesis failed")
            continue

        # Create marker
        marker_duration_s = 0.3
        marker_freq = 800
        marker_samples = int(sr * marker_duration_s)
        t = np.linspace(0, marker_duration_s, marker_samples)

        marker_beep = 0.3 * np.sin(2 * np.pi * marker_freq * t)
        fade_samples = int(sr * 0.05)
        marker_beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
        marker_beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        silence = np.zeros(int(sr * 0.1))

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
        output_file = os.path.join(output_dir, f"{model_name}_TWO_LEVEL_SWING_TABLA_AND_DRUMS.wav")
        sf.write(output_file, full_audio, sr)
        exported_files.append(output_file)

        total_duration = len(full_audio) / sr
        context_duration = len(context_audio) / sr
        tabla_duration = len(gen_tabla_audio) / sr
        drums_duration = len(gen_drums_audio) / sr

        print(f"   ✅ Exported: {output_file}")
        print(f"   Total duration: {total_duration:.2f}s")
        print(f"      - Context (tabla): {context_duration:.2f}s")
        print(f"      - Generation (tabla): {tabla_duration:.2f}s")
        print(f"      - Generation (drums): {drums_duration:.2f}s")

    print(f"\n{'=' * 100}")
    print("✅ EXPORT COMPLETE!")
    print(f"{'=' * 100}")
    print(f"\n📁 Exported {len(exported_files)} audio files with TWO-LEVEL SWING timing:")
    for f in exported_files:
        print(f"   - {os.path.basename(f)}")

    print(f"\n💡 Key features:")
    print(f"   ✓ MACRO: Swing-adjusted beat spacing preserves groove")
    print(f"   ✓ MICRO: Pattern-matched subdivisions preserve rhythmic vocabulary")
    print(f"   ✓ Fast patterns like 'Ti Re Ki T' stay fast within one beat")
    print(f"   ✓ Same timing for tabla AND drums - direct comparison!")
    print()

    return exported_files


if __name__ == "__main__":
    files = export_two_level_swing()

    import subprocess
    subprocess.call(['open', 'generated_two_level_swing'])
