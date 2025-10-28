"""
Export audio with SWING-ADJUSTED timing
Uses the swing template from input to preserve natural groove
while respecting model's predicted duration bins

Usage:
    python export_with_swing.py <input_file> <model_path> [options]

Example:
    python export_with_swing.py input/Rupak_1.wav models/best_bar_aware_lstm.pth --num_generate 32
"""

import os
import numpy as np
import soundfile as sf
import librosa
import torch
import sys
import argparse

# Add meter detection path
sys.path.insert(0, "/Users/shreyagupta/Desktop/Research MFP/B264026_Masters_Final_Project_SG/Submission/Code/HybridApproach")

from meter_conditional_lstm import create_model
from hybrid_meter_pipeline import hybrid_meter
from batch_classify_long_files import classify_tabla_file, ConvNet
from segment_by_bars import segment_notes_by_bars
from taal_utils import meter_to_taal_id, taal_id_to_name

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
    """
    Extract inter-onset intervals (IOIs) from swing-adjusted timeline
    Categorize them by duration bin for later sampling

    Returns:
        duration_clusters: dict mapping bin_index -> list of IOIs
    """
    adjusted_beats = swing_result['adjusted_beats']

    # Compute inter-onset intervals
    iois = np.diff(adjusted_beats)

    # Categorize by duration bin (matching the input durations)
    duration_clusters = {i: [] for i in range(5)}

    for duration, ioi in zip(input_durations, iois):
        bin_idx = quantize_duration(duration)
        duration_clusters[bin_idx].append(ioi)

    # Ensure we have at least some IOIs in each bin
    for i in range(5):
        if len(duration_clusters[i]) == 0:
            # Fallback to bin midpoints if no samples
            bin_midpoints = [0.15, 0.4, 0.65, 1.15, 2.0]
            duration_clusters[i] = [bin_midpoints[i]]

    return duration_clusters


def generate_swing_adjusted_durations(gen_duration_indices, duration_clusters, add_variation=True):
    """
    Map generated duration bins to actual IOIs from swing template

    Args:
        gen_duration_indices: Model-predicted duration bins
        duration_clusters: IOIs categorized by bin (from input)
        add_variation: Add slight random variation to avoid exact repeats

    Returns:
        List of actual durations in seconds
    """
    swing_durations = []

    for bin_idx in gen_duration_indices:
        # Sample from the appropriate cluster
        candidates = duration_clusters[bin_idx]

        if len(candidates) > 0:
            # Randomly sample to add variety
            base_duration = np.random.choice(candidates)

            if add_variation:
                # Add ±5% variation to avoid mechanical repetition
                variation = np.random.uniform(0.95, 1.05)
                duration = base_duration * variation
            else:
                duration = base_duration
        else:
            # Fallback to bin midpoint
            bin_midpoints = [0.15, 0.4, 0.65, 1.15, 2.0]
            duration = bin_midpoints[bin_idx]

        swing_durations.append(duration)

    return swing_durations


def synthesize_audio_seamless(notes, durations, tempo_bpm, sample_rate=44100, crossfade_percent=2.0):
    """
    Synthesize audio with EXACT swing timing preservation and tempo-relative crossfades

    This function places each sample at the EXACT time specified by the swing durations,
    allowing natural overlaps and ensuring timing accuracy. Crossfade duration scales
    with tempo to maintain musicality.

    Args:
        notes: List of note names
        durations: List of inter-onset intervals (IOIs) in seconds
        tempo_bpm: Tempo in BPM (for calculating tempo-relative crossfade)
        sample_rate: Output sample rate
        crossfade_percent: Crossfade as percentage of beat duration (default: 2%)
    """
    # Calculate tempo-relative crossfade
    # beat_duration = 60 / tempo_bpm (seconds per beat)
    # crossfade = beat_duration * (crossfade_percent / 100)
    beat_duration_s = 60.0 / tempo_bpm
    crossfade_ms = beat_duration_s * 1000 * (crossfade_percent / 100.0)
    tabla_folder = 'tabla'

    if len(notes) == 0:
        return np.array([]), sample_rate

    # Calculate exact onset times from durations (cumulative sum)
    onset_times = np.insert(np.cumsum(durations), 0, 0.0)  # [0, dur0, dur0+dur1, ...]

    # Load all samples
    samples = []
    for note in notes:
        audio_file = f"{tabla_folder}/{note}.wav"
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
    tail_fadeout_ms = 20  # 20ms fade-out at end of each sample to prevent clicks
    tail_fadeout_samples = int(sample_rate * tail_fadeout_ms / 1000)

    for i, (sample, onset_time) in enumerate(zip(samples, onset_times)):
        if len(sample) == 0:
            continue

        # Apply tail fade-out to prevent clicks from abrupt sample endings
        sample_copy = sample.copy()
        if len(sample_copy) > tail_fadeout_samples:
            fade_out = np.linspace(1, 0, tail_fadeout_samples)
            sample_copy[-tail_fadeout_samples:] *= fade_out

        onset_sample = int(onset_time * sample_rate)

        # Check if we overlap with previous sample
        if i > 0 and onset_sample < int(onset_times[i-1] * sample_rate) + len(samples[i-1]):
            # We have overlap - apply crossfade
            prev_onset = int(onset_times[i-1] * sample_rate)
            prev_end = prev_onset + len(samples[i-1])

            overlap_start = onset_sample
            overlap_end = min(onset_sample + len(sample_copy), prev_end)
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
                crossfaded = sample_copy[overlap_offset_curr:overlap_offset_curr+fade_length] * fade_in
                full_audio[overlap_start:overlap_start+fade_length] += crossfaded

                # Add rest of current sample (non-overlapping part)
                rest_start = overlap_start + fade_length
                rest_samples = sample_copy[fade_length:]
                full_audio[rest_start:rest_start+len(rest_samples)] += rest_samples
            else:
                # No significant overlap, just add
                full_audio[onset_sample:onset_sample+len(sample_copy)] += sample_copy
        else:
            # No overlap, just place the sample
            full_audio[onset_sample:onset_sample+len(sample_copy)] += sample_copy

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
    """
    Generate with swing-adjusted timing

    Returns:
        context_notes, context_durations (from input),
        gen_notes, gen_durations (swing-adjusted),
        swing_stats (for analysis),
        tempo (BPM)
    """
    # Step 1: Classify notes
    cnn_model = ConvNet(input_channels=13, num_classes=12)
    cnn_model.load_state_dict(torch.load('ConvNet_SNFPR_model.pth'))
    cnn_model.eval()

    notes, durations, onset_samples = classify_tabla_file(audio_file, cnn_model, target_length=72000)

    # Step 2: Meter detection and segmentation (includes swing analysis)
    meter_result = hybrid_meter(audio_file)
    meter = meter_result.get('final_meter')
    bar_start_samples = meter_result.get('bar_start_samples', [])
    swing_result = meter_result.get('swing_result')
    tempo = meter_result.get('tempo', 80.0)  # Default to 80 BPM if not detected

    # Map meter to taal_id using centralized function
    taal_id = meter_to_taal_id(meter)
    if taal_id is None:
        raise ValueError(f"Unsupported meter: {meter} beats. Supported: 16 (Teental), 12 (Ektaal), 10 (Jhaptaal), 7 (Rupak)")

    bars = segment_notes_by_bars(notes, durations, onset_samples, bar_start_samples)

    if len(bars) < 3:
        return None, None, None, None, None

    # Step 3: Extract swing template from input
    duration_clusters = extract_swing_template(swing_result, durations)

    # Statistics about swing template
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

    # Map duration bins to swing-adjusted durations
    gen_durations_swing = generate_swing_adjusted_durations(gen_duration_indices, duration_clusters)

    # Get context (last bar)
    last_bar = bars[-1]
    context_notes = last_bar['notes']
    context_durations = last_bar['durations']

    return context_notes, context_durations, gen_note_labels, gen_durations_swing, swing_stats, tempo


def export_with_swing(input_file, model_path, num_generate=32, temperature=0.8, output_dir='generated_audio_with_swing'):
    """
    Export audio with swing-adjusted timing for a single input file

    Args:
        input_file: Path to input audio file
        model_path: Path to trained model
        num_generate: Number of notes to generate
        temperature: Sampling temperature (0.5-1.5)
        output_dir: Output directory for generated audio
    """

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print(" " * 20 + "EXPORTING AUDIO WITH SWING-ADJUSTED TIMING")
    print("=" * 100)
    print(f"\nInput file: {os.path.basename(input_file)}")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Output directory: {output_dir}")
    print("\nFeatures:")
    print("  ✓ Extracts swing template from input audio")
    print("  ✓ Model predicts duration bins (short/medium/long)")
    print("  ✓ Durations sampled from input's natural timing clusters")
    print("  ✓ Preserves groove while respecting model's rhythm")
    print("  ✓ ±5% variation to avoid mechanical repetition")
    print()

    model_name = os.path.basename(input_file).replace('.wav', '') + '_SWING'

    print()

    # Generate with swing
    context_notes, context_durations, gen_notes, gen_durations, swing_stats, tempo = generate_with_swing(
        input_file,
        model_path,
        num_generate=num_generate,
        temperature=temperature
    )

    if context_notes is None:
        print(f"❌ Generation failed")
        return None

    print(f"\n🎵 Detected Tempo: {tempo:.1f} BPM")
    print(f"   Crossfade: 2% of beat duration = {(60/tempo)*1000*0.02:.1f} ms")

    print(f"\n🎼 Swing Template Statistics:")
    print(f"   Total timing samples: {swing_stats['total_clusters']}")
    print(f"   Cluster sizes: {swing_stats['cluster_sizes']}")
    print(f"   Cluster means: {swing_stats['cluster_means']}")

    print(f"\n📥 CONTEXT (last bar of input):")
    print(f"   Notes: {' '.join(context_notes)}")
    print(f"   Duration: {sum(context_durations):.2f}s")

    print(f"\n📤 GENERATED (swing-adjusted):")
    print(f"   Notes: {' '.join(gen_notes)}")
    print(f"   Duration: {sum(gen_durations):.2f}s")
    print(f"   Duration range: {min(gen_durations):.3f}s - {max(gen_durations):.3f}s")

    # Synthesize context audio
    print(f"\n🔨 Synthesizing context audio...")
    context_audio, sr = synthesize_audio_seamless(context_notes, context_durations, tempo)

    # Synthesize generated audio with swing-adjusted durations
    print(f"🔨 Synthesizing generated audio (swing-adjusted)...")
    gen_audio, sr = synthesize_audio_seamless(gen_notes, gen_durations, tempo)

    if len(context_audio) == 0 or len(gen_audio) == 0:
        print(f"❌ Synthesis failed")
        return None

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
    output_file = os.path.join(output_dir, f"{model_name}.wav")
    sf.write(output_file, full_audio, sr)

    total_duration = len(full_audio) / sr
    print(f"\n✅ Exported: {output_file}")
    print(f"   Total duration: {total_duration:.2f}s")

    print(f"\n💡 Key features:")
    print(f"   ✓ Durations sampled from input's natural timing distribution")
    print(f"   ✓ Model controls WHAT to play (notes + relative rhythm)")
    print(f"   ✓ Swing template controls WHEN to play (natural microtiming)")
    print(f"   ✓ Preserves the groove and feel of the input performance")
    print()

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export tabla audio with swing-adjusted timing using 4-taal meter-conditional LSTM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python export_with_swing.py input/Rupak_1.wav models/best_bar_aware_lstm.pth
  python export_with_swing.py input/Ektaal.wav models/best_bar_aware_lstm.pth --num_generate 40 --temperature 0.9
        '''
    )

    parser.add_argument('input_file', type=str,
                        help='Path to input audio file (.wav)')
    parser.add_argument('model_path', type=str,
                        help='Path to trained 4-taal meter-conditional LSTM model (.pth)')
    parser.add_argument('--num_generate', type=int, default=32,
                        help='Number of notes to generate (default: 32)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature 0.5-1.5 (default: 0.8)')
    parser.add_argument('--output_dir', type=str, default='generated_audio_with_swing',
                        help='Output directory (default: generated_audio_with_swing)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file not found: {args.input_file}")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found: {args.model_path}")
        sys.exit(1)

    # Run export
    output_file = export_with_swing(
        input_file=args.input_file,
        model_path=args.model_path,
        num_generate=args.num_generate,
        temperature=args.temperature,
        output_dir=args.output_dir
    )

    if output_file:
        # Automatically open the folder
        import subprocess
        subprocess.call(['open', args.output_dir])
