"""
Blend Knob Generator V2 - With Two-Level Swing Preservation

This version properly implements:
- MACRO: Beat-level swing timing (groove)
- MICRO: Subdivision pattern preservation ("Ti Re Ki T" stays fast)

Usage:
    python blend_knob_v2_with_swing.py <input_file> <blend_ratio> [options]
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
import random
from collections import defaultdict

# Import Model C
from meter_conditional_lstm import create_model

# ============================================================================
# CNN CLASSIFIER
# ============================================================================

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

# Duration bins and midpoints
DURATION_BINS = [0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]
BIN_MIDPOINTS = [0.15, 0.4, 0.65, 1.15, 2.0]

def quantize_duration(duration):
    """Quantize duration into bins"""
    for i, threshold in enumerate(DURATION_BINS[1:]):
        if duration < threshold:
            return i
    return len(DURATION_BINS) - 2

# ============================================================================
# STAGE 1: INPUT ANALYSIS & CLASSIFICATION
# ============================================================================

def compute_rcd_onsets(y, sr, n_fft=2048, hop_length=512, threshold=0.3):
    """Detect onsets using Rectified Complex Domain method"""
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

    if len(onset_samples) > 0 and onset_samples[-1] < len(y):
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

def Adjust_Length(audio_data, target_length):
    if len(audio_data) < target_length:
        padded_audio = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
        return padded_audio
    else:
        return audio_data

def classify_input(file_path, cnn_model, db_threshold=-30, pre_onset_samples=1000):
    """
    Stage 1: Classify input tabla file
    Returns: notes, durations, note_onset_samples (matching notes), velocities, sr
    """
    print("\n" + "="*80)
    print("STAGE 1: INPUT ANALYSIS")
    print("="*80)
    print(f"📂 Loading: {file_path}")

    y, sr = librosa.load(file_path, sr=None)
    onset_samples, onset_env = compute_rcd_onsets(y, sr)

    if 0 not in onset_samples:
        onset_samples = np.insert(onset_samples, 0, 0)

    results = []
    durations = []
    note_onset_samples = []  # Track onset for each detected note
    velocities = []  # Track velocity (amplitude) for each note

    print(f"🔍 Detecting onsets and classifying...")
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

        # Extract velocity (peak amplitude in the attack portion)
        # Use first 50ms for attack detection
        attack_samples = min(int(0.05 * sr), len(stroke))
        attack_portion = stroke[:attack_samples]
        peak_amplitude = np.max(np.abs(attack_portion))

        adjusted = Adjust_Length(stroke, 72000)
        features = preprocess(adjusted, sr)
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        cnn_model.eval()
        with torch.no_grad():
            output = cnn_model(input_tensor)
            pred_index = torch.argmax(output, dim=1).item()
            predicted_bol = note_labels[pred_index]

        results.append(predicted_bol)
        durations.append(duration_sec)
        note_onset_samples.append(onset_samples[i])  # Store onset for this note
        velocities.append(peak_amplitude)  # Store raw amplitude

    # Normalize velocities to 0.3-1.0 range (avoid too quiet)
    velocities = np.array(velocities)
    if len(velocities) > 0 and velocities.max() > 0:
        velocities_normalized = velocities / velocities.max()
        # Scale to 0.3-1.0 range (minimum velocity of 0.3)
        velocities_normalized = 0.3 + (velocities_normalized * 0.7)
    else:
        velocities_normalized = np.ones(len(velocities))

    print(f"✅ Detected {len(results)} tabla strokes")
    print(f"   Total duration: {sum(durations):.2f}s")
    print(f"   Velocity range: {velocities_normalized.min():.2f} - {velocities_normalized.max():.2f}")
    print(f"   Notes: {' '.join(results[:20])}{'...' if len(results) > 20 else ''}")

    return results, durations, np.array(note_onset_samples), velocities_normalized, sr

# ============================================================================
# STAGE 2: TWO-LEVEL SWING EXTRACTION
# ============================================================================

def detect_tempo_and_beats(durations, onset_samples, sample_rate):
    """
    Detect tempo and beat positions using simple autocorrelation
    Returns: beat_iois, beat_positions_samples
    """
    print("\n" + "="*80)
    print("STAGE 2: TEMPO & BEAT DETECTION")
    print("="*80)

    # Estimate tempo from median IOI
    median_ioi = np.median(durations)
    estimated_tempo = 60.0 / median_ioi if median_ioi > 0 else 120

    print(f"🎼 Estimated tempo: {estimated_tempo:.1f} BPM")
    print(f"   Median IOI: {median_ioi:.3f}s")

    # Simple beat detection: cluster IOIs and find beat-level ones
    # Beat-level IOIs are typically ~0.5-0.8s range
    durations_array = np.array(durations)

    # Find IOIs that are likely beat-level (not subdivisions)
    beat_threshold = 0.4  # IOIs above this are likely beats
    potential_beat_indices = np.where(durations_array >= beat_threshold)[0]

    if len(potential_beat_indices) == 0:
        # Fallback: use cumulative approach
        beat_duration_target = 60.0 / estimated_tempo
        beat_positions = [0]
        accumulated = 0.0

        for i, d in enumerate(durations):
            accumulated += d
            if accumulated >= beat_duration_target * 0.8:  # 80% threshold
                beat_positions.append(onset_samples[i+1] if i+1 < len(onset_samples) else onset_samples[-1])
                accumulated = 0.0

        beat_positions = np.array(beat_positions)
    else:
        # Use detected beat-level notes
        beat_positions = onset_samples[potential_beat_indices]

    # Calculate beat IOIs
    beat_iois = np.diff(beat_positions) / sample_rate

    print(f"✅ Detected {len(beat_iois)} beats")
    print(f"   Beat IOI range: {beat_iois.min():.3f}s - {beat_iois.max():.3f}s")
    print(f"   Beat IOI mean: {beat_iois.mean():.3f}s (CV: {beat_iois.std()/beat_iois.mean():.3f})")

    return beat_iois, beat_positions

def extract_beat_patterns(notes, durations, onset_samples, beat_positions, sample_rate):
    """
    Extract beat-level patterns with subdivision structure

    Returns:
        beat_patterns: dict mapping (num_notes, duration_bins) -> list of patterns
        swing_beat_iois: array of beat IOIs
    """
    print("\n" + "="*80)
    print("STAGE 3: BEAT PATTERN EXTRACTION")
    print("="*80)

    beat_patterns = defaultdict(list)

    # Convert to numpy arrays
    # onset_samples now matches notes length (tracked during classification)
    onset_samples = np.array(onset_samples)
    notes_array = np.array(notes)
    durations_array = np.array(durations)

    print(f"   Debug: onset_samples length: {len(onset_samples)}, notes length: {len(notes_array)}")

    # Calculate beat IOIs
    swing_beat_iois = np.diff(beat_positions) / sample_rate

    for i in range(len(beat_positions) - 1):
        beat_start = beat_positions[i]
        beat_end = beat_positions[i + 1]
        beat_duration = swing_beat_iois[i]

        # Find notes within this beat
        beat_note_mask = (onset_samples >= beat_start) & (onset_samples < beat_end)
        beat_notes = notes_array[beat_note_mask]
        beat_onsets = onset_samples[beat_note_mask]
        beat_durations = durations_array[beat_note_mask]

        if len(beat_notes) == 0:
            continue

        # Calculate relative timing within beat (MICRO timing)
        beat_onsets_relative = (beat_onsets - beat_start) / sample_rate
        beat_iois = np.diff(np.append(beat_onsets_relative, beat_duration))

        # Normalize to proportions (relative to beat duration)
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
            'relative_iois': relative_iois,
            'notes': list(beat_notes)
        }

        beat_patterns[pattern_key].append(pattern)

    print(f"📊 Extracted Beat Pattern Library:")
    print(f"   Total unique patterns: {len(beat_patterns)}")

    # Show top 10 most common patterns
    for key, patterns in sorted(beat_patterns.items(), key=lambda x: -len(x[1]))[:10]:
        num_notes, bins = key
        example_pattern = patterns[0]
        notes_str = ' '.join(example_pattern['notes'][:5])
        if len(example_pattern['notes']) > 5:
            notes_str += '...'
        print(f"   {num_notes} notes {bins}: {len(patterns)} examples (e.g., {notes_str})")

    return beat_patterns, swing_beat_iois

# ============================================================================
# STAGE 4: PATTERN MATCHING & GENERATION
# ============================================================================

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


def generate_with_two_level_swing(lstm_model, metadata, seed_notes, seed_durations,
                                  beat_patterns, swing_beat_iois,
                                  taal_id=0, num_generate=32, temperature=1.0):
    """
    Generate tabla sequence and apply two-level swing timing

    Returns:
        gen_notes: Generated note names
        gen_durations: Swing-adjusted durations
        pattern_stats: Matching statistics
    """
    print("\n" + "="*80)
    print("STAGE 4: GENERATION WITH TWO-LEVEL SWING")
    print("="*80)

    note_to_idx = {note: idx for idx, note in enumerate(metadata['note_labels'])}
    idx_to_note = metadata['note_labels']

    # Use last 16 notes as seed
    seed_window = min(16, len(seed_notes))
    seed_notes_subset = seed_notes[-seed_window:]
    seed_durations_subset = seed_durations[-seed_window:]

    # Convert to indices
    seed_note_indices = [note_to_idx[note] for note in seed_notes_subset]
    seed_duration_bins = [quantize_duration(d) for d in seed_durations_subset]

    print(f"🤖 Generating {num_generate} notes...")
    print(f"   Taal ID: {taal_id}, Temperature: {temperature}")

    # Generate
    gen_note_indices, gen_duration_indices = lstm_model.generate(
        seed_notes=seed_note_indices,
        seed_durations=seed_duration_bins,
        taal_id=taal_id,
        num_generate=num_generate,
        temperature=temperature,
        device='cpu'
    )

    # Convert note indices to names
    gen_note_labels = [idx_to_note[idx] for idx in gen_note_indices]

    print(f"   Generated notes: {' '.join(gen_note_labels[:10])}...")
    print(f"   Generated bins: {gen_duration_indices[:10]}...")

    # Apply two-level swing timing
    print(f"\n🎼 Applying two-level swing timing...")

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

    print(f"\n✅ Generation complete:")
    print(f"   {len(gen_note_labels)} notes, {sum(gen_durations_final):.2f}s")
    print(f"   Duration range: {min(gen_durations_final):.3f}s - {max(gen_durations_final):.3f}s")

    return gen_note_labels, gen_durations_final, pattern_stats

# ============================================================================
# STAGE 5: BAR SEGMENTATION & BLENDING
# ============================================================================

def segment_into_bars(notes, durations, beats_per_bar):
    """
    Segment notes into complete bars

    Returns:
        bars: List of bar dicts
        leftover: Leftover notes dict (or None)
    """
    num_notes = len(notes)
    num_complete_bars = num_notes // beats_per_bar
    num_leftover = num_notes % beats_per_bar

    bars = []
    for i in range(num_complete_bars):
        start_idx = i * beats_per_bar
        end_idx = start_idx + beats_per_bar

        bar = {
            'notes': notes[start_idx:end_idx],
            'durations': durations[start_idx:end_idx],
            'bar_index': i
        }
        bars.append(bar)

    leftover = None
    if num_leftover > 0:
        leftover = {
            'notes': notes[num_complete_bars * beats_per_bar:],
            'durations': durations[num_complete_bars * beats_per_bar:],
            'bar_index': -1
        }

    return bars, leftover


def blend_bars_with_two_level_swing(input_bars, gen_bars_pool, leftover, blend_ratio,
                                    beat_patterns, swing_beat_iois, metadata):
    """
    Blend input and generated bars with two-level swing preservation

    Args:
        input_bars: List of input bar dicts
        gen_bars_pool: List of 2 generated bar dicts (with notes and duration_bins)
        leftover: Leftover notes dict
        blend_ratio: 0.0 to 1.0
        beat_patterns: Beat pattern library
        swing_beat_iois: Swing beat IOIs
        metadata: Model metadata

    Returns:
        output_notes, output_durations, blend_decisions
    """
    print("\n" + "="*80)
    print("STAGE 5: BAR-LEVEL BLENDING WITH TWO-LEVEL SWING")
    print("="*80)
    print(f"🎛️  Blend Ratio: {blend_ratio:.1%}")

    output_notes = []
    output_durations = []
    blend_decisions = []

    for bar in input_bars:
        bar_idx = bar['bar_index']
        random_value = random.random()

        if random_value < blend_ratio and len(gen_bars_pool) > 0:
            # USE GENERATED BAR
            selected_gen_bar = random.choice(gen_bars_pool)

            # Apply two-level swing to generated bar
            gen_duration_bins = selected_gen_bar['duration_bins']

            # Group into beats
            avg_beat_duration = np.mean(swing_beat_iois)
            beat_chunks = group_notes_into_beats(gen_duration_bins, avg_beat_duration)

            # Match and apply patterns
            pattern_stats = {'exact_match': 0, 'num_notes_match': 0, 'synthesized': 0}
            bar_durations = []

            for chunk_idx, chunk_indices in enumerate(beat_chunks):
                chunk_bins = tuple([gen_duration_bins[i] for i in chunk_indices])

                # Get swing beat duration (cycle through)
                swing_beat_idx = (bar_idx * 4 + chunk_idx) % len(swing_beat_iois)
                swing_duration = swing_beat_iois[swing_beat_idx]

                # Match and apply
                beat_durs = match_and_apply_pattern(
                    chunk_bins, beat_patterns, swing_duration, pattern_stats
                )
                bar_durations.extend(beat_durs)

            output_notes.extend(selected_gen_bar['notes'])
            output_durations.extend(bar_durations)

            blend_decisions.append({
                'bar': bar_idx + 1,
                'decision': 'GENERATED',
                'source': f"Gen Bar {selected_gen_bar['label']}",
                'random_roll': random_value,
                'pattern_stats': pattern_stats
            })
        else:
            # KEEP INPUT BAR (exact durations)
            output_notes.extend(bar['notes'])
            output_durations.extend(bar['durations'])

            blend_decisions.append({
                'bar': bar_idx + 1,
                'decision': 'INPUT',
                'source': 'Original',
                'random_roll': random_value
            })

    # Always keep leftover as input
    if leftover:
        output_notes.extend(leftover['notes'])
        output_durations.extend(leftover['durations'])
        blend_decisions.append({
            'bar': 'Leftover',
            'decision': 'INPUT',
            'source': 'Original (always)',
            'random_roll': None
        })

    # Print decisions
    print(f"\n📊 Blending Decisions:")
    for decision in blend_decisions:
        bar_label = f"Bar {decision['bar']}" if isinstance(decision['bar'], int) else decision['bar']
        roll_str = f"(roll: {decision['random_roll']:.2f})" if decision['random_roll'] is not None else ""

        if decision['decision'] == 'GENERATED':
            print(f"   {bar_label}: ✨ {decision['source']} {roll_str}")
        else:
            print(f"   {bar_label}: 📥 {decision['source']} {roll_str}")

    generated_count = sum(1 for d in blend_decisions if d['decision'] == 'GENERATED' and isinstance(d.get('bar'), int))
    total_complete_bars = len(input_bars)
    actual_ratio = generated_count / total_complete_bars if total_complete_bars > 0 else 0

    print(f"\n✅ Blending complete:")
    print(f"   {generated_count}/{total_complete_bars} bars generated ({actual_ratio:.1%})")
    print(f"   Total notes: {len(output_notes)}, Duration: {sum(output_durations):.2f}s")

    return output_notes, output_durations, blend_decisions


# ============================================================================
# STAGE 6: AUDIO SYNTHESIS
# ============================================================================

def synthesize_audio(notes, durations, folder='tabla', sample_rate=44100,
                    crossfade_ms=5, ghost_notes=True, ghost_volume=0.3,
                    use_snare_variants=True, velocities=None):
    """
    Synthesize audio with exact onset timing and velocity modulation

    Args:
        velocities: Array of velocity values (0.0-1.0) for each note
        use_snare_variants: If True and folder='drums', use snare variants for
                           Ti, Re, Ki, T, Kat, Na instead of base drum samples
    """
    if len(notes) == 0:
        return np.array([]), sample_rate

    # Calculate exact onset times from durations
    onset_times = np.insert(np.cumsum(durations), 0, 0.0)

    # If no velocities provided, use uniform velocity
    if velocities is None:
        velocities = np.ones(len(notes))
    else:
        velocities = np.array(velocities)

    # Define snare-mapped strokes
    snare_strokes = ['Ti', 'Re', 'Ki', 'T', 'Kat', 'Na']

    # Load all samples
    samples = []
    for i, note in enumerate(notes):
        # Check if we should use snare variant
        if use_snare_variants and folder == 'core/drums' and note in snare_strokes:
            audio_file = f"snare_variants/snare_{note}.wav"
        else:
            audio_file = f"{folder}/{note}.wav"

        if os.path.exists(audio_file):
            try:
                audio_data, sr = librosa.load(audio_file, sr=sample_rate)

                # Apply velocity modulation
                velocity = velocities[i] if i < len(velocities) else 1.0
                audio_data = audio_data * velocity

                # Apply ghost notes (after velocity)
                if ghost_notes and note == "T":
                    audio_data = audio_data * ghost_volume

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

    # Place each sample at exact onset time
    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    tail_fadeout_samples = int(sample_rate * 20 / 1000)

    for i, (sample, onset_time) in enumerate(zip(samples, onset_times)):
        if len(sample) == 0:
            continue

        # Apply tail fade-out
        sample_copy = sample.copy()
        if len(sample_copy) > tail_fadeout_samples:
            fade_out = np.linspace(1, 0, tail_fadeout_samples)
            sample_copy[-tail_fadeout_samples:] *= fade_out

        onset_sample = int(onset_time * sample_rate)

        # Check for overlap and apply crossfade
        if i > 0:
            prev_onset = int(onset_times[i-1] * sample_rate)
            prev_end = prev_onset + len(samples[i-1])

            if onset_sample < prev_end:
                overlap_length = prev_end - onset_sample
                fade_length = min(crossfade_samples, overlap_length, len(sample_copy) // 2)

                if fade_length > 0:
                    fade_out_curve = np.linspace(1, 0, fade_length)
                    fade_in_curve = np.linspace(0, 1, fade_length)

                    full_audio[onset_sample:onset_sample+fade_length] *= fade_out_curve
                    full_audio[onset_sample:onset_sample+fade_length] += sample_copy[:fade_length] * fade_in_curve

                    if len(sample_copy) > fade_length:
                        end_pos = min(onset_sample + len(sample_copy), len(full_audio))
                        remaining_length = end_pos - (onset_sample + fade_length)
                        full_audio[onset_sample+fade_length:end_pos] += sample_copy[fade_length:fade_length+remaining_length]
                else:
                    end_pos = min(onset_sample + len(sample_copy), len(full_audio))
                    full_audio[onset_sample:end_pos] += sample_copy[:end_pos-onset_sample]
            else:
                end_pos = min(onset_sample + len(sample_copy), len(full_audio))
                full_audio[onset_sample:end_pos] += sample_copy[:end_pos-onset_sample]
        else:
            end_pos = min(onset_sample + len(sample_copy), len(full_audio))
            full_audio[onset_sample:end_pos] += sample_copy[:end_pos-onset_sample]

    # Normalize
    max_val = np.max(np.abs(full_audio))
    if max_val > 0:
        full_audio = full_audio / max_val * 0.9

    return full_audio, sample_rate


# ============================================================================
# MAIN - COMPLETE BLEND KNOB
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Blend Knob Generator V2 with Two-Level Swing')
    parser.add_argument('input_file', help='Input tabla audio file')
    parser.add_argument('blend_ratio', type=float, help='Blend ratio (0.0-1.0)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Generation temperature')
    parser.add_argument('--output_dir', default='generated_blend_v2', help='Output directory')
    parser.add_argument('--beats_per_bar', type=int, default=12, help='Beats per bar (12=Ektaal, 16=Teental)')

    args = parser.parse_args()

    if not 0.0 <= args.blend_ratio <= 1.0:
        print("❌ Error: blend_ratio must be between 0.0 and 1.0")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]

    print("\n" + "="*80)
    print("🎛️  BLEND KNOB GENERATOR V2 - WITH TWO-LEVEL SWING")
    print("="*80)
    print(f"Input: {args.input_file}")
    print(f"Blend Ratio: {args.blend_ratio:.1%}")
    print(f"Temperature: {args.temperature}")
    print(f"Beats per bar: {args.beats_per_bar}")

    # Load CNN
    print("\n📦 Loading models...")
    cnn_model = ConvNet(input_channels=13, num_classes=12)
    cnn_model.load_state_dict(torch.load("core/models/ConvNet_SNFPR_model.pth"))
    cnn_model.eval()
    print("   ✅ CNN loaded")

    # Load Model C
    checkpoint = torch.load("core/models/best_bar_aware_lstm.pth", map_location='cpu')
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
    print(f"   ✅ Model C loaded")

    # Stage 1: Classify
    notes, durations, onset_samples, velocities, sr = classify_input(args.input_file, cnn_model)

    # Stage 2: Detect tempo and beats
    beat_iois, beat_positions = detect_tempo_and_beats(durations, onset_samples, sr)

    # Stage 3: Extract beat patterns
    beat_patterns, swing_beat_iois = extract_beat_patterns(
        notes, durations, onset_samples, beat_positions, sr
    )

    # Segment into bars
    print("\n📐 Segmenting into bars...")
    input_bars, leftover = segment_into_bars(notes, durations, args.beats_per_bar)
    print(f"   {len(input_bars)} complete bars + {len(leftover['notes']) if leftover else 0} leftover notes")

    # Stage 4: Generate variation pool (2 bars)
    note_to_idx = {note: idx for idx, note in enumerate(metadata['note_labels'])}
    idx_to_note = metadata['note_labels']

    print(f"\n🤖 Generating variation pool (2 bars)...")
    gen_bars_pool = []

    for bar_label in ['A', 'B']:
        # Generate one bar
        seed_window = min(args.beats_per_bar, len(notes))
        seed_notes_subset = notes[-seed_window:]
        seed_durations_subset = durations[-seed_window:]

        seed_note_indices = [note_to_idx[note] for note in seed_notes_subset]
        seed_duration_bins = [quantize_duration(d) for d in seed_durations_subset]

        # Determine taal_id based on beats_per_bar
        if args.beats_per_bar == 16:
            taal_id = 0  # Teental
        elif args.beats_per_bar == 12:
            taal_id = 1  # Ektaal
        else:
            taal_id = 0  # Default

        gen_note_indices, gen_duration_indices = lstm_model.generate(
            seed_notes=seed_note_indices,
            seed_durations=seed_duration_bins,
            taal_id=taal_id,
            num_generate=args.beats_per_bar,
            temperature=args.temperature,
            device='cpu'
        )

        gen_notes = [idx_to_note[idx] for idx in gen_note_indices]

        gen_bar = {
            'notes': gen_notes,
            'duration_bins': gen_duration_indices,
            'label': bar_label
        }
        gen_bars_pool.append(gen_bar)

        print(f"   Bar {bar_label}: {' '.join(gen_notes[:8])}...")

    # Stage 5: Blend
    output_notes, output_durations, blend_decisions = blend_bars_with_two_level_swing(
        input_bars, gen_bars_pool, leftover, args.blend_ratio,
        beat_patterns, swing_beat_iois, metadata
    )

    # Stage 6: Synthesize
    print("\n🔨 Synthesizing audio...")
    # Use input velocities (they will be preserved through blending)
    tabla_audio, sr = synthesize_audio(output_notes, output_durations, folder='core/tabla', velocities=velocities)
    drums_audio, sr = synthesize_audio(output_notes, output_durations, folder='core/drums', velocities=velocities)

    # Save
    tabla_file = f"{args.output_dir}/{base_name}_blend_{args.blend_ratio:.2f}_tabla.wav"
    drums_file = f"{args.output_dir}/{base_name}_blend_{args.blend_ratio:.2f}_drums.wav"

    sf.write(tabla_file, tabla_audio, sr)
    sf.write(drums_file, drums_audio, sr)

    print(f"   ✅ Saved: {tabla_file}")
    print(f"   ✅ Saved: {drums_file}")

    # Summary
    print("\n" + "="*80)
    print("✨ COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Input: {len(notes)} notes, {sum(durations):.2f}s")
    print(f"   Output: {len(output_notes)} notes, {sum(output_durations):.2f}s")
    print(f"   Blend: {args.blend_ratio:.1%}")
    print(f"   Beat patterns: {len(beat_patterns)} unique")
    print(f"\n📁 Output files:")
    print(f"   {tabla_file}")
    print(f"   {drums_file}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()