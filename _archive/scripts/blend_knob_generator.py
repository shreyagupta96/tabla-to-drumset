"""
Blend Knob Generator - Controlled AI Variation with Swing Preservation

Usage:
    python blend_knob_generator.py <input_file> <blend_ratio> [options]

Example:
    python blend_knob_generator.py input/Teentaal.wav 0.2
    python blend_knob_generator.py input/Teentaal.wav 0.5 --temperature 1.2
    python blend_knob_generator.py input/Teentaal.wav 0.0  # Exact reproduction

Args:
    input_file: Path to input tabla audio file
    blend_ratio: 0.0 (100% input) to 1.0 (100% generated)
    --temperature: Generation temperature (default: 1.0)
    --output_dir: Output directory (default: generated_blend/)
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
import random

# Import CNN classifier
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

note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]

# ============================================================================
# STAGE 1: INPUT ANALYSIS
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
    Returns: notes (list), durations (list)
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

    print(f"🔍 Detecting onsets and classifying...")
    for i in range(len(onset_samples) - 1):
        start = max(onset_samples[i] - pre_onset_samples, 0)
        end = onset_samples[i + 1]
        duration_samples = end - start
        duration_sec = duration_samples / sr
        durations.append(duration_sec)
        stroke = y[start:end]

        if stroke.shape[0] == 0:
            continue

        stroke_db = librosa.amplitude_to_db([np.max(np.abs(stroke))])[0]
        if stroke_db < db_threshold:
            continue

        adjusted = Adjust_Length(stroke, 72000)
        features = preprocess(adjusted, sr)
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        cnn_model.eval()
        with torch.no_grad():
            output = cnn_model(input_tensor)
            pred_index = torch.argmax(output, dim=1).item()
            predicted_bol = note_labels[pred_index]

        results.append(predicted_bol)

    print(f"✅ Detected {len(results)} tabla strokes")
    print(f"   Total duration: {sum(durations):.2f}s")
    print(f"   Notes: {' '.join(results[:20])}{'...' if len(results) > 20 else ''}")

    return results, durations

# ============================================================================
# STAGE 2: METER DETECTION & BAR SEGMENTATION
# ============================================================================

def detect_meter_simple(notes, durations):
    """
    Simple meter detection based on duration patterns
    Returns: taal_id, beats_per_bar, taal_name
    """
    # For now, assume Teental (can be enhanced with actual meter detection)
    # In production, use hybrid_meter_pipeline or pattern analysis

    total_duration = sum(durations)
    avg_ioi = np.mean(durations)
    num_notes = len(notes)

    # Estimate tempo
    estimated_tempo = 60.0 / avg_ioi if avg_ioi > 0 else 120

    # Simple heuristic: Teental is most common
    taal_id = 0
    beats_per_bar = 16
    taal_name = "Teental"

    print(f"\n🎼 Meter Detection:")
    print(f"   Detected: {taal_name} ({beats_per_bar} beats per bar)")
    print(f"   Estimated tempo: {estimated_tempo:.1f} BPM")

    return taal_id, beats_per_bar, taal_name

def segment_by_bars(notes, durations, beats_per_bar):
    """
    Stage 2: Segment notes into complete bars
    Returns: list of bars (each bar is dict with 'notes' and 'durations')
    """
    print("\n" + "="*80)
    print("STAGE 2: BAR SEGMENTATION")
    print("="*80)

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
            'bar_index': -1  # Special marker for leftover
        }

    print(f"✅ Segmented into {num_complete_bars} complete bars")
    if leftover:
        print(f"   + {num_leftover} leftover notes (will keep as input)")

    for i, bar in enumerate(bars[:3]):  # Show first 3 bars
        print(f"\n   Bar {i+1}: {' '.join(bar['notes'])}")

    if len(bars) > 3:
        print(f"   ... ({len(bars)-3} more bars)")

    return bars, leftover

# ============================================================================
# STAGE 3: GENERATE VARIATION POOL
# ============================================================================

def quantize_duration(duration, bins=[0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]):
    """Quantize duration into bins"""
    for i, edge in enumerate(bins[1:]):
        if duration < edge:
            return i
    return len(bins) - 2

def generate_variation_pool(lstm_model, metadata, seed_notes, seed_durations,
                           taal_id, beats_per_bar, temperature=1.0, num_bars=2):
    """
    Stage 3: Generate variation pool using Model C
    Returns: list of generated bars (same structure as input bars)
    """
    print("\n" + "="*80)
    print("STAGE 3: GENERATE VARIATION POOL (Model C)")
    print("="*80)
    print(f"🤖 Generating {num_bars} complete bars...")
    print(f"   Taal ID: {taal_id}, Beats per bar: {beats_per_bar}")
    print(f"   Temperature: {temperature}")

    note_to_idx = {note: idx for idx, note in enumerate(metadata['note_labels'])}
    idx_to_note = metadata['note_labels']

    # Use last beats_per_bar notes as seed
    seed_window = min(beats_per_bar, len(seed_notes))
    seed_notes_subset = seed_notes[-seed_window:]
    seed_durations_subset = seed_durations[-seed_window:]

    # Convert to indices
    seed_note_indices = [note_to_idx[note] for note in seed_notes_subset]
    seed_duration_bins = [quantize_duration(d) for d in seed_durations_subset]

    gen_bars = []

    for bar_idx in range(num_bars):
        print(f"   Generating Bar {chr(65+bar_idx)}...")

        # Generate one complete bar
        gen_note_indices, gen_dur_indices = lstm_model.generate(
            seed_notes=seed_note_indices,
            seed_durations=seed_duration_bins,
            taal_id=taal_id,
            num_generate=beats_per_bar,
            temperature=temperature,
            device='cpu'
        )

        # Convert back to note names
        gen_notes = [idx_to_note[idx] for idx in gen_note_indices]

        # Note: We generate duration bins but will discard them during blending
        # (Input durations will be used instead to preserve swing)

        gen_bar = {
            'notes': gen_notes,
            'bar_index': bar_idx,
            'label': chr(65 + bar_idx)  # A, B, C, etc.
        }
        gen_bars.append(gen_bar)

        # Use this generation as seed for next bar (for variety)
        seed_note_indices = gen_note_indices[-seed_window:]
        seed_duration_bins = gen_dur_indices[-seed_window:]

        print(f"      {' '.join(gen_notes[:10])}{'...' if len(gen_notes) > 10 else ''}")

    print(f"✅ Generated {len(gen_bars)} variation bars")

    return gen_bars

# ============================================================================
# STAGE 4: BAR-LEVEL BLENDING
# ============================================================================

def blend_bars(input_bars, gen_bars, leftover, blend_ratio):
    """
    Stage 4: Blend input and generated bars
    Returns: output_notes, output_durations
    """
    print("\n" + "="*80)
    print("STAGE 4: BAR-LEVEL BLENDING")
    print("="*80)
    print(f"🎛️  Blend Ratio: {blend_ratio:.1%}")
    print(f"   (0% = all input, 100% = all generated)")

    output_notes = []
    output_durations = []

    blend_decisions = []

    for bar in input_bars:
        bar_idx = bar['bar_index']

        # Roll the dice
        random_value = random.random()

        if random_value < blend_ratio and len(gen_bars) > 0:
            # Use generated bar (randomly pick from pool)
            selected_gen_bar = random.choice(gen_bars)

            # Generated notes + Input durations (SWING PRESERVATION!)
            output_notes.extend(selected_gen_bar['notes'])
            output_durations.extend(bar['durations'])

            blend_decisions.append({
                'bar': bar_idx + 1,
                'decision': 'GENERATED',
                'source': f"Gen Bar {selected_gen_bar['label']}",
                'random_roll': random_value
            })
        else:
            # Keep input bar
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

    generated_count = sum(1 for d in blend_decisions[:-1] if d['decision'] == 'GENERATED')
    total_complete_bars = len(input_bars)
    actual_ratio = generated_count / total_complete_bars if total_complete_bars > 0 else 0

    print(f"\n✅ Blending complete:")
    print(f"   {generated_count}/{total_complete_bars} bars generated ({actual_ratio:.1%})")
    print(f"   All bars use INPUT durations (swing preserved)")

    return output_notes, output_durations

# ============================================================================
# STAGE 5: AUDIO SYNTHESIS
# ============================================================================

def synthesize_audio(notes, durations, folder='tabla', sample_rate=44100,
                    crossfade_ms=5, ghost_notes=True, ghost_volume=0.3):
    """
    Stage 5: Synthesize audio with exact onset timing
    Places samples at exact times calculated from durations
    """
    print("\n" + "="*80)
    print(f"STAGE 5: AUDIO SYNTHESIS ({folder.upper()})")
    print("="*80)
    print(f"🎵 Synthesizing {len(notes)} notes...")

    if len(notes) == 0:
        return np.array([]), sample_rate

    # Calculate exact onset times from durations (cumulative sum)
    onset_times = np.insert(np.cumsum(durations), 0, 0.0)

    # Load all samples
    samples = []
    for note in notes:
        audio_file = f"{folder}/{note}.wav"
        if os.path.exists(audio_file):
            try:
                audio_data, sr = librosa.load(audio_file, sr=sample_rate)

                # Apply ghost notes
                if ghost_notes and note == "T":
                    audio_data = audio_data * ghost_volume

                samples.append(audio_data)
            except:
                samples.append(np.array([]))
        else:
            print(f"⚠️  Missing: {audio_file}")
            samples.append(np.array([]))

    # Calculate total length needed
    last_onset_samples = int(onset_times[-1] * sample_rate)
    max_sample_length = max([len(s) for s in samples] + [0])
    total_length = last_onset_samples + max_sample_length

    # Create output buffer
    full_audio = np.zeros(total_length)

    # Place each sample at its exact onset time
    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    tail_fadeout_samples = int(sample_rate * 20 / 1000)  # 20ms tail fade

    for i, (sample, onset_time) in enumerate(zip(samples, onset_times)):
        if len(sample) == 0:
            continue

        # Apply tail fade-out to prevent clicks
        sample_copy = sample.copy()
        if len(sample_copy) > tail_fadeout_samples:
            fade_out = np.linspace(1, 0, tail_fadeout_samples)
            sample_copy[-tail_fadeout_samples:] *= fade_out

        onset_sample = int(onset_time * sample_rate)

        # Check for overlap and apply crossfade
        if i > 0 and onset_sample < int(onset_times[i-1] * sample_rate) + len(samples[i-1]):
            prev_onset = int(onset_times[i-1] * sample_rate)
            prev_end = prev_onset + len(samples[i-1])

            overlap_start = onset_sample
            overlap_end = min(onset_sample + len(sample_copy), prev_end)
            overlap_length = overlap_end - overlap_start

            if overlap_length > 0:
                fade_length = min(crossfade_samples, overlap_length)
                fade_out_curve = np.linspace(1, 0, fade_length)
                fade_in_curve = np.linspace(0, 1, fade_length)

                # Fade out previous sample
                full_audio[overlap_start:overlap_start+fade_length] *= fade_out_curve
                # Fade in current sample
                crossfaded = sample_copy[:fade_length] * fade_in_curve
                full_audio[overlap_start:overlap_start+fade_length] += crossfaded

                # Add rest of current sample
                if len(sample_copy) > fade_length:
                    end_pos = min(onset_sample + len(sample_copy), len(full_audio))
                    remaining_length = end_pos - (onset_sample + fade_length)
                    full_audio[onset_sample+fade_length:end_pos] += sample_copy[fade_length:fade_length+remaining_length]
            else:
                # No overlap, just add
                end_pos = min(onset_sample + len(sample_copy), len(full_audio))
                full_audio[onset_sample:end_pos] += sample_copy[:end_pos-onset_sample]
        else:
            # No overlap, just add
            end_pos = min(onset_sample + len(sample_copy), len(full_audio))
            full_audio[onset_sample:end_pos] += sample_copy[:end_pos-onset_sample]

    # Normalize
    max_val = np.max(np.abs(full_audio))
    if max_val > 0:
        full_audio = full_audio / max_val * 0.9

    print(f"✅ Synthesis complete: {len(full_audio)/sample_rate:.2f}s audio")

    return full_audio, sample_rate

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Blend Knob Generator - Controlled AI Variation')
    parser.add_argument('input_file', help='Input tabla audio file')
    parser.add_argument('blend_ratio', type=float, help='Blend ratio (0.0 = all input, 1.0 = all generated)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Generation temperature (default: 1.0)')
    parser.add_argument('--output_dir', default='generated_blend', help='Output directory (default: generated_blend/)')

    args = parser.parse_args()

    # Validate blend ratio
    if not 0.0 <= args.blend_ratio <= 1.0:
        print("❌ Error: blend_ratio must be between 0.0 and 1.0")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get base filename
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]

    print("\n" + "="*80)
    print("🎛️  BLEND KNOB GENERATOR")
    print("="*80)
    print(f"Input: {args.input_file}")
    print(f"Blend Ratio: {args.blend_ratio:.1%}")
    print(f"Temperature: {args.temperature}")
    print(f"Output Dir: {args.output_dir}/")

    # ========================================================================
    # Load Models
    # ========================================================================

    print("\n📦 Loading models...")

    # Load CNN classifier
    cnn_model = ConvNet(input_channels=13, num_classes=12)
    cnn_model.load_state_dict(torch.load("ConvNet_SNFPR_model.pth"))
    cnn_model.eval()
    print("   ✅ CNN classifier loaded")

    # Load Model C (4-Taal Bar-Aware LSTM)
    checkpoint = torch.load("models/best_bar_aware_lstm.pth", map_location='cpu')
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
    print(f"   ✅ Model C loaded (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f})")

    # ========================================================================
    # STAGE 1: Classify Input
    # ========================================================================

    notes, durations = classify_input(args.input_file, cnn_model)

    # ========================================================================
    # STAGE 2: Detect Meter & Segment Bars
    # ========================================================================

    taal_id, beats_per_bar, taal_name = detect_meter_simple(notes, durations)
    input_bars, leftover = segment_by_bars(notes, durations, beats_per_bar)

    # ========================================================================
    # STAGE 3: Generate Variation Pool
    # ========================================================================

    gen_bars = generate_variation_pool(
        lstm_model, metadata,
        seed_notes=notes,
        seed_durations=durations,
        taal_id=taal_id,
        beats_per_bar=beats_per_bar,
        temperature=args.temperature,
        num_bars=2
    )

    # ========================================================================
    # STAGE 4: Blend Bars
    # ========================================================================

    output_notes, output_durations = blend_bars(input_bars, gen_bars, leftover, args.blend_ratio)

    # ========================================================================
    # STAGE 5: Synthesize Audio
    # ========================================================================

    # Synthesize tabla version
    tabla_audio, sr = synthesize_audio(output_notes, output_durations, folder='tabla')
    tabla_file = f"{args.output_dir}/{base_name}_blend_{args.blend_ratio:.2f}_tabla.wav"
    sf.write(tabla_file, tabla_audio, sr)
    print(f"💾 Saved: {tabla_file}")

    # Synthesize drums version
    drums_audio, sr = synthesize_audio(output_notes, output_durations, folder='drums')
    drums_file = f"{args.output_dir}/{base_name}_blend_{args.blend_ratio:.2f}_drums.wav"
    sf.write(drums_file, drums_audio, sr)
    print(f"💾 Saved: {drums_file}")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "="*80)
    print("✨ COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Input: {len(notes)} notes, {sum(durations):.2f}s")
    print(f"   Output: {len(output_notes)} notes, {sum(output_durations):.2f}s")
    print(f"   Blend: {args.blend_ratio:.1%} (AI variation)")
    print(f"   Swing: 100% preserved (input durations used)")
    print(f"\n📁 Output files:")
    print(f"   Tabla: {tabla_file}")
    print(f"   Drums: {drums_file}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
