"""
Export with PROPER DRUM MAPPING
Maps tabla strokes to actual drum kit pieces (snare, kick, ride, hihat, tomb)
Based on mapping from mapping.png
"""

import os
import numpy as np
import soundfile as sf
import librosa
import torch
import sys

sys.path.insert(0, "/Users/shreyagupta/Desktop/Research MFP/B264026_Masters_Final_Project_SG/Submission/Code/HybridApproach")

from meter_conditional_lstm import create_model
from hybrid_meter_pipeline import hybrid_meter
from batch_classify_long_files import classify_tabla_file, ConvNet
from segment_by_bars import segment_notes_by_bars

note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]
note_to_idx = {note: idx for idx, note in enumerate(note_labels)}

DURATION_BINS = [0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]

# DRUM KIT MAPPING
DRUM_KIT_BASE = "/Users/shreyagupta/Desktop/AI_Research_Data/drums"

# Define which drum samples to use
DRUM_SAMPLES = {
    'snare': f"{DRUM_KIT_BASE}/Snare/Air Layered Snare.wav",
    'kick': f"{DRUM_KIT_BASE}/Kick/Afterparty Kick 2.wav",
    'ride': f"{DRUM_KIT_BASE}/Ride/TS_VD_ride_mellow_ping.wav",
    'hihat': f"{DRUM_KIT_BASE}/Closed Hihat/60_s Hat Brush 2.wav",
    'tomb': f"{DRUM_KIT_BASE}/Tomb/BOS_PMHH_Percussion_Tom_One_Shot_Standard_Tomb.wav"
}

# TABLA → DRUM MAPPING (from mapping.png)
TABLA_TO_DRUM_MAPPING = {
    'Dha': ['snare', 'ride'],      # Snare drum + Ride
    'Dhin': ['kick', 'ride'],      # Bass drum + Ride
    'Ghe': ['kick'],               # Bass drum
    'Na': ['snare'],               # Snare drum
    'Ta': ['ride'],                # Ride
    'Tun': ['tomb', 'ride'],       # Tomb + Ride
    'Tin': ['hihat'],              # Hihat
    'Ti': ['snare'],               # Snare Drum
    'Re': ['snare'],               # Snare Drum
    'Ki': ['snare'],               # Snare Drum
    'T': ['snare'],                # Snare Drum
    'Kat': ['snare']               # Snare Drum
}


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


# Cache for loaded drum samples
_drum_sample_cache = {}

def load_drum_sample(drum_type, sample_rate=44100):
    """Load and cache a drum sample"""
    if drum_type not in _drum_sample_cache:
        sample_path = DRUM_SAMPLES[drum_type]
        if os.path.exists(sample_path):
            audio, sr = librosa.load(sample_path, sr=sample_rate)
            _drum_sample_cache[drum_type] = audio
        else:
            print(f"  ⚠️  Warning: Drum sample not found: {sample_path}")
            _drum_sample_cache[drum_type] = None

    return _drum_sample_cache[drum_type]


def synthesize_drums_with_mapping(notes, durations, sample_rate=44100, crossfade_ms=10):
    """
    Synthesize drums using proper drum kit mapping

    Args:
        notes: List of tabla note names
        durations: List of durations in seconds
        sample_rate: Output sample rate
        crossfade_ms: Crossfade duration
    """
    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    audio_segments = []

    for i, (note, duration) in enumerate(zip(notes, durations)):
        # Get drum pieces for this tabla note
        drum_pieces = TABLA_TO_DRUM_MAPPING.get(note, ['snare'])  # Default to snare if not found

        # Load and mix drum samples
        mixed_audio = None

        for drum_type in drum_pieces:
            drum_sample = load_drum_sample(drum_type, sample_rate)

            if drum_sample is not None:
                # Trim or pad to target duration
                target_samples = int(sample_rate * duration)

                if len(drum_sample) > target_samples:
                    audio_data = drum_sample[:target_samples].copy()
                else:
                    # Pad with silence if needed
                    audio_data = np.pad(drum_sample, (0, max(0, target_samples - len(drum_sample))))

                # Mix samples (if multiple drum pieces)
                if mixed_audio is None:
                    mixed_audio = audio_data
                else:
                    # Ensure same length
                    min_len = min(len(mixed_audio), len(audio_data))
                    mixed_audio = mixed_audio[:min_len] + audio_data[:min_len]

        if mixed_audio is not None:
            # Apply fade out to end
            fade_samples = min(crossfade_samples, len(mixed_audio) // 4)
            if fade_samples > 0:
                fade_out = np.linspace(1, 0, fade_samples)
                mixed_audio[-fade_samples:] *= fade_out

            audio_segments.append(mixed_audio)

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


def synthesize_tabla(notes, durations, folder='tabla', sample_rate=44100, crossfade_ms=10):
    """Synthesize tabla audio"""
    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    audio_segments = []

    for note, duration in zip(notes, durations):
        audio_file = f"{folder}/{note}.wav"

        if not os.path.exists(audio_file):
            continue

        try:
            audio_data, sr = librosa.load(audio_file, sr=sample_rate)
            target_samples = int(sample_rate * duration)

            if len(audio_data) > target_samples:
                audio_data = audio_data[:target_samples]

            fade_samples = min(crossfade_samples, len(audio_data) // 4)
            if fade_samples > 0:
                fade_out = np.linspace(1, 0, fade_samples)
                audio_data[-fade_samples:] *= fade_out

            audio_segments.append(audio_data)

        except Exception as e:
            pass

    if not audio_segments:
        return np.array([]), sample_rate

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

    max_val = np.abs(full_audio).max()
    if max_val > 0:
        full_audio = full_audio / max_val * 0.9

    return full_audio, sample_rate


def generate_with_swing(audio_file, model_path, num_generate=32, temperature=1.0):
    """Generate with swing-adjusted timing"""
    cnn_model = ConvNet(input_channels=13, num_classes=12)
    cnn_model.load_state_dict(torch.load('ConvNet_SNFPR_model.pth'))
    cnn_model.eval()

    notes, durations, onset_samples = classify_tabla_file(audio_file, cnn_model, target_length=72000)

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
        return None, None, None, None

    duration_clusters = extract_swing_template(swing_result, durations)

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

    seed_bars = bars[-2:]
    seed_notes = []
    seed_durations = []

    for bar in seed_bars:
        seed_notes.extend([note_to_idx[n] for n in bar['notes']])
        seed_durations.extend([quantize_duration(d) for d in bar['durations']])

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

    last_bar = bars[-1]
    context_notes = last_bar['notes']
    context_durations = last_bar['durations']

    return context_notes, context_durations, gen_note_labels, gen_durations_swing


def create_marker_beep(sample_rate=44100, duration_s=0.3, freq=800):
    """Create transition marker beep"""
    marker_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, marker_samples)
    marker_beep = 0.3 * np.sin(2 * np.pi * freq * t)
    fade_samples = int(sample_rate * 0.05)
    marker_beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
    marker_beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return marker_beep


def export_proper_drum_mapping(output_dir='generated_proper_drums'):
    """Export with proper drum kit mapping"""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print(" " * 20 + "EXPORTING WITH PROPER DRUM KIT MAPPING")
    print("=" * 100)
    print(f"\nOutput directory: {output_dir}")
    print("\nDrum Mapping (from mapping.png):")
    for tabla_note, drum_pieces in sorted(TABLA_TO_DRUM_MAPPING.items()):
        drums_str = " + ".join(drum_pieces)
        print(f"  {tabla_note:5s} → {drums_str}")
    print("\nFile structure:")
    print("  1. 🎵 CONTEXT: Tabla (input)")
    print("  2. 📢 MARKER: Beep")
    print("  3. 🎵 GENERATION: Tabla")
    print("  4. 📢 MARKER: Beep")
    print("  5. 🥁 GENERATION: Drums (proper mapping)")
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

        context_notes, context_durations, gen_notes, gen_durations = generate_with_swing(
            ektaal_file,
            model_info['path'],
            num_generate=32,
            temperature=1.0
        )

        if context_notes is None:
            print(f"   ❌ Generation failed")
            continue

        print(f"\n   📥 CONTEXT: {' '.join(context_notes[:10])}... ({sum(context_durations):.2f}s)")
        print(f"   📤 GENERATED: {' '.join(gen_notes[:10])}... ({sum(gen_durations):.2f}s)")

        # Synthesize
        print(f"\n   🔨 Synthesizing context (tabla)...")
        context_audio, sr = synthesize_tabla(context_notes, context_durations)

        print(f"   🔨 Synthesizing generation (tabla)...")
        gen_tabla_audio, sr = synthesize_tabla(gen_notes, gen_durations)

        print(f"   🔨 Synthesizing generation (drums with proper mapping)...")
        gen_drums_audio, sr = synthesize_drums_with_mapping(gen_notes, gen_durations)

        if len(context_audio) == 0 or len(gen_drums_audio) == 0:
            print(f"   ❌ Synthesis failed")
            continue

        # Create markers
        marker_beep = create_marker_beep(sr)
        silence = np.zeros(int(sr * 0.1))

        # Concatenate
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
        output_file = os.path.join(output_dir, f"{model_name}_PROPER_DRUMS.wav")
        sf.write(output_file, full_audio, sr)
        exported_files.append(output_file)

        print(f"   ✅ Exported: {output_file}")
        print(f"   Total duration: {len(full_audio)/sr:.2f}s")

    print(f"\n{'=' * 100}")
    print("✅ EXPORT COMPLETE!")
    print(f"{'=' * 100}")
    print(f"\n📁 Exported {len(exported_files)} files with proper drum mapping")
    print()

    return exported_files


if __name__ == "__main__":
    files = export_proper_drum_mapping()
