"""
Export audio files for all 4 model generations
Synthesizes actual audio from the generated note sequences
"""

import os
import numpy as np
import soundfile as sf
import librosa

# Generated sequences from all 4 models
MODEL_OUTPUTS = {
    'Model_A_Original_Reg': {
        'notes': ['Ghe', 'Ghe', 'T', 'Ti', 'T', 'Ghe', 'Ti', 'Dha', 'Ghe', 'T', 'T', 'Dha', 'Ki', 'T', 'T', 'Re', 'T', 'Dhin', 'Dha', 'Dhin', 'Kat', 'Re', 'Ghe', 'Ta', 'T', 'T', 'Dhin', 'T', 'Re', 'Ghe', 'Ki', 'Kat'],
        'description': 'Original Data + Regularization',
        'val_loss': 3.0473,
        'max_repeats': 2,
        'unique_ratio': 0.281
    },
    'Model_B_Original_NoReg': {
        'notes': ['Ta', 'Re', 'Ghe', 'Tun', 'Re', 'Na', 'Dha', 'Dhin', 'T', 'T', 'T', 'Dhin', 'Re', 'T', 'Dha', 'Ghe', 'Ki', 'T', 'Re', 'Ghe', 'Dha', 'T', 'Dhin', 'Ghe', 'Ta', 'T', 'Ta', 'Re', 'Ghe', 'Dhin', 'T', 'Ki'],
        'description': 'Original Data + No Regularization',
        'val_loss': 3.0094,
        'max_repeats': 3,
        'unique_ratio': 0.281
    },
    'Model_C_Corrected_Reg': {
        'notes': ['Dhin', 'Tin', 'T', 'Dhin', 'Na', 'T', 'T', 'Re', 'Dhin', 'Ki', 'Dhin', 'Na', 'Dhin', 'Ghe', 'Ghe', 'Tin', 'Ghe', 'Kat', 'T', 'Tin', 'Dhin', 'Re', 'Dhin', 'Na', 'Ghe', 'T', 'Na', 'Tin', 'Na', 'Dha', 'Ki', 'T'],
        'description': 'Corrected Data (69 fixes) + Regularization',
        'val_loss': 3.0360,
        'max_repeats': 2,
        'unique_ratio': 0.281
    },
    'Model_D_Corrected_NoReg': {
        'notes': ['Tin', 'Na', 'T', 'Ta', 'Re', 'Dhin', 'Dhin', 'Re', 'Dhin', 'Dhin', 'Dha', 'T', 'Ta', 'Dha', 'Dhin', 'Kat', 'Ta', 'Dhin', 'Na', 'Dhin', 'Dhin', 'Dhin', 'T', 'T', 'Ghe', 'Dhin', 'Dhin', 'Re', 'Dhin', 'Dhin', 'Dhin', 'Dhin'],
        'description': 'Corrected Data (69 fixes) + No Regularization',
        'val_loss': 3.0375,
        'max_repeats': 4,
        'unique_ratio': 0.281
    }
}

def synthesize_audio(notes, tempo_bpm=100, sample_rate=44100, note_duration=0.3):
    """
    Synthesize audio from note sequence

    Args:
        notes: List of note names
        tempo_bpm: Tempo in BPM (affects spacing between notes)
        sample_rate: Output sample rate
        note_duration: Base duration for each note in seconds

    Returns:
        audio_array: Synthesized audio
        sample_rate: Sample rate of audio
    """
    tabla_folder = 'tabla'

    # Calculate inter-note spacing based on tempo
    beat_duration = 60.0 / tempo_bpm

    audio_segments = []

    print(f"   Synthesizing {len(notes)} notes...")

    for i, note in enumerate(notes):
        audio_file = f"{tabla_folder}/{note}.wav"

        if not os.path.exists(audio_file):
            print(f"   ⚠️  Warning: {audio_file} not found, using silence")
            # Use silence if note not found
            silence_samples = int(sample_rate * beat_duration)
            audio_segments.append(np.zeros(silence_samples))
            continue

        # Load the note sample
        try:
            audio_data, sr = librosa.load(audio_file, sr=sample_rate)

            # Apply fade in/out to avoid clicks
            fade_samples = int(sr * 0.005)  # 5ms fade
            fade_samples = min(fade_samples, len(audio_data) // 4)

            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples)
                audio_data[:fade_samples] *= fade_in
                fade_out = np.linspace(1, 0, fade_samples)
                audio_data[-fade_samples:] *= fade_out

            # Trim or pad to beat duration
            target_samples = int(sample_rate * beat_duration)
            if len(audio_data) > target_samples:
                audio_data = audio_data[:target_samples]
            else:
                # Pad with silence
                padding = np.zeros(target_samples - len(audio_data))
                audio_data = np.concatenate([audio_data, padding])

            audio_segments.append(audio_data)

        except Exception as e:
            print(f"   ⚠️  Error loading {audio_file}: {e}")
            silence_samples = int(sample_rate * beat_duration)
            audio_segments.append(np.zeros(silence_samples))

    # Concatenate all segments
    full_audio = np.concatenate(audio_segments)

    # Normalize to prevent clipping
    max_val = np.abs(full_audio).max()
    if max_val > 0:
        full_audio = full_audio / max_val * 0.9

    return full_audio, sample_rate


def export_all_models(output_dir='generated_audio', tempo_bpm=100):
    """Export audio for all 4 models"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print(" " * 30 + "EXPORTING AUDIO FOR ALL MODELS")
    print("=" * 100)
    print(f"\nOutput directory: {output_dir}")
    print(f"Tempo: {tempo_bpm} BPM")
    print(f"Total models: {len(MODEL_OUTPUTS)}")
    print()

    exported_files = []

    for model_name, model_data in MODEL_OUTPUTS.items():
        print(f"\n{'=' * 100}")
        print(f"🎵 {model_name}")
        print(f"{'=' * 100}")
        print(f"   Description: {model_data['description']}")
        print(f"   Val Loss: {model_data['val_loss']}")
        print(f"   Max Repeats: {model_data['max_repeats']}")
        print(f"   Unique Ratio: {model_data['unique_ratio']:.1%}")
        print(f"   Notes: {' '.join(model_data['notes'])}")

        # Synthesize audio
        audio, sr = synthesize_audio(
            model_data['notes'],
            tempo_bpm=tempo_bpm
        )

        # Generate filename
        output_file = os.path.join(output_dir, f"{model_name}.wav")

        # Save audio
        sf.write(output_file, audio, sr)
        exported_files.append(output_file)

        duration = len(audio) / sr
        print(f"   ✅ Exported: {output_file}")
        print(f"   Duration: {duration:.2f} seconds")

    print(f"\n{'=' * 100}")
    print("✅ EXPORT COMPLETE!")
    print(f"{'=' * 100}")
    print(f"\n📁 Exported {len(exported_files)} audio files:")
    for f in exported_files:
        print(f"   - {os.path.basename(f)}")

    print(f"\n💡 To play all files:")
    print(f"   open {output_dir}")
    print()

    return exported_files


if __name__ == "__main__":
    # Export with Ektaal tempo (100 BPM)
    files = export_all_models(tempo_bpm=100)

    # Automatically open the folder
    import subprocess
    subprocess.call(['open', 'generated_audio'])