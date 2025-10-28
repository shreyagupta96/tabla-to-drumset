"""
Convert .aif files to .wav format for training
"""

import os
import librosa
import soundfile as sf
from pathlib import Path

input_dir = "training_data/Taal for training"
output_dir = "training_data/wav_files"

os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("CONVERTING .AIF TO .WAV")
print("=" * 80)

aif_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.aif')])

print(f"\nFound {len(aif_files)} .aif files to convert")

for i, aif_file in enumerate(aif_files, 1):
    input_path = os.path.join(input_dir, aif_file)
    output_filename = aif_file.replace('.aif', '.wav')
    output_path = os.path.join(output_dir, output_filename)

    print(f"\n[{i}/{len(aif_files)}] {aif_file}")

    try:
        # Load audio
        y, sr = librosa.load(input_path, sr=44100, mono=True)

        # Save as wav
        sf.write(output_path, y, sr)

        duration = len(y) / sr
        print(f"  ✓ Converted: {duration:.1f}s at {sr}Hz")

    except Exception as e:
        print(f"  ✗ Error: {e}")

print(f"\n{'=' * 80}")
print(f"✓ Conversion complete!")
print(f"  Output directory: {output_dir}")
print(f"{'=' * 80}")
