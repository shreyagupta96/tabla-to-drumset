"""
Fix drum samples by trimming silence/buildup before the onset
This ensures drum attacks align with the start of the file, like tabla samples
"""

import os
import numpy as np
import soundfile as sf
import librosa

drum_folder = "drums"
drum_fixed_folder = "drums_fixed"

os.makedirs(drum_fixed_folder, exist_ok=True)

note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]

print("="*80)
print("FIXING DRUM SAMPLES - TRIMMING TO ONSET")
print("="*80)

for note in note_labels:
    drum_file = f"{drum_folder}/{note}.wav"

    if not os.path.exists(drum_file):
        print(f"\nSkipping {note} - file not found")
        continue

    # Load drum sample
    audio, sr = librosa.load(drum_file, sr=None)

    # Detect onset
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

    if len(onset_frames) == 0:
        print(f"\n{note}: No onset detected, keeping original")
        sf.write(f"{drum_fixed_folder}/{note}.wav", audio, sr)
        continue

    # Get the first strong onset
    onset_sample = librosa.frames_to_samples(onset_frames[0])

    # Add small pre-attack window (10ms) to preserve attack transient
    pre_attack_samples = int(sr * 0.010)  # 10ms
    trim_point = max(0, onset_sample - pre_attack_samples)

    # Trim audio from onset point
    trimmed_audio = audio[trim_point:]

    original_length_ms = len(audio) / sr * 1000
    trimmed_length_ms = len(trimmed_audio) / sr * 1000
    trimmed_amount_ms = (len(audio) - len(trimmed_audio)) / sr * 1000
    onset_time_ms = onset_sample / sr * 1000

    print(f"\n{note}:")
    print(f"  Original length: {original_length_ms:.1f}ms")
    print(f"  Onset detected at: {onset_time_ms:.1f}ms")
    print(f"  Trimmed: {trimmed_amount_ms:.1f}ms")
    print(f"  New length: {trimmed_length_ms:.1f}ms")

    # Save fixed sample
    output_file = f"{drum_fixed_folder}/{note}.wav"
    sf.write(output_file, trimmed_audio, sr)

print(f"\n{'='*80}")
print(f"COMPLETE! Fixed drum samples saved to: {drum_fixed_folder}/")
print(f"{'='*80}")
print(f"\nNext step: Update export_drum_mapping.py to use 'drums_fixed' folder")
print(f"This should preserve the swing timing in drum generation!")
