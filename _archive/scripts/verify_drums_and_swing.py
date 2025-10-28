"""
Verify drum mapping and swing timing
"""

import os
import numpy as np
import soundfile as sf
import librosa

print("=" * 80)
print("VERIFICATION: DRUM MAPPING & SWING TIMING")
print("=" * 80)

# 1. VERIFY DRUM MAPPING
print("\n1. DRUM MAPPING VERIFICATION")
print("-" * 80)

drum_folder = "drums_fixed"  # Using onset-trimmed drums
tabla_folder = "tabla"

note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]

print(f"\nChecking if drum samples exist for all {len(note_labels)} notes:")
for note in note_labels:
    drum_file = f"{drum_folder}/{note}.wav"
    tabla_file = f"{tabla_folder}/{note}.wav"

    drum_exists = os.path.exists(drum_file)
    tabla_exists = os.path.exists(tabla_file)

    status = "✅" if drum_exists else "❌"
    print(f"  {status} {note}: drums/{note}.wav {'EXISTS' if drum_exists else 'MISSING'}")

# 2. SAMPLE COMPARISON: Tabla vs Drums
print(f"\n2. SAMPLE COMPARISON (Tabla vs Drums)")
print("-" * 80)

test_notes = ["Dha", "Ti", "Ta"]
print(f"\nLoading sample notes to verify they're different:")

for note in test_notes:
    tabla_file = f"{tabla_folder}/{note}.wav"
    drum_file = f"{drum_folder}/{note}.wav"

    if os.path.exists(tabla_file) and os.path.exists(drum_file):
        tabla_audio, tabla_sr = librosa.load(tabla_file, sr=None)
        drum_audio, drum_sr = librosa.load(drum_file, sr=None)

        # Compare spectral features
        tabla_spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=tabla_audio, sr=tabla_sr))
        drum_spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=drum_audio, sr=drum_sr))

        diff = abs(tabla_spectral_centroid - drum_spectral_centroid)

        print(f"\n  {note}:")
        print(f"    Tabla: {len(tabla_audio)} samples, spectral centroid: {tabla_spectral_centroid:.1f} Hz")
        print(f"    Drum:  {len(drum_audio)} samples, spectral centroid: {drum_spectral_centroid:.1f} Hz")
        print(f"    Difference: {diff:.1f} Hz ({'DIFFERENT' if diff > 100 else 'SIMILAR'})")

# 3. VERIFY SWING TIMING IN EXPORTED FILES
print(f"\n3. SWING TIMING VERIFICATION")
print("-" * 80)

# Load one of the exported drum files
drum_export = "generated_drums/Model_A_Original_Reg_DRUMS.wav"
if os.path.exists(drum_export):
    audio, sr = sf.read(drum_export)

    # Detect onsets in the exported file
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')

    # Calculate inter-onset intervals (IOIs)
    iois = np.diff(onsets)

    print(f"\nAnalyzing: {drum_export}")
    print(f"  Total duration: {len(audio)/sr:.2f}s")
    print(f"  Detected onsets: {len(onsets)}")
    print(f"  Inter-onset intervals (IOIs):")
    print(f"    Min:  {iois.min():.3f}s")
    print(f"    Max:  {iois.max():.3f}s")
    print(f"    Mean: {iois.mean():.3f}s")
    print(f"    Std:  {iois.std():.3f}s")

    # Check if timing is variable (swing) vs fixed (no swing)
    cv = iois.std() / iois.mean()  # Coefficient of variation
    print(f"\n  Coefficient of Variation: {cv:.3f}")
    if cv > 0.15:
        print(f"  ✅ SWING DETECTED: Timing is variable (CV > 0.15)")
    else:
        print(f"  ⚠️  NO SWING: Timing is too regular (CV < 0.15)")

    # Show first 20 IOIs
    print(f"\n  First 20 IOIs (seconds):")
    print(f"  {[f'{ioi:.3f}' for ioi in iois[:20]]}")

else:
    print(f"  ❌ File not found: {drum_export}")

# 4. COMPARE CONTEXT (TABLA) VS GENERATION (DRUMS) TIMING
print(f"\n4. CONTEXT vs GENERATION TIMING")
print("-" * 80)

# Load the exported file and try to identify the transition point
if os.path.exists(drum_export):
    # Find the beep marker (800 Hz tone)
    print(f"\nSearching for transition marker (beep at ~800 Hz)...")

    # Use a sliding window to find high frequency content
    hop_length = 512
    stft = librosa.stft(audio, hop_length=hop_length)
    freq_bins = librosa.fft_frequencies(sr=sr)

    # Find 800 Hz bin
    target_freq = 800
    target_bin = np.argmin(np.abs(freq_bins - target_freq))

    # Get energy at 800 Hz over time
    energy_800 = np.abs(stft[target_bin, :])

    # Find peak (beep location)
    peak_frame = np.argmax(energy_800)
    peak_time = librosa.frames_to_time(peak_frame, sr=sr, hop_length=hop_length)

    print(f"  Beep marker detected at: {peak_time:.2f}s")
    print(f"  → Context (tabla): 0.00s - {peak_time:.2f}s")
    print(f"  → Generation (drums): {peak_time:.2f}s - {len(audio)/sr:.2f}s")

    # Analyze timing in each section
    context_onsets = onsets[onsets < peak_time - 0.5]  # before beep
    gen_onsets = onsets[onsets > peak_time + 0.5]  # after beep

    if len(context_onsets) > 1:
        context_iois = np.diff(context_onsets)
        print(f"\n  Context (Tabla) timing:")
        print(f"    Notes: {len(context_onsets)}")
        print(f"    IOI mean: {context_iois.mean():.3f}s, std: {context_iois.std():.3f}s")
        print(f"    CV: {context_iois.std() / context_iois.mean():.3f}")

    if len(gen_onsets) > 1:
        gen_iois = np.diff(gen_onsets)
        print(f"\n  Generation (Drums) timing:")
        print(f"    Notes: {len(gen_onsets)}")
        print(f"    IOI mean: {gen_iois.mean():.3f}s, std: {gen_iois.std():.3f}s")
        print(f"    CV: {gen_iois.std() / gen_iois.mean():.3f}")

        if gen_iois.std() / gen_iois.mean() > 0.15:
            print(f"    ✅ Drums are using SWING timing")
        else:
            print(f"    ⚠️  Drums timing is too regular")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
