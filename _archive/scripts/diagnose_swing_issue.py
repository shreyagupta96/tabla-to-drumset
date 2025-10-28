"""
Diagnose why drums show different timing than tabla despite using same durations
"""

import os
import numpy as np
import soundfile as sf
import librosa

print("="*80)
print("DIAGNOSING SWING TIMING ISSUE")
print("="*80)

# Load the exported drum file
drum_file = "generated_drums/Model_A_Original_Reg_DRUMS.wav"

if not os.path.exists(drum_file):
    print(f"File not found: {drum_file}")
    exit(1)

audio, sr = sf.read(drum_file)
print(f"\nLoaded: {drum_file}")
print(f"Duration: {len(audio)/sr:.2f}s")

# Detect onsets
onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')

print(f"\nTotal onsets detected: {len(onsets)}")

# Find the beep marker (800 Hz tone)
hop_length = 512
stft = librosa.stft(audio, hop_length=hop_length)
freq_bins = librosa.fft_frequencies(sr=sr)
target_bin = np.argmin(np.abs(freq_bins - 800))
energy_800 = np.abs(stft[target_bin, :])
peak_frame = np.argmax(energy_800)
peak_time = librosa.frames_to_time(peak_frame, sr=sr, hop_length=hop_length)

print(f"\nBeep marker at: {peak_time:.2f}s")

# Split onsets
context_onsets = onsets[onsets < peak_time - 0.5]
drum_onsets = onsets[onsets > peak_time + 0.5]

print(f"\nContext (tabla) onsets: {len(context_onsets)}")
print(f"Generation (drums) onsets: {len(drum_onsets)}")

# Analyze both sections
if len(context_onsets) > 1:
    context_iois = np.diff(context_onsets)
    print(f"\nContext (Tabla) IOIs:")
    print(f"  First 10 IOIs: {[f'{ioi:.3f}' for ioi in context_iois[:10]]}")
    print(f"  Mean: {context_iois.mean():.3f}s, Std: {context_iois.std():.3f}s")
    print(f"  CV: {context_iois.std() / context_iois.mean():.3f}")

if len(drum_onsets) > 1:
    drum_iois = np.diff(drum_onsets)
    print(f"\nGeneration (Drums) IOIs:")
    print(f"  First 10 IOIs: {[f'{ioi:.3f}' for ioi in drum_iois[:10]]}")
    print(f"  Mean: {drum_iois.mean():.3f}s, Std: {drum_iois.std():.3f}s")
    print(f"  CV: {drum_iois.std() / drum_iois.mean():.3f}")

# Check if drum samples might have different attack characteristics
print(f"\n" + "="*80)
print("CHECKING DRUM SAMPLE CHARACTERISTICS")
print("="*80)

drum_folder = "drums"
tabla_folder = "tabla"
test_notes = ["Dha", "Ti", "Ta", "Na"]

for note in test_notes:
    drum_file = f"{drum_folder}/{note}.wav"
    tabla_file = f"{tabla_folder}/{note}.wav"

    if os.path.exists(drum_file) and os.path.exists(tabla_file):
        drum_audio, drum_sr = librosa.load(drum_file, sr=None)
        tabla_audio, tabla_sr = librosa.load(tabla_file, sr=None)

        # Detect onset in the sample itself
        drum_onset_env = librosa.onset.onset_strength(y=drum_audio, sr=drum_sr)
        tabla_onset_env = librosa.onset.onset_strength(y=tabla_audio, sr=tabla_sr)

        drum_peak_idx = np.argmax(drum_onset_env)
        tabla_peak_idx = np.argmax(tabla_onset_env)

        drum_peak_time = librosa.frames_to_time(drum_peak_idx, sr=drum_sr)
        tabla_peak_time = librosa.frames_to_time(tabla_peak_idx, sr=tabla_sr)

        print(f"\n{note}:")
        print(f"  Drum onset peak: {drum_peak_time*1000:.1f}ms")
        print(f"  Tabla onset peak: {tabla_peak_time*1000:.1f}ms")
        print(f"  Difference: {abs(drum_peak_time - tabla_peak_time)*1000:.1f}ms")
        print(f"  Drum length: {len(drum_audio)/drum_sr*1000:.1f}ms")
        print(f"  Tabla length: {len(tabla_audio)/tabla_sr*1000:.1f}ms")

print("\n" + "="*80)
print("POSSIBLE CAUSES:")
print("="*80)
print("1. If drum samples have later onset peaks, this could shift all detected onsets")
print("2. If drum samples have different decay, crossfading might work differently")
print("3. Onset detection might be less sensitive to drum transients")
