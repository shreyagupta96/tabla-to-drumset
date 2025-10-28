"""
Diagnose if drum samples are being synthesized with correct swing durations
"""

import os
import numpy as np
import soundfile as sf
import librosa
import torch
import sys

sys.path.insert(0, "/Users/shreyagupta/Desktop/Research MFP/B264026_Masters_Final_Project_SG/Submission/Code/HybridApproach")

from export_tabla_and_drums import generate_with_swing

print("="*80)
print("DIAGNOSING DRUM TIMING TRANSFER")
print("="*80)

ektaal_file = "/Users/shreyagupta/Desktop/AI_Research_Data/Tabla_files/Ektaal.wav"
model_path = "models/best_bar_aware_lstm.pth"

# Generate with swing
context_notes, context_durations, gen_notes, gen_durations, swing_stats = generate_with_swing(
    ektaal_file,
    model_path,
    num_generate=32,
    temperature=1.0
)

print(f"\n{'='*80}")
print("REQUESTED DURATIONS (from swing template)")
print(f"{'='*80}")
print(f"First 10 durations: {[f'{d:.3f}' for d in gen_durations[:10]]}")
print(f"Mean: {np.mean(gen_durations):.3f}s")
print(f"Std:  {np.std(gen_durations):.3f}s")
print(f"CV:   {np.std(gen_durations)/np.mean(gen_durations):.3f}")

# Check drum sample lengths
drum_folder = '/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/drums'
note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]

print(f"\n{'='*80}")
print("DRUM SAMPLE DURATIONS")
print(f"{'='*80}")

drum_lengths = {}
for note in note_labels:
    drum_file = f"{drum_folder}/{note}.wav"
    if os.path.exists(drum_file):
        audio, sr = librosa.load(drum_file, sr=44100)
        duration = len(audio) / sr
        drum_lengths[note] = duration
        print(f"{note:5s}: {duration:.3f}s")
    else:
        print(f"{note:5s}: FILE NOT FOUND")

# Check if requested durations exceed sample lengths
print(f"\n{'='*80}")
print("DURATION MISMATCH ANALYSIS")
print(f"{'='*80}")

mismatches = []
for i, (note, requested_dur) in enumerate(zip(gen_notes[:20], gen_durations[:20])):
    sample_dur = drum_lengths.get(note, 0)

    if requested_dur > sample_dur:
        status = "⚠️ TRUNCATED"
        mismatches.append((note, requested_dur, sample_dur))
    elif requested_dur < sample_dur * 0.8:  # If we're only using <80% of sample
        status = "✓ Using partial"
    else:
        status = "✓ OK"

    print(f"{i:2d}. {note:5s}: Requested {requested_dur:.3f}s | Sample {sample_dur:.3f}s | {status}")

if mismatches:
    print(f"\n⚠️  Found {len(mismatches)} cases where requested duration exceeds sample length!")
    print("This means the actual duration will be shorter than requested,")
    print("breaking the swing timing synchronization between tabla and drums.")
else:
    print("\n✓ All requested durations fit within sample lengths")

# Now let's check what the actual synthesized drum audio looks like
print(f"\n{'='*80}")
print("ANALYZING ACTUAL SYNTHESIZED DRUM AUDIO")
print(f"{'='*80}")

# Read one of the exported files
drum_file = "generated_tabla_and_drums/Model_A_Original_Reg_TABLA_AND_DRUMS.wav"
if os.path.exists(drum_file):
    audio, sr = sf.read(drum_file)

    # Detect onsets
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')

    print(f"Total onsets detected: {len(onsets)}")

    # Find the beep markers (they should be around 800Hz)
    # The structure is: context + beep + tabla + beep + drums
    # So we need to find where the drums section starts

    # The beep is at 800Hz, so we can detect it by looking for spectral peaks
    spec = np.abs(librosa.stft(audio))
    freqs = librosa.fft_frequencies(sr=sr)
    beep_freq_idx = np.argmin(np.abs(freqs - 800))
    beep_energy = spec[beep_freq_idx, :]

    # Find peaks in beep energy
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(beep_energy, height=np.max(beep_energy) * 0.3)
    peak_times = librosa.frames_to_time(peaks, sr=sr)

    if len(peak_times) >= 2:
        print(f"\nBeep markers found at: {peak_times[:2]}")

        # Drums section starts after second beep
        drums_start = peak_times[1] + 0.5  # Add buffer
        drum_onsets = onsets[onsets > drums_start]

        print(f"\nDrum section onsets: {len(drum_onsets)}")

        if len(drum_onsets) > 1:
            drum_iois = np.diff(drum_onsets)
            print(f"\nDrum IOIs (inter-onset intervals):")
            print(f"  First 10: {[f'{ioi:.3f}' for ioi in drum_iois[:10]]}")
            print(f"  Mean: {np.mean(drum_iois):.3f}s")
            print(f"  Std:  {np.std(drum_iois):.3f}s")
            print(f"  CV:   {np.std(drum_iois)/np.mean(drum_iois):.3f}")

            # Compare with requested durations
            print(f"\nCOMPARISON:")
            print(f"  Requested durations CV: {np.std(gen_durations)/np.mean(gen_durations):.3f}")
            print(f"  Actual drum IOIs CV:    {np.std(drum_iois)/np.mean(drum_iois):.3f}")

            if np.std(drum_iois)/np.mean(drum_iois) < 0.5 * (np.std(gen_durations)/np.mean(gen_durations)):
                print(f"\n❌ Drum timing variation is much lower than requested!")
                print(f"   This confirms swing is NOT being transferred correctly.")
            else:
                print(f"\n✓ Drum timing variation matches requested durations")
