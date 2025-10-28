"""
Test generation with Teental Dugun input
"""

import torch
import json
import sys
import numpy as np

# Load models
sys.path.insert(0, "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG Copy 3")
from lstm_model import create_model as create_model_copy3
sys.path.pop(0)
from lstm_model import create_model as create_model_current

def load_model(model_path, create_fn):
    checkpoint = torch.load(model_path, map_location='cpu')
    vocab_size = checkpoint['vocab_size']
    model = create_fn(vocab_size=vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    note_to_idx = checkpoint['note_to_idx']
    idx_to_note = {int(k): v for k, v in checkpoint['idx_to_note'].items()}
    return model, note_to_idx, idx_to_note

def quantize_duration(duration, bins=[0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]):
    for i, edge in enumerate(bins[1:]):
        if duration < edge:
            return i
    return len(bins) - 2

def duration_bins_to_seconds(duration_bins):
    bin_midpoints = [0.15, 0.4, 0.65, 1.15, 2.0]
    return [bin_midpoints[idx] for idx in duration_bins]

def generate_response(model, seed_notes, seed_durations, note_to_idx, idx_to_note,
                     num_generate, temperature=1.0):
    seed_window = min(8, len(seed_notes))
    seed_notes_subset = seed_notes[-seed_window:]
    seed_durations_subset = seed_durations[-seed_window:]

    seed_note_indices = [note_to_idx[note] for note in seed_notes_subset]
    seed_duration_bins = [quantize_duration(d) for d in seed_durations_subset]

    gen_note_indices, gen_dur_indices = model.generate(
        seed_notes=seed_note_indices,
        seed_durations=seed_duration_bins,
        num_generate=num_generate,
        temperature=temperature,
        device='cpu'
    )

    gen_note_names = [idx_to_note[idx] for idx in gen_note_indices]
    gen_durations = duration_bins_to_seconds(gen_dur_indices)

    return gen_note_names, gen_durations

def analyze_pattern(notes):
    """Analyze repetition patterns in generated sequence"""
    # Find consecutive repetitions
    max_rep = 1
    current_rep = 1
    rep_note = None

    for i in range(1, len(notes)):
        if notes[i] == notes[i-1]:
            current_rep += 1
            if current_rep > max_rep:
                max_rep = current_rep
                rep_note = notes[i]
        else:
            current_rep = 1

    # Count unique notes
    unique_notes = len(set(notes))

    # Find most common note
    from collections import Counter
    note_counts = Counter(notes)
    most_common = note_counts.most_common(3)

    return {
        'unique_notes': unique_notes,
        'total_notes': len(notes),
        'max_repetition': max_rep,
        'repeated_note': rep_note,
        'most_common': most_common
    }

print("=" * 100)
print(" " * 25 + "TEENTAL DUGUN GENERATION TEST")
print("=" * 100)

# Load classified Teental data
with open("teental_dugun_classification.json", "r") as f:
    data = json.load(f)

input_notes = data['notes']
input_durations = data['durations']

print(f"\n📂 Input: {len(input_notes)} classified tabla strokes")
print(f"📊 First 32 notes: {' '.join(input_notes[:32])}")
print(f"📊 Last 8 notes (seed): {' '.join(input_notes[-8:])}")

# Load both models
print("\n📦 Loading models...")
copy3_model, copy3_note_to_idx, copy3_idx_to_note = load_model(
    "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG Copy 3/tabla_lstm_model.pth",
    create_model_copy3
)
current_model, current_note_to_idx, current_idx_to_note = load_model(
    "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/tabla_lstm_model.pth",
    create_model_current
)
print("✅ Models loaded")

# Generate with both models
temperature = 1.0
num_generate = 32  # Generate 32 notes

print(f"\n🔄 Generating {num_generate} notes at temperature {temperature}...")

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

copy3_notes, copy3_durs = generate_response(
    copy3_model, input_notes, input_durations,
    copy3_note_to_idx, copy3_idx_to_note,
    num_generate=num_generate,
    temperature=temperature
)

torch.manual_seed(42)
np.random.seed(42)

current_notes, current_durs = generate_response(
    current_model, input_notes, input_durations,
    current_note_to_idx, current_idx_to_note,
    num_generate=num_generate,
    temperature=temperature
)

print("\n" + "=" * 100)
print("📊 GENERATED PATTERNS")
print("=" * 100)

print(f"\n🔴 COPY 3 (No Embedding Regularization):")
print(f"   {' '.join(copy3_notes)}")

print(f"\n🟢 CURRENT (WITH Embedding Regularization):")
print(f"   {' '.join(current_notes)}")

# Analyze patterns
print("\n" + "=" * 100)
print("📊 PATTERN ANALYSIS")
print("=" * 100)

copy3_analysis = analyze_pattern(copy3_notes)
current_analysis = analyze_pattern(current_notes)

print(f"\n🔴 COPY 3:")
print(f"   Unique notes: {copy3_analysis['unique_notes']}/{copy3_analysis['total_notes']}")
print(f"   Max consecutive repetition: {copy3_analysis['max_repetition']}x ({copy3_analysis['repeated_note']})")
print(f"   Most common notes: {', '.join([f'{note}({count})' for note, count in copy3_analysis['most_common']])}")

print(f"\n🟢 CURRENT:")
print(f"   Unique notes: {current_analysis['unique_notes']}/{current_analysis['total_notes']}")
print(f"   Max consecutive repetition: {current_analysis['max_repetition']}x ({current_analysis['repeated_note']})")
print(f"   Most common notes: {', '.join([f'{note}({count})' for note, count in current_analysis['most_common']])}")

# Check for embedding similarity features
print("\n" + "=" * 100)
print("📊 EMBEDDING SIMILARITY FEATURES")
print("=" * 100)

copy3_has_na_ta = 'Na' in copy3_notes and 'Ta' in copy3_notes
current_has_na_ta = 'Na' in current_notes and 'Ta' in current_notes

copy3_has_tin_tun = 'Tin' in copy3_notes and 'Tun' in copy3_notes
current_has_tin_tun = 'Tin' in current_notes and 'Tun' in current_notes

copy3_has_dha_dhin = 'Dha' in copy3_notes and 'Dhin' in copy3_notes
current_has_dha_dhin = 'Dha' in current_notes and 'Dhin' in current_notes

print(f"\nNa+Ta variation:")
print(f"   Copy 3:  {'✅ Yes' if copy3_has_na_ta else '❌ No'}")
print(f"   Current: {'✅ Yes' if current_has_na_ta else '❌ No'}")

print(f"\nTin+Tun variation:")
print(f"   Copy 3:  {'✅ Yes' if copy3_has_tin_tun else '❌ No'}")
print(f"   Current: {'✅ Yes' if current_has_tin_tun else '❌ No'}")

print(f"\nDha+Dhin variation:")
print(f"   Copy 3:  {'✅ Yes' if copy3_has_dha_dhin else '❌ No'}")
print(f"   Current: {'✅ Yes' if current_has_dha_dhin else '❌ No'}")

print("\n" + "=" * 100)
print("✅ Analysis complete!")
print("=" * 100)
