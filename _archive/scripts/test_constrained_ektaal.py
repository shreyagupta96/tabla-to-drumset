"""
Test Constrained Generation on Real Ektaal File
Same test as before, but with "Ti Re Ki T" rule embedded
"""

import torch
import sys
import os
import glob

sys.path.insert(0, "/Users/shreyagupta/Desktop/Research MFP/B264026_Masters_Final_Project_SG/Submission/Code/HybridApproach")

from meter_conditional_lstm import create_model
from hybrid_meter_pipeline import hybrid_meter
from batch_classify_long_files import classify_tabla_file, ConvNet
from segment_by_bars import segment_notes_by_bars
from constrained_generation import generate_with_rules

note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]
note_to_idx = {note: idx for idx, note in enumerate(note_labels)}

DURATION_BINS = [0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]

def quantize_duration(duration):
    for i, threshold in enumerate(DURATION_BINS[1:]):
        if duration < threshold:
            return i
    return len(DURATION_BINS) - 2

# Find Ektaal Variation 2 file
base = "/Users/shreyagupta/Desktop/AI_Research_Data/From Tabla Player"
pattern = os.path.join(base, "Taal Variations*/Bounces/Ektaal Variation 2#03.wav")
files = glob.glob(pattern)

if not files:
    print("❌ Test file not found!")
    exit(1)

audio_file = files[0]

print("=" * 100)
print(" " * 25 + "CONSTRAINED BAR-AWARE LSTM TEST")
print(" " * 30 + "(With Ti Re Ki T Rule)")
print("=" * 100)

print(f"\n📂 Test file: {os.path.basename(audio_file)}")

# Step 1: Classify
print(f"\n{'='*100}")
print("STEP 1: CLASSIFY NOTES WITH CNN")
print(f"{'='*100}")

cnn_model = ConvNet(input_channels=13, num_classes=12)
cnn_model.load_state_dict(torch.load('ConvNet_SNFPR_model.pth'))
cnn_model.eval()

notes, durations, onset_samples = classify_tabla_file(audio_file, cnn_model, target_length=72000)

print(f"\n✅ Classification: {len(notes)} notes")
print(f"   First 20: {' '.join(notes[:20])}")

# Step 2: Meter detection
print(f"\n{'='*100}")
print("STEP 2: METER DETECTION & BAR SEGMENTATION")
print(f"{'='*100}")

meter_result = hybrid_meter(audio_file)
meter = meter_result.get('final_meter')
tempo = meter_result.get('tempo')
bar_start_samples = meter_result.get('bar_start_samples', [])

if meter == 12:
    taal = 'Ektaal'
    taal_id = 1
else:
    taal = 'Unknown'
    taal_id = 2

print(f"\n✅ Meter: {taal} (meter={meter}, tempo={tempo:.1f} BPM)")
print(f"   Bar boundaries: {len(bar_start_samples)} samples")

bars = segment_notes_by_bars(notes, durations, onset_samples, bar_start_samples)

print(f"✅ Segmentation: {len(bars)} complete bars")
if len(bars) > 0:
    avg_notes = sum(b['num_notes'] for b in bars) / len(bars)
    print(f"   Notes per bar: avg={avg_notes:.1f}")

print(f"\n📊 Last 2 bars (seed):")
for bar in bars[-2:]:
    print(f"   Bar {bar['bar_num']}: {' '.join(bar['notes'])}")

if len(bars) < 3:
    print(f"\n❌ Need at least 3 bars")
    exit(1)

# Step 3: Generate with CONSTRAINED LSTM (with Ti Re Ki T rule)
print(f"\n{'='*100}")
print("STEP 3: GENERATE WITH CONSTRAINED BAR-AWARE LSTM")
print(f"{'='*100}")

checkpoint = torch.load('models/best_bar_aware_lstm.pth', map_location='cpu')
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

print(f"✅ Model loaded (epoch {checkpoint['epoch']+1}, val_loss={checkpoint['val_loss']:.4f})")

# Use last 2 bars as seed
seed_bars = bars[-2:]
seed_notes = []
seed_durations = []

for bar in seed_bars:
    seed_notes.extend([note_to_idx[n] for n in bar['notes']])
    seed_durations.extend([quantize_duration(d) for d in bar['durations']])

print(f"\n🌱 Seed ({len(seed_notes)} notes from 2 bars):")
for bar in seed_bars:
    print(f"   Bar {bar['bar_num']}: {' '.join(bar['notes'])}")

# Define rules
rules = [
    ['Ti', 'Re', 'Ki', 'T']  # Ti Re Ki T must always appear in sequence
]

print(f"\n📜 Active Rules:")
for i, rule in enumerate(rules, 1):
    print(f"   Rule {i}: {' → '.join(rule)}")

# Generate with rules
gen_notes, gen_durations, rule_apps = generate_with_rules(
    model=model,
    seed_notes=seed_notes,
    seed_durations=seed_durations,
    taal_id=taal_id,
    note_labels=note_labels,
    rules=rules,
    num_generate=32,
    temperature=1.0,
    device='cpu'
)

gen_note_labels = [note_labels[idx] for idx in gen_notes]

# Analysis
print(f"\n{'='*100}")
print("GENERATION QUALITY ANALYSIS")
print(f"{'='*100}")

print(f"\n🎵 Generated sequence ({len(gen_note_labels)} notes):")
print(f"   {' '.join(gen_note_labels)}")

# Check for Ti Re Ki T
full_sequence = [note_labels[i] for i in seed_notes] + gen_note_labels
print(f"\n🔍 Full sequence (seed + generated):")
wrapped = ' '.join(full_sequence)
if len(wrapped) > 100:
    print(f"   {wrapped[:100]}...")
    print(f"   ...{wrapped[-100:]}")
else:
    print(f"   {wrapped}")

# Find Ti Re Ki T patterns
pattern = ['Ti', 'Re', 'Ki', 'T']
pattern_locations = []
for i in range(len(full_sequence) - 3):
    if full_sequence[i:i+4] == pattern:
        pattern_locations.append(i)

if pattern_locations:
    print(f"\n✅ 'Ti Re Ki T' found at {len(pattern_locations)} location(s):")
    for loc in pattern_locations:
        context_start = max(0, loc - 3)
        context_end = min(len(full_sequence), loc + 7)
        context = ' '.join(full_sequence[context_start:context_end])
        print(f"   Position {loc}: ...{context}...")

        # Check if this was in seed or generated
        if loc < len(seed_notes):
            print(f"      → In SEED (already present)")
        else:
            gen_pos = loc - len(seed_notes)
            if gen_pos in rule_apps:
                print(f"      → GENERATED by rule at step {gen_pos}")
            else:
                print(f"      → Generated naturally")
else:
    print(f"\n⚠️  'Ti Re Ki T' not found in full sequence")

# Max consecutive repeats
max_repeat = 1
current_repeat = 1
for i in range(1, len(gen_notes)):
    if gen_notes[i] == gen_notes[i-1]:
        current_repeat += 1
        max_repeat = max(max_repeat, current_repeat)
    else:
        current_repeat = 1

# Unique notes
unique_notes = len(set(gen_notes))
unique_ratio = unique_notes / len(gen_notes)

print(f"\n📊 Quality Metrics:")
print(f"   Max consecutive repeats: {max_repeat}")
print(f"   Unique notes: {unique_notes}/{len(gen_notes)} ({unique_ratio:.1%})")
print(f"   Rules triggered: {len(rule_apps)} times at positions {rule_apps}")

from collections import Counter
note_counts = Counter(gen_note_labels)
print(f"\n   Note distribution:")
for note, count in note_counts.most_common():
    print(f"      {note}: {count}")

print(f"\n{'='*100}")
print("✅ CONSTRAINED GENERATION TEST COMPLETE!")
print(f"{'='*100}")
