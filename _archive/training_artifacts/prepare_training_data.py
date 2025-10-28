"""
Phase 3: Training Dataset Preparation
Create taal-specific sequences (2-bar context → 1-bar target) for meter-conditional LSTM
"""

import json
import os
import pickle
import numpy as np
from collections import defaultdict

# Note vocabulary (same as original)
note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]
note_to_idx = {note: idx for idx, note in enumerate(note_labels)}

# Duration bins for quantization
DURATION_BINS = [0.0, 0.3, 0.5, 0.8, 1.5, np.inf]

def quantize_duration(duration):
    """Quantize duration into bins"""
    for i, threshold in enumerate(DURATION_BINS[1:]):
        if duration < threshold:
            return i
    return len(DURATION_BINS) - 2

def create_sequences_from_file(bars, taal_id, meter, filename):
    """
    Create training sequences from bars of a single file
    Format: 2-bar context → 1-bar target

    Returns list of sequences, each with:
    {
        'context_notes': [note indices from bars i and i+1],
        'context_durations': [duration bins from bars i and i+1],
        'target_notes': [note indices from bar i+2],
        'target_durations': [duration bins from bar i+2],
        'taal_id': 0 (Teental) or 1 (Ektaal),
        'meter': 16 or 12,
        'source_file': filename
    }
    """
    sequences = []

    if len(bars) < 3:
        return sequences  # Need at least 3 bars for 2-context + 1-target

    for i in range(len(bars) - 2):
        # Context: bars i and i+1
        context_notes = []
        context_durations = []

        for bar_idx in [i, i+1]:
            bar = bars[bar_idx]
            for note, dur in zip(bar['notes'], bar['durations']):
                context_notes.append(note_to_idx[note])
                context_durations.append(quantize_duration(dur))

        # Target: bar i+2
        target_bar = bars[i+2]
        target_notes = [note_to_idx[note] for note in target_bar['notes']]
        target_durations = [quantize_duration(dur) for dur in target_bar['durations']]

        sequences.append({
            'context_notes': context_notes,
            'context_durations': context_durations,
            'target_notes': target_notes,
            'target_durations': target_durations,
            'taal_id': taal_id,
            'meter': meter,
            'source_file': filename,
            'bar_range': f"{i}-{i+1} → {i+2}"
        })

    return sequences

def prepare_training_data(segmented_dir, output_dir, train_split=0.8):
    """
    Prepare complete training dataset from segmented bars
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print(" " * 30 + "PHASE 3: TRAINING DATA PREPARATION")
    print("=" * 100)

    # Load all segmented files
    bar_files = [f for f in os.listdir(segmented_dir) if f.endswith('_bars.json')]
    bar_files.sort()

    print(f"\n📋 Found {len(bar_files)} segmented files")

    # Group by taal
    teental_files = []
    ektaal_files = []

    for filename in bar_files:
        filepath = os.path.join(segmented_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

        if data['taal'] == 'Teental':
            teental_files.append((filename, data))
        elif data['taal'] == 'Ektaal':
            ektaal_files.append((filename, data))

    print(f"\n📊 File distribution:")
    print(f"   Teental: {len(teental_files)} files")
    print(f"   Ektaal: {len(ektaal_files)} files")

    # Create sequences
    print(f"\n🔄 Creating training sequences...")

    all_sequences = {
        'teental': [],
        'ektaal': []
    }

    # Teental sequences (taal_id=0)
    print(f"\n   Processing Teental files...")
    for filename, data in teental_files:
        seqs = create_sequences_from_file(
            bars=data['bars'],
            taal_id=0,
            meter=16,
            filename=data['file']
        )
        all_sequences['teental'].extend(seqs)
        print(f"      {data['file']:50} → {len(seqs):3} sequences")

    # Ektaal sequences (taal_id=1)
    print(f"\n   Processing Ektaal files...")
    for filename, data in ektaal_files:
        seqs = create_sequences_from_file(
            bars=data['bars'],
            taal_id=1,
            meter=12,
            filename=data['file']
        )
        all_sequences['ektaal'].extend(seqs)
        print(f"      {data['file']:50} → {len(seqs):3} sequences")

    # Statistics
    print(f"\n\n{'=' * 100}")
    print(" " * 35 + "SEQUENCE STATISTICS")
    print(f"{'=' * 100}")

    print(f"\n📈 Sequence counts:")
    print(f"   Teental sequences: {len(all_sequences['teental'])}")
    print(f"   Ektaal sequences: {len(all_sequences['ektaal'])}")
    print(f"   Total sequences: {len(all_sequences['teental']) + len(all_sequences['ektaal'])}")

    # Analyze sequence lengths
    print(f"\n📏 Sequence length statistics:")

    for taal_name in ['teental', 'ektaal']:
        seqs = all_sequences[taal_name]
        if len(seqs) > 0:
            context_lengths = [len(s['context_notes']) for s in seqs]
            target_lengths = [len(s['target_notes']) for s in seqs]

            print(f"\n   {taal_name.capitalize()}:")
            print(f"      Context length: min={min(context_lengths)}, max={max(context_lengths)}, avg={np.mean(context_lengths):.1f}")
            print(f"      Target length:  min={min(target_lengths)}, max={max(target_lengths)}, avg={np.mean(target_lengths):.1f}")

    # Split by files (not by sequences) to avoid data leakage
    print(f"\n\n{'=' * 100}")
    print(" " * 35 + "TRAIN/VALIDATION SPLIT")
    print(f"{'=' * 100}")

    # Split Teental files
    num_teental_train = int(len(teental_files) * train_split)
    teental_train_files = set([f[1]['file'] for f in teental_files[:num_teental_train]])

    # Split Ektaal files
    num_ektaal_train = int(len(ektaal_files) * train_split)
    ektaal_train_files = set([f[1]['file'] for f in ektaal_files[:num_ektaal_train]])

    # Separate sequences
    train_sequences = []
    val_sequences = []

    for seq in all_sequences['teental']:
        if seq['source_file'] in teental_train_files:
            train_sequences.append(seq)
        else:
            val_sequences.append(seq)

    for seq in all_sequences['ektaal']:
        if seq['source_file'] in ektaal_train_files:
            train_sequences.append(seq)
        else:
            val_sequences.append(seq)

    print(f"\n📂 Split summary:")
    print(f"   Training files:   {num_teental_train} Teental + {num_ektaal_train} Ektaal = {num_teental_train + num_ektaal_train} total")
    print(f"   Validation files: {len(teental_files) - num_teental_train} Teental + {len(ektaal_files) - num_ektaal_train} Ektaal = {len(teental_files) + len(ektaal_files) - num_teental_train - num_ektaal_train} total")
    print(f"\n   Training sequences:   {len(train_sequences)}")
    print(f"   Validation sequences: {len(val_sequences)}")

    # Save dataset
    dataset = {
        'train': train_sequences,
        'val': val_sequences,
        'metadata': {
            'note_labels': note_labels,
            'note_to_idx': note_to_idx,
            'duration_bins': DURATION_BINS,
            'num_classes': len(note_labels),
            'num_duration_bins': len(DURATION_BINS) - 1,
            'taal_mapping': {0: 'Teental', 1: 'Ektaal'},
            'meter_mapping': {0: 16, 1: 12},
            'total_files': len(bar_files),
            'teental_files': len(teental_files),
            'ektaal_files': len(ektaal_files)
        }
    }

    output_file = os.path.join(output_dir, 'bar_aware_dataset.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\n💾 Dataset saved to: {output_file}")

    # Save human-readable summary
    summary_file = os.path.join(output_dir, 'dataset_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("BAR-AWARE LSTM TRAINING DATASET SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total sequences: {len(train_sequences) + len(val_sequences)}\n")
        f.write(f"  Training: {len(train_sequences)}\n")
        f.write(f"  Validation: {len(val_sequences)}\n\n")
        f.write(f"Teental sequences: {len(all_sequences['teental'])}\n")
        f.write(f"Ektaal sequences: {len(all_sequences['ektaal'])}\n\n")
        f.write(f"Format: 2-bar context → 1-bar target\n")
        f.write(f"Features: note indices + duration bins\n")
        f.write(f"Taal-specific: 0=Teental (16 beats), 1=Ektaal (12 beats)\n\n")

        f.write("TRAINING FILES:\n")
        f.write("  Teental:\n")
        for filename, data in teental_files[:num_teental_train]:
            f.write(f"    - {data['file']}\n")
        f.write("  Ektaal:\n")
        for filename, data in ektaal_files[:num_ektaal_train]:
            f.write(f"    - {data['file']}\n")

        f.write("\nVALIDATION FILES:\n")
        f.write("  Teental:\n")
        for filename, data in teental_files[num_teental_train:]:
            f.write(f"    - {data['file']}\n")
        f.write("  Ektaal:\n")
        for filename, data in ektaal_files[num_ektaal_train:]:
            f.write(f"    - {data['file']}\n")

    print(f"📄 Summary saved to: {summary_file}")

    # Show example sequences
    print(f"\n\n{'=' * 100}")
    print(" " * 35 + "EXAMPLE SEQUENCES")
    print(f"{'=' * 100}")

    if len(train_sequences) > 0:
        example = train_sequences[0]
        print(f"\n🔍 Example training sequence:")
        print(f"   Source: {example['source_file']}")
        print(f"   Taal: {dataset['metadata']['taal_mapping'][example['taal_id']]} (meter={example['meter']})")
        print(f"   Bar range: {example['bar_range']}")
        print(f"   Context length: {len(example['context_notes'])} notes")
        print(f"   Target length: {len(example['target_notes'])} notes")

        context_notes_str = ' '.join([note_labels[idx] for idx in example['context_notes'][:10]])
        target_notes_str = ' '.join([note_labels[idx] for idx in example['target_notes'][:10]])

        print(f"   Context notes (first 10): {context_notes_str}...")
        print(f"   Target notes (first 10):  {target_notes_str}...")

    print(f"\n\n{'=' * 100}")
    print("✅ Phase 3 Complete! Ready for Phase 4 (Meter-Conditional LSTM Implementation)")
    print(f"{'=' * 100}")

    return dataset

if __name__ == "__main__":
    SEGMENTED_DIR = "segmented_bars"
    OUTPUT_DIR = "training_data"

    print("\n📌 Configuration:")
    print(f"   Input directory: {SEGMENTED_DIR}")
    print(f"   Output directory: {OUTPUT_DIR}")
    print(f"   Train/val split: 80/20")

    # Prepare dataset
    dataset = prepare_training_data(SEGMENTED_DIR, OUTPUT_DIR, train_split=0.8)
