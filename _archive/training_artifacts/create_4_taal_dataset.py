"""
Create training dataset with 4 taals:
- Teental (16 beats) → taal_id = 0
- Ektaal (12 beats) → taal_id = 1
- Jhaptaal (10 beats) → taal_id = 2
- Rupak (7 beats) → taal_id = 3
"""

import os
import sys
import torch
import pickle
import numpy as np

# Add paths
sys.path.insert(0, "/Users/shreyagupta/Desktop/Research MFP/B264026_Masters_Final_Project_SG/Submission/Code/HybridApproach")

from batch_classify_long_files import classify_tabla_file, ConvNet
from hybrid_meter_pipeline import hybrid_meter
from segment_by_bars import segment_notes_by_bars

# Note labels
note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]
note_to_idx = {note: idx for idx, note in enumerate(note_labels)}

# Duration bins
DURATION_BINS = [0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]

def quantize_duration(duration):
    """Quantize duration into bins"""
    for i, threshold in enumerate(DURATION_BINS[1:]):
        if duration < threshold:
            return i
    return len(DURATION_BINS) - 2


def meter_to_taal_id(meter):
    """Map detected meter to taal_id"""
    if meter == 16:
        return 0  # Teental
    elif meter == 12:
        return 1  # Ektaal
    elif meter == 10:
        return 2  # Jhaptaal
    elif meter == 7:
        return 3  # Rupak
    else:
        return None  # Unsupported


def process_file(audio_path, cnn_model, filename):
    """
    Process one audio file:
    1. Classify notes
    2. Detect meter and segment by bars
    3. Create 2-bar context → 1-bar target sequences

    Returns list of sequences
    """
    print(f"\n{'='*100}")
    print(f"Processing: {filename}")
    print(f"{'='*100}")

    # Step 1: Classify notes
    print("  [1/3] Classifying notes...")
    notes, durations, onset_samples = classify_tabla_file(
        audio_path,
        cnn_model,
        target_length=72000
    )

    print(f"        Found {len(notes)} notes, duration {sum(durations):.1f}s")

    # Step 2: Detect meter and segment by bars
    print("  [2/3] Detecting meter and segmenting bars...")
    meter_result = hybrid_meter(audio_path)
    meter = meter_result.get('final_meter')
    bar_start_samples = meter_result.get('bar_start_samples', [])

    print(f"        Detected meter: {meter} beats")
    print(f"        Found {len(bar_start_samples)-1} bars")

    # Map to taal_id
    taal_id = meter_to_taal_id(meter)
    if taal_id is None:
        print(f"        ⚠️  Unsupported meter {meter}, skipping file")
        return []

    # Segment into bars
    bars = segment_notes_by_bars(notes, durations, onset_samples, bar_start_samples)

    if len(bars) < 3:
        print(f"        ⚠️  Too few bars ({len(bars)}), skipping file")
        return []

    # Step 3: Create training sequences (2 bars context → 1 bar target)
    print("  [3/3] Creating training sequences...")
    sequences = []

    for i in range(len(bars) - 2):
        # Context: 2 consecutive bars
        context_bar_1 = bars[i]
        context_bar_2 = bars[i + 1]

        context_notes = context_bar_1['notes'] + context_bar_2['notes']
        context_durations = context_bar_1['durations'] + context_bar_2['durations']

        # Target: next bar
        target_bar = bars[i + 2]
        target_notes = target_bar['notes']
        target_durations = target_bar['durations']

        # Convert to indices
        context_note_indices = [note_to_idx[n] for n in context_notes]
        context_duration_bins = [quantize_duration(d) for d in context_durations]

        target_note_indices = [note_to_idx[n] for n in target_notes]
        target_duration_bins = [quantize_duration(d) for d in target_durations]

        sequences.append({
            'context_notes': context_note_indices,
            'context_durations': context_duration_bins,
            'target_notes': target_note_indices,
            'target_durations': target_duration_bins,
            'taal_id': taal_id,
            'filename': filename,
            'bar_indices': (i, i+1, i+2)
        })

    print(f"        Created {len(sequences)} sequences")

    return sequences


def create_4_taal_dataset():
    """Main function to create dataset with 4 taals"""

    print("="*100)
    print(" "*30 + "CREATING 4-TAAL DATASET")
    print("="*100)

    # File configuration
    INPUT_DIR = "training_data/wav_files"
    OUTPUT_PATH = "training_data/bar_aware_dataset_4_taals.pkl"
    CNN_MODEL_PATH = "ConvNet_SNFPR_model.pth"

    # Load CNN model
    print(f"\n📦 Loading CNN model: {CNN_MODEL_PATH}")
    cnn_model = ConvNet(input_channels=13, num_classes=12)
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH))
    cnn_model.eval()

    # Get all .wav files
    wav_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')])

    print(f"\n📁 Found {len(wav_files)} audio files:")
    for f in wav_files:
        print(f"   - {f}")

    # Define train/validation split
    # Put one file from each taal in validation
    validation_files = [
        'Teental Variation 3#01.wav',
        'Teental Variation 3#02.wav',
        'Ektaal Variation 2#03.wav',
        'Jhap Dugun#01.wav',
        'Rupak#02.wav'
    ]

    train_files = [f for f in wav_files if f not in validation_files]

    print(f"\n📊 Split:")
    print(f"   Training: {len(train_files)} files")
    print(f"   Validation: {len(validation_files)} files")

    # Process all files
    train_sequences = []
    val_sequences = []

    taal_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Count sequences per taal
    taal_names = {0: 'Teental', 1: 'Ektaal', 2: 'Jhaptaal', 3: 'Rupak'}

    # Process training files
    print(f"\n{'='*100}")
    print("PROCESSING TRAINING FILES")
    print(f"{'='*100}")

    for i, filename in enumerate(train_files, 1):
        filepath = os.path.join(INPUT_DIR, filename)

        print(f"\n[{i}/{len(train_files)}] {filename}")

        try:
            sequences = process_file(filepath, cnn_model, filename)
            train_sequences.extend(sequences)

            # Count by taal
            for seq in sequences:
                taal_counts[seq['taal_id']] += 1

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Process validation files
    print(f"\n{'='*100}")
    print("PROCESSING VALIDATION FILES")
    print(f"{'='*100}")

    for i, filename in enumerate(validation_files, 1):
        filepath = os.path.join(INPUT_DIR, filename)

        if not os.path.exists(filepath):
            print(f"\n[{i}/{len(validation_files)}] {filename} - NOT FOUND, skipping")
            continue

        print(f"\n[{i}/{len(validation_files)}] {filename}")

        try:
            sequences = process_file(filepath, cnn_model, filename)
            val_sequences.extend(sequences)

            # Count by taal
            for seq in sequences:
                taal_counts[seq['taal_id']] += 1

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Create dataset dictionary
    dataset_dict = {
        'train': train_sequences,
        'val': val_sequences,
        'metadata': {
            'num_classes': 12,
            'num_duration_bins': 5,
            'note_labels': note_labels,
            'taal_mapping': {
                0: 'Teental (16 beats)',
                1: 'Ektaal (12 beats)',
                2: 'Jhaptaal (10 beats)',
                3: 'Rupak (7 beats)'
            },
            'num_taals': 4,
            'train_files': train_files,
            'val_files': validation_files
        }
    }

    # Save dataset
    print(f"\n{'='*100}")
    print("SAVING DATASET")
    print(f"{'='*100}")

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(dataset_dict, f)

    print(f"\n✅ Saved to: {OUTPUT_PATH}")

    # Print summary
    print(f"\n{'='*100}")
    print("DATASET SUMMARY")
    print(f"{'='*100}")

    print(f"\nTotal sequences: {len(train_sequences) + len(val_sequences)}")
    print(f"  Training: {len(train_sequences)}")
    print(f"  Validation: {len(val_sequences)}")

    print(f"\nSequences by taal:")
    for taal_id, count in sorted(taal_counts.items()):
        taal_name = taal_names[taal_id]
        print(f"  {taal_name:15s} (id={taal_id}): {count:4d} sequences")

    print(f"\nFormat: 2-bar context → 1-bar target")
    print(f"Features: note indices + duration bins")
    print(f"Taal-conditional: 4 taals supported")

    # Save summary to text file
    summary_path = OUTPUT_PATH.replace('.pkl', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("4-TAAL LSTM TRAINING DATASET SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total sequences: {len(train_sequences) + len(val_sequences)}\n")
        f.write(f"  Training: {len(train_sequences)}\n")
        f.write(f"  Validation: {len(val_sequences)}\n\n")

        f.write("Sequences by taal:\n")
        for taal_id, count in sorted(taal_counts.items()):
            taal_name = taal_names[taal_id]
            f.write(f"  {taal_name} (id={taal_id}): {count} sequences\n")

        f.write("\nFormat: 2-bar context → 1-bar target\n")
        f.write("Features: note indices + duration bins\n")
        f.write("Taal mapping: 0=Teental(16), 1=Ektaal(12), 2=Jhaptaal(10), 3=Rupak(7)\n\n")

        f.write("TRAINING FILES:\n")
        for f_name in train_files:
            f.write(f"  - {f_name}\n")

        f.write("\nVALIDATION FILES:\n")
        for f_name in validation_files:
            f.write(f"  - {f_name}\n")

    print(f"\n📄 Summary saved to: {summary_path}")

    print(f"\n{'='*100}")
    print("✅ DATASET CREATION COMPLETE!")
    print(f"{'='*100}")

    return dataset_dict


if __name__ == "__main__":
    create_4_taal_dataset()
