"""
Data preprocessing for LSTM training
Converts tabla note sequences to LSTM-friendly format
"""

import json
import numpy as np
import pickle
from collections import Counter

# Note labels vocabulary (same as CNN classifier)
NOTE_LABELS = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]

def create_vocabulary(note_sequences):
    """
    Create vocabulary mapping from note sequences

    Returns:
        note_to_idx: dict mapping note names to indices
        idx_to_note: dict mapping indices to note names
    """
    # Use predefined vocabulary
    note_to_idx = {note: idx for idx, note in enumerate(NOTE_LABELS)}
    idx_to_note = {idx: note for idx, note in enumerate(NOTE_LABELS)}

    return note_to_idx, idx_to_note

def quantize_duration(duration, bins=[0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]):
    """
    Quantize continuous duration into discrete bins

    Args:
        duration: Duration in seconds
        bins: Bin edges for quantization

    Returns:
        Bin index (0-4)
    """
    for i, edge in enumerate(bins[1:]):
        if duration < edge:
            return i
    return len(bins) - 2

def prepare_sequences(json_file, sequence_length=8):
    """
    Prepare training sequences for LSTM

    Args:
        json_file: Path to corrected classifications JSON
        sequence_length: Length of input sequences (context window)

    Returns:
        X_notes: Input note sequences
        X_durations: Input duration sequences
        y_notes: Target next notes
        y_durations: Target next durations
        note_to_idx: Vocabulary mapping
        idx_to_note: Reverse vocabulary mapping
    """
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)

    successful = [c for c in data['classifications'] if c['status'] == 'success']

    # Extract all note sequences
    all_sequences = []
    for classification in successful:
        notes = classification['notes']
        durations = classification['durations']
        all_sequences.append((notes, durations))

    # Create vocabulary
    note_to_idx, idx_to_note = create_vocabulary([seq[0] for seq in all_sequences])

    print("="*80)
    print("DATA PREPROCESSING FOR LSTM")
    print("="*80)
    print(f"\n📊 Vocabulary size: {len(note_to_idx)} notes")
    print(f"📊 Notes: {', '.join(NOTE_LABELS)}")
    print(f"📊 Total sequences: {len(all_sequences)}")
    print(f"📊 Sequence length (context): {sequence_length}")

    # Create training examples using sliding window
    X_notes = []
    X_durations = []
    y_notes = []
    y_durations = []

    for notes, durations in all_sequences:
        # Convert notes to indices
        note_indices = [note_to_idx[note] for note in notes]

        # Quantize durations
        duration_bins = [quantize_duration(d) for d in durations]

        # Create sliding window sequences
        for i in range(len(notes) - sequence_length):
            # Input: sequence_length notes/durations
            X_notes.append(note_indices[i:i+sequence_length])
            X_durations.append(duration_bins[i:i+sequence_length])

            # Target: next note/duration
            y_notes.append(note_indices[i+sequence_length])
            y_durations.append(duration_bins[i+sequence_length])

    # Convert to numpy arrays
    X_notes = np.array(X_notes)
    X_durations = np.array(X_durations)
    y_notes = np.array(y_notes)
    y_durations = np.array(y_durations)

    print(f"\n📈 Training examples created: {len(X_notes)}")
    print(f"📈 Input shape: ({len(X_notes)}, {sequence_length})")
    print(f"📈 Note distribution in targets:")

    # Show distribution
    note_counts = Counter(y_notes)
    for note_idx, count in sorted(note_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        note_name = idx_to_note[note_idx]
        percentage = count / len(y_notes) * 100
        print(f"   {note_name:8s}: {count:4d} ({percentage:5.1f}%)")

    print("="*80)

    return X_notes, X_durations, y_notes, y_durations, note_to_idx, idx_to_note

def save_preprocessed_data(output_file, X_notes, X_durations, y_notes, y_durations,
                           note_to_idx, idx_to_note):
    """Save preprocessed data for training"""
    data = {
        'X_notes': X_notes,
        'X_durations': X_durations,
        'y_notes': y_notes,
        'y_durations': y_durations,
        'note_to_idx': note_to_idx,
        'idx_to_note': idx_to_note,
        'vocab_size': len(note_to_idx)
    }

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"\n💾 Preprocessed data saved to: {output_file}")

if __name__ == "__main__":
    # Prepare data
    INPUT_JSON = "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/training_data_corrected_25pct.json"
    OUTPUT_FILE = "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/lstm_training_data.pkl"

    X_notes, X_durations, y_notes, y_durations, note_to_idx, idx_to_note = prepare_sequences(
        INPUT_JSON,
        sequence_length=8
    )

    save_preprocessed_data(
        OUTPUT_FILE,
        X_notes, X_durations, y_notes, y_durations,
        note_to_idx, idx_to_note
    )

    print("\n✅ Data preprocessing complete!")
    print(f"📝 Ready for LSTM training")
