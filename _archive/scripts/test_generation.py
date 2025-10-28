"""
Test LSTM generation with different temperatures
"""

import torch
import pickle
from lstm_model import create_model

def load_model(model_path, data_path):
    """Load trained model and vocabulary"""
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    vocab_size = checkpoint['vocab_size']

    # Create model
    model = create_model(vocab_size=vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load vocabulary
    idx_to_note = checkpoint['idx_to_note']
    note_to_idx = checkpoint['note_to_idx']

    # Convert idx_to_note keys from string to int if needed
    idx_to_note = {int(k): v for k, v in idx_to_note.items()}

    return model, note_to_idx, idx_to_note

def duration_bins_to_seconds(duration_bins):
    """Convert duration bin indices to approximate seconds"""
    # Bins: [0.0, 0.3], [0.3, 0.5], [0.5, 0.8], [0.8, 1.5], [1.5, inf]
    bin_midpoints = [0.15, 0.4, 0.65, 1.15, 2.0]
    return [bin_midpoints[idx] for idx in duration_bins]

def test_generation(model, seed_notes, seed_durations, idx_to_note,
                   temperatures=[0.5, 0.8, 1.0, 1.2, 1.5],
                   num_generate=16):
    """
    Test generation with different temperatures

    Args:
        model: Trained LSTM model
        seed_notes: List of seed note indices
        seed_durations: List of seed duration indices
        idx_to_note: Mapping from index to note name
        temperatures: List of temperatures to test
        num_generate: Number of notes to generate
    """
    print("="*80)
    print("TABLA LSTM GENERATION TEST")
    print("="*80)

    # Convert seed to note names
    seed_note_names = [idx_to_note[idx] for idx in seed_notes]
    print(f"\n🎵 Seed sequence: {' '.join(seed_note_names)}")
    print(f"📊 Generating {num_generate} notes with different temperatures")
    print("-"*80)

    for temp in temperatures:
        print(f"\n🌡️  Temperature: {temp:.1f}")

        # Generate
        gen_note_indices, gen_dur_indices = model.generate(
            seed_notes=seed_notes,
            seed_durations=seed_durations,
            num_generate=num_generate,
            temperature=temp,
            device='cpu'
        )

        # Convert to note names
        gen_note_names = [idx_to_note[idx] for idx in gen_note_indices]
        gen_durations = duration_bins_to_seconds(gen_dur_indices)

        # Print generated sequence
        print(f"   Notes:     {' '.join(gen_note_names)}")
        print(f"   Durations: {' '.join([f'{d:.2f}' for d in gen_durations])}")

        # Calculate total duration
        total_dur = sum(gen_durations)
        print(f"   Total duration: {total_dur:.2f}s")

    print("="*80)

if __name__ == "__main__":
    MODEL_PATH = "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/tabla_lstm_model.pth"
    DATA_PATH = "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/lstm_training_data.pkl"

    # Load model
    print("Loading model...")
    model, note_to_idx, idx_to_note = load_model(MODEL_PATH, DATA_PATH)
    print(f"✅ Model loaded with vocabulary size: {len(note_to_idx)}")

    # Test with "Ti Re Ki T" seed
    print("\n" + "="*80)
    print("TEST 1: 'Ti Re Ki T' seed (classic tabla phrase)")
    print("="*80)

    seed_notes_1 = [note_to_idx[note] for note in ['Ti', 'Re', 'Ki', 'T']]
    seed_durations_1 = [1, 1, 1, 1]  # Medium duration bin

    test_generation(model, seed_notes_1, seed_durations_1, idx_to_note,
                   temperatures=[0.5, 0.8, 1.0, 1.2, 1.5], num_generate=16)

    # Test with different seed
    print("\n" + "="*80)
    print("TEST 2: 'Dha Ti Dha Ghe' seed (bass-heavy phrase)")
    print("="*80)

    seed_notes_2 = [note_to_idx[note] for note in ['Dha', 'Ti', 'Dha', 'Ghe']]
    seed_durations_2 = [2, 1, 2, 2]  # Mix of durations

    test_generation(model, seed_notes_2, seed_durations_2, idx_to_note,
                   temperatures=[0.5, 0.8, 1.0, 1.2, 1.5], num_generate=16)

    # Test with open notes seed
    print("\n" + "="*80)
    print("TEST 3: 'Ta Ka Dhin Na' seed (mixed phrase)")
    print("="*80)

    seed_notes_3 = [note_to_idx[note] for note in ['Ta', 'Kat', 'Dhin', 'Na']]
    seed_durations_3 = [1, 1, 2, 2]

    test_generation(model, seed_notes_3, seed_durations_3, idx_to_note,
                   temperatures=[0.5, 0.8, 1.0, 1.2, 1.5], num_generate=16)

    print("\n🎉 Generation testing complete!")
    print("📝 Next step: Evaluate quality and choose optimal temperature")
