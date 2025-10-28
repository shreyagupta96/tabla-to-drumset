"""
Quick test of new LSTM model with embedding regularization
"""

import torch
from lstm_model import create_model

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    vocab_size = checkpoint['vocab_size']

    model = create_model(vocab_size=vocab_size)
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
    # Take last 8 notes as seed
    seed_window = min(8, len(seed_notes))
    seed_notes_subset = seed_notes[-seed_window:]
    seed_durations_subset = seed_durations[-seed_window:]

    # Convert to indices
    seed_note_indices = [note_to_idx[note] for note in seed_notes_subset]
    seed_duration_bins = [quantize_duration(d) for d in seed_durations_subset]

    # Generate
    gen_note_indices, gen_dur_indices = model.generate(
        seed_notes=seed_note_indices,
        seed_durations=seed_duration_bins,
        num_generate=num_generate,
        temperature=temperature,
        device='cpu'
    )

    # Convert back
    gen_note_names = [idx_to_note[idx] for idx in gen_note_indices]
    gen_durations = duration_bins_to_seconds(gen_dur_indices)

    return gen_note_names, gen_durations

def main():
    print("="*80)
    print("TESTING NEW MODEL WITH EMBEDDING REGULARIZATION")
    print("="*80)

    # Load model
    model_path = "tabla_lstm_model.pth"
    print(f"\n📦 Loading model: {model_path}")
    model, note_to_idx, idx_to_note = load_model(model_path)
    print("✅ Model loaded successfully")

    # Test examples from real tabla classifications
    test_cases = [
        {
            "name": "Rupak (7 beats)",
            "notes": ["Tin", "Tin", "Ta", "Dhin", "Na", "Dhin", "Na"],
            "durations": [0.73, 0.77, 0.80, 0.71, 0.75, 0.80, 0.76]
        },
        {
            "name": "Jhaptaal (10 beats)",
            "notes": ["Dhin", "Na", "Dhin", "Dhin", "Ta", "Tin", "Ta", "Dhin", "Dhin", "Na"],
            "durations": [0.73, 0.64, 0.72, 0.75, 0.70, 0.72, 0.75, 0.68, 0.72, 0.75]
        },
        {
            "name": "Teentaal phrase",
            "notes": ["Dha", "Dhin", "Dhin", "Dha", "Dha", "Dhin", "Dhin", "Dha"],
            "durations": [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
        }
    ]

    temperatures = [0.8, 1.0, 1.2]

    for test in test_cases:
        print(f"\n{'='*80}")
        print(f"🎵 {test['name']}")
        print(f"{'='*80}")
        print(f"\nINPUT: {' '.join(test['notes'])}")

        for temp in temperatures:
            gen_notes, gen_durs = generate_response(
                model, test['notes'], test['durations'],
                note_to_idx, idx_to_note,
                num_generate=len(test['notes']),
                temperature=temp
            )

            print(f"\n🌡️  Temperature {temp}:")
            print(f"   OUTPUT: {' '.join(gen_notes)}")

            # Check for Na/Ta variations
            if 'Na' in gen_notes and 'Ta' in gen_notes:
                print(f"   ✨ Uses both Na and Ta (embedding similarity working!)")
            if 'Tin' in gen_notes and 'Tun' in gen_notes:
                print(f"   ✨ Uses both Tin and Tun (embedding similarity working!)")

    print(f"\n{'='*80}")
    print("✅ Testing complete!")
    print("="*80)

if __name__ == "__main__":
    main()
