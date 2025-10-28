"""
Test Bar-Aware LSTM Generation
Demonstrates the complete pipeline on a tabla groove
"""

import torch
import pickle
import json
import sys
import os

# Add meter detection path
sys.path.insert(0, "/Users/shreyagupta/Desktop/Research MFP/B264026_Masters_Final_Project_SG/Submission/Code/HybridApproach")

from meter_conditional_lstm import create_model
from hybrid_meter_pipeline import hybrid_meter
from batch_classify_long_files import classify_tabla_file, ConvNet
from segment_by_bars import segment_notes_by_bars

note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]
note_to_idx = {note: idx for idx, note in enumerate(note_labels)}

DURATION_BINS = [0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]

def quantize_duration(duration):
    """Quantize duration into bins"""
    for i, threshold in enumerate(DURATION_BINS[1:]):
        if duration < threshold:
            return i
    return len(DURATION_BINS) - 2

def test_on_audio_file(audio_file, cnn_model_path='ConvNet_SNFPR_model.pth',
                        lstm_model_path='models/best_bar_aware_lstm.pth',
                        num_generate=32, temperature=1.0):
    """
    Complete pipeline test:
    1. Classify notes with CNN
    2. Detect meter and segment by bars
    3. Generate new sequence with bar-aware LSTM
    """

    print("=" * 100)
    print(" " * 30 + "BAR-AWARE LSTM TEST")
    print("=" * 100)

    print(f"\n📂 Test file: {os.path.basename(audio_file)}")

    # Step 1: Classify with CNN
    print(f"\n{'='*100}")
    print("STEP 1: CLASSIFY NOTES WITH CNN")
    print(f"{'='*100}")

    print(f"\n📦 Loading CNN model...")
    cnn_model = ConvNet(input_channels=13, num_classes=12)
    cnn_model.load_state_dict(torch.load(cnn_model_path))
    cnn_model.eval()
    print(f"✅ CNN model loaded")

    print(f"\n🎵 Classifying tabla strokes...")
    notes, durations, onset_samples = classify_tabla_file(
        audio_file, cnn_model, target_length=72000
    )

    print(f"\n✅ Classification complete:")
    print(f"   Total notes: {len(notes)}")
    print(f"   First 20 notes: {' '.join(notes[:20])}")

    # Step 2: Meter detection and segmentation
    print(f"\n{'='*100}")
    print("STEP 2: METER DETECTION & BAR SEGMENTATION")
    print(f"{'='*100}")

    print(f"\n🔍 Running meter detection...")
    meter_result = hybrid_meter(audio_file)

    meter = meter_result.get('final_meter')
    tempo = meter_result.get('tempo')
    bar_start_samples = meter_result.get('bar_start_samples', [])

    if meter == 16:
        taal = 'Teental'
        taal_id = 0
    elif meter == 12:
        taal = 'Ektaal'
        taal_id = 1
    else:
        taal = 'Unknown'
        taal_id = 2

    print(f"\n✅ Meter detection:")
    print(f"   Taal: {taal} (meter={meter})")
    print(f"   Tempo: {tempo:.1f} BPM")
    print(f"   Bar boundaries: {len(bar_start_samples)} samples")

    print(f"\n✂️  Segmenting notes into bars...")
    bars = segment_notes_by_bars(notes, durations, onset_samples, bar_start_samples)

    print(f"\n✅ Segmentation complete:")
    print(f"   Total bars: {len(bars)}")
    if len(bars) > 0:
        print(f"   Notes per bar: min={min(b['num_notes'] for b in bars)}, "
              f"max={max(b['num_notes'] for b in bars)}, "
              f"avg={sum(b['num_notes'] for b in bars)/len(bars):.1f}")

    # Show first 2 bars
    print(f"\n📊 First 2 bars:")
    for i in range(min(2, len(bars))):
        bar = bars[i]
        print(f"   Bar {bar['bar_num']}: {' '.join(bar['notes'])}")

    if len(bars) < 3:
        print(f"\n❌ Need at least 3 bars for generation (got {len(bars)})")
        return None

    # Step 3: Generate with bar-aware LSTM
    print(f"\n{'='*100}")
    print("STEP 3: GENERATE WITH BAR-AWARE LSTM")
    print(f"{'='*100}")

    print(f"\n📦 Loading bar-aware LSTM model...")
    checkpoint = torch.load(lstm_model_path, map_location='cpu')
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

    print(f"✅ Bar-aware LSTM loaded")
    print(f"   Trained epochs: {checkpoint['epoch']+1}")
    print(f"   Val loss: {checkpoint['val_loss']:.4f}")

    # Use last 2 bars as seed (2-bar context)
    seed_bars = bars[-2:]
    seed_notes = []
    seed_durations = []

    for bar in seed_bars:
        seed_notes.extend([note_to_idx[n] for n in bar['notes']])
        seed_durations.extend([quantize_duration(d) for d in bar['durations']])

    print(f"\n🌱 Seed (last 2 bars):")
    print(f"   Bar {seed_bars[0]['bar_num']}: {' '.join(seed_bars[0]['notes'])}")
    print(f"   Bar {seed_bars[1]['bar_num']}: {' '.join(seed_bars[1]['notes'])}")
    print(f"   Total seed length: {len(seed_notes)} notes")

    print(f"\n🎲 Generating {num_generate} notes with temperature={temperature}...")

    gen_notes, gen_durations = model.generate(
        seed_notes=seed_notes,
        seed_durations=seed_durations,
        taal_id=taal_id,
        num_generate=num_generate,
        temperature=temperature,
        device='cpu'
    )

    gen_note_labels = [note_labels[idx] for idx in gen_notes]

    print(f"\n✅ Generation complete!")
    print(f"\n🎵 Generated sequence ({len(gen_note_labels)} notes):")
    print(f"   {' '.join(gen_note_labels)}")

    # Analyze generation quality
    print(f"\n{'='*100}")
    print("GENERATION QUALITY ANALYSIS")
    print(f"{'='*100}")

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

    print(f"\n📊 Metrics:")
    print(f"   Max consecutive repeats: {max_repeat}")
    print(f"   Unique notes: {unique_notes}/{len(gen_notes)} ({unique_ratio:.1%})")
    print(f"   Note distribution:")

    from collections import Counter
    note_counts = Counter(gen_note_labels)
    for note, count in note_counts.most_common():
        print(f"      {note}: {count}")

    # Check for Na/Ta and Tin/Tun variation (embedding regularization)
    has_na = 'Na' in gen_note_labels
    has_ta = 'Ta' in gen_note_labels
    has_tin = 'Tin' in gen_note_labels
    has_tun = 'Tun' in gen_note_labels

    print(f"\n✨ Embedding regularization effects:")
    if has_na and has_ta:
        print(f"   ✅ Both Na and Ta present (good variation!)")
    elif has_na or has_ta:
        print(f"   ⚠️  Only {'Na' if has_na else 'Ta'} present")

    if has_tin and has_tun:
        print(f"   ✅ Both Tin and Tun present (good variation!)")
    elif has_tin or has_tun:
        print(f"   ⚠️  Only {'Tin' if has_tin else 'Tun'} present")

    print(f"\n{'='*100}")
    print("✅ TEST COMPLETE!")
    print(f"{'='*100}")

    return {
        'notes': notes,
        'bars': bars,
        'taal': taal,
        'meter': meter,
        'tempo': tempo,
        'generated_notes': gen_note_labels,
        'generated_durations': gen_durations,
        'max_consecutive_repeats': max_repeat,
        'unique_note_ratio': unique_ratio
    }


if __name__ == "__main__":
    # Test on one of the already segmented Ektaal files
    test_files = [
        "/Users/shreyagupta/Desktop/AI_Research_Data/From Tabla Player/Taal Variations – 12-01-2025/Bounces/Basic Ektaal#03.wav",
        "/Users/shreyagupta/Desktop/AI_Research_Data/From Tabla Player/Taal Variations – 12-01-2025/Bounces/Ektaal Dugun#01.wav",
    ]

    # Find which file exists
    audio_file = None
    for f in test_files:
        if os.path.exists(f):
            audio_file = f
            break

    if audio_file is None:
        # Fallback: use glob to find any Ektaal file
        import glob
        base = "/Users/shreyagupta/Desktop/AI_Research_Data/From Tabla Player"
        pattern = os.path.join(base, "Taal Variations*/Bounces/Ektaal*.wav")
        files = glob.glob(pattern)
        if files:
            audio_file = files[0]

    if audio_file is None:
        print("❌ No test file found!")
        print("\nPlease provide a tabla audio file path as argument:")
        print("  python test_bar_aware_lstm.py <audio_file.wav>")
        exit(1)

    # Run test
    result = test_on_audio_file(
        audio_file=audio_file,
        num_generate=32,  # Generate ~1-2 bars worth
        temperature=1.0
    )
