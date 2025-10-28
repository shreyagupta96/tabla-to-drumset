"""
Quick test script for 4-taal meter-conditional LSTM with Rupak input
"""

import sys
import torch
import numpy as np

sys.path.insert(0, "/Users/shreyagupta/Desktop/Research MFP/B264026_Masters_Final_Project_SG/Submission/Code/HybridApproach")

from meter_conditional_lstm import create_model
from hybrid_meter_pipeline import hybrid_meter
from batch_classify_long_files import classify_tabla_file, ConvNet
from segment_by_bars import segment_notes_by_bars
from taal_utils import meter_to_taal_id, taal_id_to_name

note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]
note_to_idx = {note: idx for idx, note in enumerate(note_labels)}

DURATION_BINS = [0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]

def quantize_duration(duration):
    """Quantize duration into bins"""
    for i, threshold in enumerate(DURATION_BINS[1:]):
        if duration < threshold:
            return i
    return len(DURATION_BINS) - 2


def test_rupak_generation():
    """Test classification and generation on Rupak input"""

    # Configuration
    audio_path = "/Users/shreyagupta/Desktop/AI_Research_Data/Tabla_files/Rupak_1.wav"
    model_path = "models/best_bar_aware_lstm.pth"

    print("=" * 100)
    print(" " * 30 + "TESTING 4-TAAL MODEL WITH RUPAK")
    print("=" * 100)
    print()

    # Step 1: Classify notes
    print("📝 Step 1: Classifying tabla notes...")
    cnn_model = ConvNet(input_channels=13, num_classes=12)
    cnn_model.load_state_dict(torch.load('ConvNet_SNFPR_model.pth'))
    cnn_model.eval()

    notes, durations, onset_samples = classify_tabla_file(
        audio_path, cnn_model, target_length=72000
    )

    print(f"   ✅ Found {len(notes)} notes")
    print(f"   Notes: {' '.join(notes[:15])}...")
    print()

    # Step 2: Detect meter
    print("🎯 Step 2: Detecting meter...")
    meter_result = hybrid_meter(audio_path)
    meter = meter_result.get('final_meter')
    bar_start_samples = meter_result.get('bar_start_samples', [])

    print(f"   ✅ Detected meter: {meter} beats")
    print(f"   Bars found: {len(bar_start_samples) - 1}")
    print()

    # Step 3: Map to taal_id
    print("🔄 Step 3: Mapping to taal_id...")
    taal_id = meter_to_taal_id(meter)
    if taal_id is None:
        print(f"   ❌ ERROR: Unsupported meter {meter} beats")
        return

    taal_name = taal_id_to_name(taal_id)
    print(f"   ✅ Taal ID: {taal_id} ({taal_name})")
    print()

    # Step 4: Segment by bars
    print("📊 Step 4: Segmenting by bars...")
    bars = segment_notes_by_bars(notes, durations, onset_samples, bar_start_samples)
    print(f"   ✅ Segmented into {len(bars)} bars")

    if len(bars) < 3:
        print(f"   ❌ ERROR: Need at least 3 bars for generation")
        return

    print(f"   Bar 0: {len(bars[0]['notes'])} notes")
    print(f"   Bar 1: {len(bars[1]['notes'])} notes")
    print(f"   Bar 2: {len(bars[2]['notes'])} notes")
    print()

    # Step 5: Load 4-taal model
    print("🤖 Step 5: Loading 4-taal meter-conditional LSTM...")
    checkpoint = torch.load(model_path, map_location='cpu')
    metadata = checkpoint['metadata']

    print(f"   Model trained on {metadata['num_taals']} taals:")
    for tid, tname in metadata['taal_mapping'].items():
        print(f"     - {tid}: {tname}")

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
    print(f"   ✅ Model loaded successfully")
    print()

    # Step 6: Prepare context (last 2 bars)
    print("📥 Step 6: Preparing context (last 2 bars)...")
    seed_bars = bars[-2:]
    seed_notes = []
    seed_durations = []

    for bar in seed_bars:
        seed_notes.extend([note_to_idx[n] for n in bar['notes']])
        seed_durations.extend([quantize_duration(d) for d in bar['durations']])

    print(f"   Context length: {len(seed_notes)} notes")
    print(f"   Context notes: {' '.join([note_labels[i] for i in seed_notes[:10]])}...")
    print()

    # Step 7: Generate with correct taal_id
    print(f"🎵 Step 7: Generating with taal_id={taal_id} ({taal_name})...")
    num_generate = 20

    gen_note_indices, gen_duration_indices = model.generate(
        seed_notes=seed_notes,
        seed_durations=seed_durations,
        taal_id=taal_id,  # THIS IS THE KEY - using Rupak taal_id!
        num_generate=num_generate,
        temperature=0.8,
        device='cpu'
    )

    gen_notes = [note_labels[i] for i in gen_note_indices]

    print(f"   ✅ Generated {len(gen_notes)} notes")
    print(f"   Generated notes: {' '.join(gen_notes)}")
    print()

    print("=" * 100)
    print("✅ TEST COMPLETE!")
    print("=" * 100)
    print()
    print(f"Summary:")
    print(f"  Input: Rupak_1.wav")
    print(f"  Detected meter: {meter} beats → taal_id={taal_id} ({taal_name})")
    print(f"  Context: {len(seed_notes)} notes from 2 bars")
    print(f"  Generated: {len(gen_notes)} notes conditioned on Rupak taal")
    print()


if __name__ == "__main__":
    test_rupak_generation()
