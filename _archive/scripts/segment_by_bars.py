"""
Phase 2: Bar Segmentation Pipeline
Segment classified notes into complete bars using meter detection
"""

import json
import os
import sys
import numpy as np

# Add meter detection path
sys.path.insert(0, "/Users/shreyagupta/Desktop/Research MFP/B264026_Masters_Final_Project_SG/Submission/Code/HybridApproach")
from hybrid_meter_pipeline import hybrid_meter

def segment_notes_by_bars(notes, durations, onset_samples, bar_start_samples):
    """
    Simplified bar segmentation: Any note with onset between
    bar_start[i] and bar_start[i+1] belongs to Bar i

    No beat-level alignment needed!
    """
    bars = []

    for i in range(len(bar_start_samples) - 1):
        bar_start = bar_start_samples[i]
        bar_end = bar_start_samples[i + 1]

        # Find all notes in this bar
        bar_notes = []
        bar_durs = []
        bar_onsets = []

        for j, onset in enumerate(onset_samples):
            if bar_start <= onset < bar_end:
                bar_notes.append(notes[j])
                bar_durs.append(durations[j])
                bar_onsets.append(onset)

        # Only include bars with notes
        if len(bar_notes) > 0:
            bars.append({
                'bar_num': i,
                'notes': bar_notes,
                'durations': bar_durs,
                'onset_samples': bar_onsets,
                'num_notes': len(bar_notes),
                'bar_start_sample': int(bar_start),
                'bar_end_sample': int(bar_end)
            })

    return bars

def process_file(classified_file, audio_dir, output_dir):
    """
    Process one classified file: run meter detection and segment by bars
    """
    # Load classified data
    with open(classified_file, 'r') as f:
        data = json.load(f)

    filename = data['file']
    notes = data['notes']
    durations = data['durations']
    onset_samples = data['onset_samples']

    print(f"\n📂 Processing: {filename}")
    print(f"   Classified notes: {len(notes)}")

    # Find original audio file
    audio_path = os.path.join(audio_dir, filename)

    if not os.path.exists(audio_path):
        print(f"   ❌ Audio file not found: {audio_path}")
        return None

    # Run meter detection
    print(f"   🔍 Running meter detection...")
    try:
        meter_result = hybrid_meter(audio_path)
    except Exception as e:
        print(f"   ❌ Meter detection failed: {e}")
        return None

    meter = meter_result.get('final_meter')
    tempo = meter_result.get('tempo')
    bar_start_samples = meter_result.get('bar_start_samples', [])

    # Determine taal
    if meter == 16:
        taal = 'Teental'
    elif meter == 12:
        taal = 'Ektaal'
    else:
        taal = 'Unknown'

    print(f"   ✅ Detected: {taal} (meter={meter}, tempo={tempo:.1f} BPM)")
    print(f"   📊 Bar boundaries: {len(bar_start_samples)} samples")

    # Segment by bars
    print(f"   ✂️  Segmenting notes into bars...")
    bars = segment_notes_by_bars(notes, durations, onset_samples, bar_start_samples)

    print(f"   ✅ Segmented into {len(bars)} complete bars")

    # Calculate statistics
    notes_per_bar = [b['num_notes'] for b in bars]
    avg_notes = np.mean(notes_per_bar) if notes_per_bar else 0
    min_notes = min(notes_per_bar) if notes_per_bar else 0
    max_notes = max(notes_per_bar) if notes_per_bar else 0

    print(f"   📈 Notes per bar: avg={avg_notes:.1f}, min={min_notes}, max={max_notes}")

    # Create output (ensure all numpy types converted to Python types)
    output_data = {
        'file': filename,
        'filepath': audio_path,
        'taal': taal,
        'meter': int(meter) if meter is not None else None,
        'tempo': float(tempo) if tempo is not None else None,
        'bars': bars,
        'total_bars': len(bars),
        'total_notes': len(notes),
        'bar_start_samples': [int(s) for s in bar_start_samples],
        'statistics': {
            'avg_notes_per_bar': float(avg_notes),
            'min_notes_per_bar': int(min_notes),
            'max_notes_per_bar': int(max_notes)
        }
    }

    # Save
    output_file = os.path.join(output_dir, filename.replace('.wav', '_bars.json'))
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"   💾 Saved to: {os.path.basename(output_file)}")

    return output_data

def batch_segment_files(classified_dir, audio_dir, output_dir):
    """
    Batch segment all classified files
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print(" " * 30 + "PHASE 2: BAR SEGMENTATION")
    print("=" * 100)

    # Find all classified files
    classified_files = [f for f in os.listdir(classified_dir) if f.endswith('_classified.json')]
    classified_files.sort()

    print(f"\n📋 Found {len(classified_files)} classified files")

    results = []

    for i, filename in enumerate(classified_files, 1):
        filepath = os.path.join(classified_dir, filename)

        print(f"\n{'=' * 100}")
        print(f"FILE {i}/{len(classified_files)}: {filename}")
        print(f"{'=' * 100}")

        try:
            # Check if already processed
            output_file = os.path.join(output_dir, filename.replace('_classified.json', '.wav_bars.json'))
            if os.path.exists(output_file):
                print(f"⏭️  Already segmented, skipping...")
                with open(output_file, 'r') as f:
                    data = json.load(f)
                results.append({
                    'file': data['file'],
                    'status': 'already_done',
                    'taal': data['taal'],
                    'num_bars': data['total_bars']
                })
                continue

            # Process
            result = process_file(filepath, audio_dir, output_dir)

            if result:
                results.append({
                    'file': result['file'],
                    'status': 'success',
                    'taal': result['taal'],
                    'num_bars': result['total_bars'],
                    'meter': result['meter']
                })
            else:
                results.append({
                    'file': filename,
                    'status': 'error'
                })

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'file': filename,
                'status': 'error',
                'error': str(e)
            })

    # Print summary
    print(f"\n\n{'=' * 100}")
    print(" " * 35 + "SEGMENTATION SUMMARY")
    print(f"{'=' * 100}")

    success = sum(1 for r in results if r['status'] in ['success', 'already_done'])
    errors = sum(1 for r in results if r['status'] == 'error')

    print(f"\n✅ Successfully segmented: {success}/{len(classified_files)}")
    print(f"❌ Errors: {errors}/{len(classified_files)}")

    if success > 0:
        print(f"\n📊 Segmented Files:")

        teental_bars = 0
        ektaal_bars = 0
        total_bars = 0

        for r in results:
            if r['status'] in ['success', 'already_done']:
                status_icon = '✅' if r['status'] == 'success' else '⏭️'
                print(f"   {status_icon} {r['file']:50} → {r['taal']:10} {r['num_bars']:3} bars")
                total_bars += r['num_bars']

                if r['taal'] == 'Teental':
                    teental_bars += r['num_bars']
                elif r['taal'] == 'Ektaal':
                    ektaal_bars += r['num_bars']

        print(f"\n📈 Total Statistics:")
        print(f"   Total bars: {total_bars}")
        print(f"   Teental bars: {teental_bars}")
        print(f"   Ektaal bars: {ektaal_bars}")

        # Estimate training sequences
        teental_sequences = max(0, teental_bars - 2)  # 2-bar context → 1-bar target
        ektaal_sequences = max(0, ektaal_bars - 2)
        total_sequences = teental_sequences + ektaal_sequences

        print(f"\n🎯 Estimated Training Sequences:")
        print(f"   Teental: ~{teental_sequences} sequences (from {teental_bars} bars)")
        print(f"   Ektaal: ~{ektaal_sequences} sequences (from {ektaal_bars} bars)")
        print(f"   Total: ~{total_sequences} sequences")

    print(f"\n{'=' * 100}")
    print("✅ Phase 2 Complete! Ready for Phase 3 (Training Dataset Preparation)")
    print(f"{'=' * 100}")

    return results

if __name__ == "__main__":
    # Auto-detect directories
    base = "/Users/shreyagupta/Desktop/AI_Research_Data/From Tabla Player"
    taal_dirs = [d for d in os.listdir(base) if 'Taal Variations' in d]

    AUDIO_DIR = None
    for d in taal_dirs:
        bounces_path = os.path.join(base, d, 'Bounces')
        if os.path.exists(bounces_path):
            AUDIO_DIR = bounces_path
            break

    CLASSIFIED_DIR = "classified_long_files"
    OUTPUT_DIR = "segmented_bars"

    print("\n📌 Configuration:")
    print(f"   Classified files: {CLASSIFIED_DIR}")
    print(f"   Original audio: {AUDIO_DIR}")
    print(f"   Output directory: {OUTPUT_DIR}")

    # Run segmentation
    results = batch_segment_files(CLASSIFIED_DIR, AUDIO_DIR, OUTPUT_DIR)
