"""
Interactive Classification Review Tool
Helps manually review and correct classifications
"""

import json
import random

def review_random_samples(json_file, num_samples=10):
    """
    Display random samples from classified data for spot-checking

    Args:
        json_file: Path to classification JSON
        num_samples: Number of random samples to review
    """

    print("=" * 70)
    print("CLASSIFICATION REVIEW - RANDOM SAMPLES")
    print("=" * 70)

    # Load classifications
    with open(json_file, 'r') as f:
        data = json.load(f)

    successful = [r for r in data["classifications"] if r["status"] == "success"]

    if len(successful) == 0:
        print("❌ No successful classifications found!")
        return

    # Select random samples
    num_samples = min(num_samples, len(successful))
    samples = random.sample(successful, num_samples)

    print(f"\n📋 Reviewing {num_samples} random samples out of {len(successful)} total files\n")

    for idx, sample in enumerate(samples, 1):
        print("-" * 70)
        print(f"SAMPLE {idx}/{num_samples}")
        print("-" * 70)
        print(f"File: {sample['relative_path']}")
        print(f"Notes detected: {sample['num_notes']}")
        print(f"Duration: {sample['total_duration']:.2f}s")
        print(f"\nSequence:")

        # Display notes in chunks of 10 for readability
        notes = sample['notes']
        durations = sample['durations']

        for i in range(0, len(notes), 10):
            chunk_notes = notes[i:i+10]
            chunk_durs = durations[i:i+10]

            print(f"  {i+1:3d}-{min(i+10, len(notes)):3d}: ", end="")
            for note, dur in zip(chunk_notes, chunk_durs):
                print(f"{note:4s}({dur:.2f}s) ", end="")
            print()

        print()

    print("=" * 70)
    print("REVIEW COMPLETE")
    print("=" * 70)
    print(f"\n💡 To edit classifications, modify: {json_file}")
    print("   Each entry has 'notes' (list) and 'durations' (list) fields")
    print("   You can add, remove, or change any notes/durations\n")

def show_statistics(json_file):
    """Show detailed statistics about classifications"""

    with open(json_file, 'r') as f:
        data = json.load(f)

    print("\n" + "=" * 70)
    print("DETAILED STATISTICS")
    print("=" * 70)

    successful = [r for r in data["classifications"] if r["status"] == "success"]

    # Group by source (TablaLive vs TablaPU)
    tablalive = [r for r in successful if "TablaLive" in r["file_path"]]
    tablapu = [r for r in successful if "TablaPU" in r["file_path"]]

    print(f"\n📊 TablaLive files: {len(tablalive)}")
    if tablalive:
        avg_notes_live = sum(r["num_notes"] for r in tablalive) / len(tablalive)
        print(f"   Average notes per file: {avg_notes_live:.1f}")

    print(f"\n📊 TablaPU files: {len(tablapu)}")
    if tablapu:
        avg_notes_pu = sum(r["num_notes"] for r in tablapu) / len(tablapu)
        print(f"   Average notes per file: {avg_notes_pu:.1f}")

    # Show files with very few or very many notes (potential issues)
    print("\n⚠️  Files with unusually few notes (< 5):")
    few_notes = [r for r in successful if r["num_notes"] < 5]
    for r in few_notes[:5]:
        print(f"   - {r['relative_path']}: {r['num_notes']} notes")

    print("\n📈 Files with many notes (> 50):")
    many_notes = [r for r in successful if r["num_notes"] > 50]
    for r in many_notes[:5]:
        print(f"   - {r['relative_path']}: {r['num_notes']} notes")

    print()

def export_for_training(json_file, output_file, min_notes=3, max_notes=None):
    """
    Export cleaned data ready for LSTM training

    Args:
        json_file: Input classification JSON
        output_file: Output file for training data
        min_notes: Minimum notes required (filter out too short sequences)
        max_notes: Maximum notes (optional, filter out too long sequences)
    """

    with open(json_file, 'r') as f:
        data = json.load(f)

    successful = [r for r in data["classifications"] if r["status"] == "success"]

    # Filter by note count
    filtered = [r for r in successful if r["num_notes"] >= min_notes]
    if max_notes:
        filtered = [r for r in filtered if r["num_notes"] <= max_notes]

    print(f"\n💾 Exporting {len(filtered)} sequences for training")
    print(f"   (filtered from {len(successful)} total sequences)")

    # Create training data format
    training_data = {
        "sequences": []
    }

    for r in filtered:
        training_data["sequences"].append({
            "notes": r["notes"],
            "durations": r["durations"],
            "source_file": r["relative_path"]
        })

    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"✅ Training data saved to: {output_file}\n")

if __name__ == "__main__":
    JSON_FILE = "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/training_data_classified.json"

    print("\nCLASSIFICATION REVIEW TOOL")
    print("=" * 70)
    print("\nOptions:")
    print("1. Review random samples (spot check)")
    print("2. Show detailed statistics")
    print("3. Export for training")
    print("4. All of the above")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1" or choice == "4":
        num = input("How many samples to review? (default 10): ").strip()
        num_samples = int(num) if num else 10
        review_random_samples(JSON_FILE, num_samples)

    if choice == "2" or choice == "4":
        show_statistics(JSON_FILE)

    if choice == "3" or choice == "4":
        output = "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/training_data_clean.json"
        export_for_training(JSON_FILE, output, min_notes=5)