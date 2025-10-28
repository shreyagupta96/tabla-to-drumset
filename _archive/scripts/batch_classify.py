"""
Batch Classification Script for Tabla Training Data
Classifies all tabla files and saves results for manual review
"""

import os
import json
import torch
import librosa
import numpy as np

# Import directly - these are safe imports that won't trigger Flask
from api import ConvNet, predict_tabla_bols, Adjust_Length, note_labels

def batch_classify_tabla_files(data_dir, output_file, model_path="ConvNet_SNFPR_model.pth"):
    """
    Classify all tabla files in the dataset directory

    Args:
        data_dir: Path to directory containing tabla files
        output_file: Path to save JSON output
        model_path: Path to trained CNN model
    """

    # Load the CNN model
    print("=" * 70)
    print("BATCH TABLA CLASSIFICATION")
    print("=" * 70)
    print(f"\n📦 Loading CNN model from {model_path}...")

    model_CNN = ConvNet(input_channels=13, num_classes=12)
    model_CNN.load_state_dict(torch.load(model_path))
    model_CNN.eval()

    print("✅ Model loaded successfully!\n")

    # Find all .wav files recursively
    wav_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    wav_files.sort()
    print(f"🎵 Found {len(wav_files)} WAV files to classify\n")
    print("-" * 70)

    # Store all results
    all_results = {
        "metadata": {
            "total_files": len(wav_files),
            "model_path": model_path,
            "note_vocabulary": note_labels
        },
        "classifications": []
    }

    # Process each file
    for idx, wav_file in enumerate(wav_files, 1):
        relative_path = os.path.relpath(wav_file, data_dir)
        print(f"\n[{idx}/{len(wav_files)}] Processing: {relative_path}")

        try:
            # Classify the file
            predicted_notes, durations = predict_tabla_bols(
                file_path=wav_file,
                model=model_CNN,
                adjust_length_fn=Adjust_Length,
                target_length=72000
            )

            # Create result entry
            result = {
                "file_path": wav_file,
                "relative_path": relative_path,
                "filename": os.path.basename(wav_file),
                "num_notes": len(predicted_notes),
                "notes": predicted_notes,
                "durations": durations,
                "total_duration": sum(durations),
                "status": "success"
            }

            # Print summary
            print(f"   ✅ Detected {len(predicted_notes)} notes")
            print(f"   🎵 Sequence: {' '.join(predicted_notes[:10])}" +
                  (" ..." if len(predicted_notes) > 10 else ""))
            print(f"   ⏱️  Total duration: {sum(durations):.2f}s")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            result = {
                "file_path": wav_file,
                "relative_path": relative_path,
                "filename": os.path.basename(wav_file),
                "status": "error",
                "error_message": str(e)
            }

        all_results["classifications"].append(result)

    # Save results to JSON
    print("\n" + "=" * 70)
    print(f"💾 Saving results to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("✅ Results saved successfully!")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    successful = [r for r in all_results["classifications"] if r["status"] == "success"]
    failed = [r for r in all_results["classifications"] if r["status"] == "error"]

    print(f"\n✅ Successfully classified: {len(successful)} files")
    print(f"❌ Failed: {len(failed)} files")

    if successful:
        total_notes = sum(r["num_notes"] for r in successful)
        avg_notes = total_notes / len(successful)
        total_duration = sum(r["total_duration"] for r in successful)

        print(f"\n📊 Total notes across all files: {total_notes}")
        print(f"📊 Average notes per file: {avg_notes:.1f}")
        print(f"📊 Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")

        # Note distribution
        from collections import Counter
        all_notes = []
        for r in successful:
            all_notes.extend(r["notes"])

        note_counts = Counter(all_notes)
        print(f"\n🎵 Note Distribution (top 5):")
        for note, count in note_counts.most_common(5):
            percentage = (count / total_notes) * 100
            print(f"   {note:8s}: {count:5d} ({percentage:5.1f}%)")

    if failed:
        print(f"\n❌ Failed files:")
        for r in failed:
            print(f"   - {r['relative_path']}: {r['error_message']}")

    print("\n" + "=" * 70)
    print("🎉 Batch classification complete!")
    print(f"📝 Next step: Review and edit {output_file} for accuracy")
    print("=" * 70)

    return all_results

if __name__ == "__main__":
    # Configuration
    DATA_DIR = "/Users/shreyagupta/Desktop/Machine_Learning/Group_Project/Project - Tabla/DataSet/For_Generation"
    OUTPUT_FILE = "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/training_data_classified.json"

    # Run batch classification
    results = batch_classify_tabla_files(DATA_DIR, OUTPUT_FILE)
