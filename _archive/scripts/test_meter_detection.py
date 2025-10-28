"""
Test meter detection on multiple tabla files
"""

import sys
import os

# Add meter detection path
sys.path.insert(0, "/Users/shreyagupta/Desktop/Research MFP/B264026_Masters_Final_Project_SG/Submission/Code/HybridApproach")
from hybrid_meter_pipeline import hybrid_meter

def test_meter_on_files(files_to_test):
    """Test meter detection on list of files"""

    results = []

    for i, filepath in enumerate(files_to_test):
        print(f"\n{'=' * 100}")
        print(f"FILE {i+1}/{len(files_to_test)}: {os.path.basename(filepath)}")
        print(f"{'=' * 100}")

        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            results.append({
                'file': filepath,
                'status': 'not_found'
            })
            continue

        try:
            result = hybrid_meter(filepath)

            meter = result.get("final_meter")
            tempo = result.get("tempo")
            bar_samples = result.get("bar_start_samples", [])
            num_bars = len(bar_samples) - 1 if len(bar_samples) > 1 else len(bar_samples)
            rule_based = not result.get("rule_based_skipped", False)

            # Get swing-adjusted beats
            swing_result = result.get("swing_result", {})
            adjusted_beats = swing_result.get("adjusted_beats", [])

            print(f"\n✅ Detection successful!")
            print(f"   Method: {'🎯 Rule-based' if rule_based else '🤖 ResNet'}")
            print(f"   Meter: {meter} beats per bar")
            print(f"   Tempo: {tempo:.2f} BPM")
            print(f"   Bars detected: {num_bars}")
            print(f"   Total beats: {len(adjusted_beats)}")

            # Identify taal type
            taal_name = "Unknown"
            if meter == 16:
                taal_name = "Teental"
            elif meter == 7:
                taal_name = "Rupak"
            elif meter == 10:
                taal_name = "Jhaptaal"
            elif meter == 12:
                taal_name = "Ektaal"
            elif meter == 4:
                taal_name = "Possibly Dadra/Kaherva"
            elif meter == 6:
                taal_name = "Possibly Dadra"

            print(f"   Taal: {taal_name}")

            results.append({
                'file': filepath,
                'status': 'success',
                'meter': meter,
                'taal': taal_name,
                'tempo': tempo,
                'num_bars': num_bars,
                'num_beats': len(adjusted_beats),
                'method': 'rule_based' if rule_based else 'resnet'
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'file': filepath,
                'status': 'error',
                'error': str(e)
            })

    return results

# Test files
print("=" * 100)
print(" " * 30 + "METER DETECTION TEST SUITE")
print("=" * 100)

test_files = [
    # Long files from Tabla Player
    "/Users/shreyagupta/Desktop/AI_Research_Data/From Tabla Player/Taal Variations – 12-01-2025/Bounces/Teental Dugun Variation#02.wav",

    # Sample from training set
    "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/input/Teentaal-063.wav",
    "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/input/Rupak_06.wav",
    "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/input/Jhaptaal_01.wav",
]

print(f"\n📋 Testing {len(test_files)} files...")

results = test_meter_on_files(test_files)

# Summary
print(f"\n\n{'=' * 100}")
print("📊 SUMMARY")
print(f"{'=' * 100}")

success_count = sum(1 for r in results if r['status'] == 'success')
error_count = sum(1 for r in results if r['status'] == 'error')
not_found_count = sum(1 for r in results if r['status'] == 'not_found')

print(f"\n✅ Successful: {success_count}/{len(test_files)}")
print(f"❌ Errors: {error_count}/{len(test_files)}")
print(f"🔍 Not found: {not_found_count}/{len(test_files)}")

if success_count > 0:
    print(f"\n📊 Detected Taals:")
    for r in results:
        if r['status'] == 'success':
            filename = os.path.basename(r['file'])
            print(f"   {filename:40} → {r['taal']:20} ({r['meter']} beats, {r['num_bars']} bars, {r['method']})")

print(f"\n{'=' * 100}")
print("💡 INSIGHTS FOR BAR-AWARE LSTM:")
print(f"{'=' * 100}")

if success_count > 0:
    meters = [r['meter'] for r in results if r['status'] == 'success']
    bar_counts = [r['num_bars'] for r in results if r['status'] == 'success']

    print(f"\n• Meter range: {min(meters)} - {max(meters)} beats per bar")
    print(f"• Short files have: ~{min([b for b in bar_counts if b < 10])} bars (training set)")
    print(f"• Long files have: ~{max(bar_counts)} bars (test files)")
    print(f"\n🎯 Recommendation:")
    print(f"   - Use 2 complete bars as LSTM seed (16-32 notes for Teental)")
    print(f"   - Increase LSTM context window from 8 → {max(meters)*2}")
    print(f"   - Re-segment training data by detected bar boundaries")
    print(f"   - Train on longer files to capture complete cycles")

print(f"\n{'=' * 100}")
