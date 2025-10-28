"""
Post-processing script for tabla classifications
Applies musical grammar rules to improve accuracy
"""

import json
import copy

# Define note categories
CRISP_NOTES = ['Ti', 'Re', 'Ki', 'T', 'Kat']  # Closed, sharp sounds
OPEN_NOTES = ['Ta', 'Na', 'Tun', 'Tin']       # Open, ringing sounds
BASS_NOTES = ['Dha', 'Dhin', 'Ghe']           # Bass/resonant sounds

def detect_crisp_sequence(notes, start_idx, length=4):
    """Check if 'length' consecutive notes are all crisp category"""
    if start_idx + length > len(notes):
        return False
    sequence = notes[start_idx:start_idx+length]
    return all(note in CRISP_NOTES for note in sequence)

def check_relative_duration_similarity(durations, start_idx, length=4, tolerance=0.5):
    """
    Check if durations are similar relative to each other (tempo-independent)

    Args:
        durations: List of durations
        start_idx: Starting index
        length: Number of durations to check
        tolerance: Allowed variation (0.5 = 50% variation from average)
    """
    if start_idx + length > len(durations):
        return False

    dur_group = durations[start_idx:start_idx+length]
    avg_dur = sum(dur_group) / length

    # Check if all durations are within tolerance of average
    return all(
        (1 - tolerance) * avg_dur <= d <= (1 + tolerance) * avg_dur
        for d in dur_group
    )

def calculate_pattern_match_score(notes, start_idx, target_pattern=['Ti', 'Re', 'Ki', 'T']):
    """
    Calculate how well the current sequence matches the target pattern
    Returns score between 0.0 and 1.0
    """
    if start_idx + len(target_pattern) > len(notes):
        return 0.0

    current_sequence = notes[start_idx:start_idx+len(target_pattern)]
    matches = sum(1 for curr, target in zip(current_sequence, target_pattern) if curr == target)
    return matches / len(target_pattern)

def apply_ti_re_ki_t_correction(notes, durations, confidence_threshold=0.75, duration_tolerance=0.5):
    """
    Apply probabilistic "Ti Re Ki T" pattern correction

    Args:
        notes: List of note names
        durations: List of note durations
        confidence_threshold: Minimum pattern match score to apply correction (0.0-1.0)
        duration_tolerance: Allowed duration variation (0.5 = 50%)

    Returns:
        corrected_notes: List of corrected notes
        corrections_made: List of correction details
    """
    corrected_notes = copy.deepcopy(notes)
    corrections_made = []

    target_pattern = ['Ti', 'Re', 'Ki', 'T']

    for i in range(len(notes) - 3):
        # Check if 4 consecutive notes are all crisp category
        if not detect_crisp_sequence(notes, i, length=4):
            continue

        # Check if durations are similar (tempo-independent)
        if not check_relative_duration_similarity(durations, i, length=4, tolerance=duration_tolerance):
            continue

        # Calculate pattern match score
        match_score = calculate_pattern_match_score(notes, i, target_pattern)

        # Apply correction if confidence is high enough
        if match_score >= confidence_threshold:
            original_sequence = notes[i:i+4]
            corrected_notes[i:i+4] = target_pattern

            corrections_made.append({
                'position': i,
                'original': original_sequence,
                'corrected': target_pattern,
                'confidence': match_score,
                'reason': f'Ti Re Ki T pattern detected with {match_score*100:.0f}% confidence'
            })

    return corrected_notes, corrections_made

def post_process_classification_file(input_json, output_json, confidence_threshold=0.75):
    """
    Post-process entire classification JSON file

    Args:
        input_json: Path to original classification file
        output_json: Path to save corrected classifications
        confidence_threshold: Minimum confidence to apply corrections
    """
    # Load original classifications
    with open(input_json, 'r') as f:
        data = json.load(f)

    print("="*80)
    print("POST-PROCESSING TABLA CLASSIFICATIONS")
    print("="*80)
    print(f"\nInput file: {input_json}")
    print(f"Confidence threshold: {confidence_threshold*100:.0f}%")
    print(f"Pattern: Ti Re Ki T (crisp note sequence)")
    print("-"*80)

    total_corrections = 0
    successful_classifications = [c for c in data['classifications'] if c['status'] == 'success']

    # Process each classification
    for idx, classification in enumerate(successful_classifications, 1):
        notes = classification['notes']
        durations = classification['durations']

        # Apply corrections
        corrected_notes, corrections = apply_ti_re_ki_t_correction(
            notes, durations, confidence_threshold=confidence_threshold
        )

        if corrections:
            print(f"\n[{idx}/{len(successful_classifications)}] {classification['relative_path']}")
            print(f"  Found {len(corrections)} correction(s):")

            for corr in corrections:
                pos = corr['position']
                print(f"    Position {pos+1}-{pos+4}: {' '.join(corr['original'])} → {' '.join(corr['corrected'])} ({corr['confidence']*100:.0f}%)")

            # Update classification
            classification['notes'] = corrected_notes
            classification['notes_original'] = notes  # Keep original for reference
            classification['corrections_applied'] = corrections
            total_corrections += len(corrections)

    # Save corrected data
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)

    print("\n" + "="*80)
    print("POST-PROCESSING COMPLETE")
    print("="*80)
    print(f"✅ Total corrections applied: {total_corrections}")
    print(f"📁 Corrected data saved to: {output_json}")
    print("="*80)

    return data

if __name__ == "__main__":
    INPUT_FILE = "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/training_data_classified.json"
    OUTPUT_FILE = "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/training_data_corrected.json"

    # Apply post-processing with 75% confidence threshold
    corrected_data = post_process_classification_file(
        input_json=INPUT_FILE,
        output_json=OUTPUT_FILE,
        confidence_threshold=0.75
    )

    print(f"\n💡 Next step: Test accuracy improvement on your review samples")
