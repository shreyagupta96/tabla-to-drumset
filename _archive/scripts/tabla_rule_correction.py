"""
Tabla Composition Rule Correction
Post-processing for CNN classified notes before LSTM training/inference
"""

def correct_ti_re_ki_t_pattern(notes, durations=None):
    """
    Apply "Ti Re Ki T" rule correction to classified notes

    Rule: Any 4 consecutive closed strokes from {Ti, Re, Ki, T}
          should be corrected to "Ti Re Ki T" in that order

    Args:
        notes: List of note labels (strings)
        durations: Optional list of duration values (same length as notes)

    Returns:
        corrected_notes: List of corrected note labels
        corrected_durations: List of durations (unchanged if provided)
        corrections_made: List of (start_index, original_sequence) tuples
    """

    # Timbral categories
    pattern_set = {'Ti', 'Re', 'Ki', 'T'}  # Notes that belong to the pattern
    closed_crisp = {'Ti', 'Re', 'Ki', 'T', 'Kat'}  # All closed/crisp strokes

    # Target pattern
    target_pattern = ['Ti', 'Re', 'Ki', 'T']

    # Work with a copy
    corrected_notes = notes.copy()
    corrected_durations = durations.copy() if durations is not None else None
    corrections_made = []

    i = 0
    while i < len(corrected_notes):
        # Check if current note is closed and from pattern set
        if corrected_notes[i] in pattern_set:
            # Look ahead to count consecutive closed strokes from pattern set
            consecutive_pattern_notes = [corrected_notes[i]]
            j = i + 1

            while j < len(corrected_notes) and len(consecutive_pattern_notes) < 4:
                if corrected_notes[j] in pattern_set:
                    consecutive_pattern_notes.append(corrected_notes[j])
                    j += 1
                elif corrected_notes[j] not in closed_crisp:
                    # Hit an open stroke - pattern broken
                    break
                else:
                    # Hit another closed stroke (like Kat) - pattern broken
                    break

            # If we have exactly 4 consecutive notes from pattern set, correct them
            if len(consecutive_pattern_notes) == 4:
                # Store original for logging
                original_sequence = consecutive_pattern_notes.copy()

                # Replace with correct pattern
                for k in range(4):
                    corrected_notes[i + k] = target_pattern[k]

                # Record correction
                corrections_made.append((i, original_sequence))

                # Skip past the corrected sequence
                i += 4
            else:
                i += 1
        else:
            i += 1

    return corrected_notes, corrected_durations, corrections_made


def apply_corrections_and_report(notes, durations, filename=""):
    """
    Apply corrections and print a report

    Returns:
        corrected_notes, corrected_durations, num_corrections
    """
    corrected_notes, corrected_durations, corrections = correct_ti_re_ki_t_pattern(
        notes, durations
    )

    if corrections:
        print(f"\n  📝 Applied {len(corrections)} 'Ti Re Ki T' correction(s){' in ' + filename if filename else ''}:")
        for idx, original in corrections:
            corrected = corrected_notes[idx:idx+4]
            print(f"     Position {idx}: {' '.join(original)} → {' '.join(corrected)}")

    return corrected_notes, corrected_durations, len(corrections)


if __name__ == "__main__":
    # Test the correction function
    print("="*80)
    print("TESTING TI RE KI T CORRECTION")
    print("="*80)

    # Test case 1: Perfect pattern (should not change)
    notes1 = ["Dha", "Ti", "Re", "Ki", "T", "Dhin"]
    print(f"\nTest 1 - Perfect pattern:")
    print(f"  Input:  {' '.join(notes1)}")
    corrected1, _, corr1 = correct_ti_re_ki_t_pattern(notes1)
    print(f"  Output: {' '.join(corrected1)}")
    print(f"  Corrections: {len(corr1)}")

    # Test case 2: Wrong order (should correct)
    notes2 = ["Dha", "Re", "Ki", "T", "Ti", "Dhin"]
    print(f"\nTest 2 - Wrong order:")
    print(f"  Input:  {' '.join(notes2)}")
    corrected2, _, corr2 = correct_ti_re_ki_t_pattern(notes2)
    print(f"  Output: {' '.join(corrected2)}")
    print(f"  Corrections: {len(corr2)} - {corr2}")

    # Test case 3: With Kat (should not trigger)
    notes3 = ["Dha", "Kat", "Re", "Ki", "T"]
    print(f"\nTest 3 - Contains Kat:")
    print(f"  Input:  {' '.join(notes3)}")
    corrected3, _, corr3 = correct_ti_re_ki_t_pattern(notes3)
    print(f"  Output: {' '.join(corrected3)}")
    print(f"  Corrections: {len(corr3)}")

    # Test case 4: Broken by open stroke (should not trigger)
    notes4 = ["Ti", "Re", "Dha", "Ki", "T"]
    print(f"\nTest 4 - Broken by open stroke:")
    print(f"  Input:  {' '.join(notes4)}")
    corrected4, _, corr4 = correct_ti_re_ki_t_pattern(notes4)
    print(f"  Output: {' '.join(corrected4)}")
    print(f"  Corrections: {len(corr4)}")

    # Test case 5: Multiple patterns
    notes5 = ["Dha", "T", "Ti", "Re", "Ki", "Ghe", "Ki", "T", "Ti", "Re", "Dhin"]
    print(f"\nTest 5 - Multiple patterns:")
    print(f"  Input:  {' '.join(notes5)}")
    corrected5, _, corr5 = correct_ti_re_ki_t_pattern(notes5)
    print(f"  Output: {' '.join(corrected5)}")
    print(f"  Corrections: {len(corr5)} - {corr5}")

    print(f"\n{'='*80}")
    print("✅ TESTING COMPLETE")
    print(f"{'='*80}")
