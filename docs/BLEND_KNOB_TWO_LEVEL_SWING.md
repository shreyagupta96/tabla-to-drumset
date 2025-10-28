# Blend Knob with Two-Level Swing Preservation

## Overview

The **Blend Knob V2** system implements controlled AI variation with **two-level swing preservation**, addressing the critical hierarchical timing structure in tabla music: beat-level groove (MACRO) and subdivision patterns (MICRO).

This document explains the complete pipeline from input analysis to audio synthesis, with a focus on how subdivision patterns like "Ti Re Ki T" are preserved while applying AI variation.

---

## The Problem We Solved

### Initial Approach (Wrong) ❌

```python
# Incorrect blending - ignored rhythmic intent
if use_generated_bar:
    output_notes = generated_notes
    output_durations = input_bar_durations  # ← Just copying!
```

**Why this was wrong:**
- Generated notes have their own rhythmic density (fast vs slow)
- Forcing them to use input durations ignores the model's rhythmic intent
- Example: Generated `Tin` (fast, ~0.15s) forced to use input duration (0.42s) → rhythmically wrong

### Two-Level Swing Approach (Correct) ✅

```python
# Extract MACRO (beat-level swing) + MICRO (subdivision patterns)
beat_patterns = extract_beat_patterns(input)
swing_beat_iois = [0.64, 0.61, 0.63, 0.62, ...]  # Beat-to-beat timing

# Generate with Model C
gen_notes, gen_duration_bins = model.generate(...)

# Map bins → swing durations (two-level)
for each beat in generated:
    match pattern from input
    apply swing-adjusted durations
```

**Why this is correct:**
- Model predicts **rhythmic density** (bins 0-4)
- We map bins to **natural timing** from input (swing template)
- Both beat-level groove AND subdivision structure preserved

---

## Two-Level Timing Structure

Tabla rhythm has **two hierarchical timing levels**:

### MACRO: Beat-Level Swing (Groove)

The main pulse with natural human timing variation:

```
Beat:    |----1----|----2----|----3----|----4----|
IOIs:    0.64s      0.61s     0.63s     0.62s
         ↑ Varies by ~5% (swing/groove)
```

### MICRO: Subdivision Patterns

Notes within each beat create rhythmic density:

```
Beat 1: |-----Dhin-----|  (1 note filling whole beat)
        Duration: 0.64s

Beat 2: |--Ghe--|--Re--|  (2 notes splitting beat)
        Durations: 0.32s, 0.32s

Beat 3: |Ti|Ki|Kat|Tun|  (4 fast subdivisions in one beat!)
        Durations: 0.16s, 0.17s, 0.17s, 0.17s
```

**Key Insight:** The "Ti Ki Kat Tun" pattern represents **one beat subdivided into 4 fast notes**, not 4 separate beats!

---

## Complete Pipeline

### Stage 1: Input Analysis & Classification

**Input:** Audio file (e.g., `Ektaal.wav`)

```python
# CNN classification
notes, durations, onset_samples = classify_input(audio_file, cnn_model)

# Example output
notes = ['Dhin', 'Dhin', 'Dha', 'Ghe', 'Re', 'Ti', 'Ki', 'Kat', 'Tun', ...]
durations = [0.64, 0.61, 0.32, 0.32, 0.16, 0.17, 0.17, 0.17, ...]
```

**Result:** 115 notes detected, 38.62s total

---

### Stage 2: Tempo & Beat Detection

**Detect beat-level timing (MACRO):**

```python
# Find IOIs that indicate beat-level notes (not subdivisions)
beat_threshold = 0.4s  # IOIs above this are likely beats

# Extract beat positions
beat_positions = onset_samples[durations >= beat_threshold]
beat_iois = np.diff(beat_positions) / sample_rate

# Example: Ektaal
beat_iois = [0.64, 0.61, 1.20, 0.63, 0.59, 1.81, ...]
```

**Result:**
- 29 beats detected
- Beat IOI range: 0.592s - 1.811s (natural variation!)
- Mean: 1.200s, CV: 0.413 (swing coefficient)

---

### Stage 3: Beat Pattern Extraction (MICRO)

**Extract subdivision patterns within each beat:**

```python
for each beat:
    # Find notes within this beat
    beat_notes = notes[onset_samples >= beat_start & < beat_end]

    # Calculate relative timing (proportions of beat)
    relative_iois = beat_iois / beat_duration

    # Quantize to duration bins
    duration_bins = tuple([quantize_duration(d) for d in beat_durations])

    # Store pattern
    pattern_key = (num_notes, duration_bins)
    beat_patterns[pattern_key].append({
        'relative_iois': relative_iois,
        'actual_iois': beat_iois,
        'notes': beat_notes
    })
```

**Example Patterns Extracted:**

```
Pattern 1: (1, (2,)) - 1 note, bin 2 (medium)
  Examples: Dhin (0.64s), Dha (0.61s), Na (0.63s)
  Relative IOI: [1.0] (fills whole beat)

Pattern 2: (7, (2,1,1,0,0,0,0)) - 7 notes, mixed bins
  Examples: Dhin(0.32) Dha(0.16) Ghe(0.16) Re(0.04) Ti(0.04) Ki(0.04) Kat(0.04)
  Relative IOIs: [0.30, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]
  ↑ Fast subdivisions like "Ti Re Ki T"!

Pattern 3: (3, (2,1,1)) - 3 notes, triplet feel
  Examples: Tun(0.42) Na(0.21) Ta(0.21)
  Relative IOIs: [0.50, 0.25, 0.25]

Pattern 4: (4, (1,0,1,1)) - 4 notes, even subdivisions
  Examples: Dhin(0.32) Re(0.16) Na(0.32) Na(0.32)
  Relative IOIs: [0.30, 0.15, 0.30, 0.25]
```

**Result:** 4 unique beat patterns extracted

---

### Stage 4: Generation with Pattern Matching

**Generate with Model C and apply two-level swing:**

```python
# Step 4.1: Generate with Model C
gen_note_indices, gen_duration_bins = lstm_model.generate(
    seed_notes=seed_note_indices,
    seed_durations=seed_duration_bins,
    taal_id=taal_id,
    num_generate=num_generate,
    temperature=temperature
)

# Example generation
gen_notes = ['Tin', 'T', 'T', 'Ghe', 'Ghe', 'T', 'Dhin', 'Ghe', ...]
gen_bins = [0, 2, 0, 0, 1, 0, 0, 0, 0, 2, ...]
             ↑ fast  ↑ slow  ↑ fast...
```

**Step 4.2: Group into beat chunks**

```python
# Accumulate duration bins until ~1 beat worth
beat_chunks = group_notes_into_beats(gen_duration_bins, avg_beat_duration=1.2)

# Example grouping
Beat Chunk 1: indices [0,1,2] → bins [0,2,0] → notes ['Tin', 'T', 'T']
Beat Chunk 2: indices [3,4,5,6,7,8] → bins [0,0,1,0,0,0] → 6 notes
...
```

**Step 4.3: Match patterns and apply swing**

```python
for each beat_chunk:
    chunk_bins = tuple(gen_duration_bins[chunk_indices])
    swing_duration = swing_beat_iois[beat_idx % len(swing_beat_iois)]

    # Try to match pattern
    if (num_notes, chunk_bins) in beat_patterns:
        # Exact match - use input pattern's relative IOIs
        pattern = random.choice(beat_patterns[(num_notes, chunk_bins)])
        actual_durations = pattern['relative_iois'] * swing_duration

    elif matching_patterns_by_note_count:
        # Fallback: match by note count only
        pattern = random.choice(patterns_with_same_note_count)
        actual_durations = pattern['relative_iois'] * swing_duration

    else:
        # Synthesize: use bin midpoints as proportions
        relative_iois = BIN_MIDPOINTS[chunk_bins] / sum(BIN_MIDPOINTS)
        actual_durations = relative_iois * swing_duration
```

**Example:**

```
Generated Beat Chunk: [0, 0, 0, 1] → 4 notes, mostly fast
Swing duration for this beat: 0.63s

Try exact match: (4, (0,0,0,1)) → Not found
Try note-count match: (4, ...) → Found Pattern 4!
  Pattern 4 relative IOIs: [0.30, 0.15, 0.30, 0.25]

Apply swing:
  actual_durations = [0.30, 0.15, 0.30, 0.25] * 0.63
                   = [0.19s, 0.09s, 0.19s, 0.16s]

Result: Fast subdivisions preserved with input's groove!
```

**Generation Results:**
- 32 notes generated
- Pattern matching: 0 exact, 5 by note-count, 3 synthesized
- Duration range: 0.078s - 0.894s (preserves fast AND slow!)

---

### Stage 5: Bar Segmentation & Blending

**Segment input into complete bars:**

```python
# For Ektaal (12 beats per bar)
input_bars, leftover = segment_into_bars(notes, durations, beats_per_bar=12)

# Example
Bar 1: notes[0:12]   (12 notes)
Bar 2: notes[12:24]  (12 notes)
...
Bar 9: notes[96:108] (12 notes)
Leftover: notes[108:115] (7 notes)
```

**Generate variation pool (2 bars):**

```python
# Generate 2 complete bars matching input meter
gen_bar_A = generate_one_bar(beats_per_bar=12, temperature=1.0)
gen_bar_B = generate_one_bar(beats_per_bar=12, temperature=1.0)

# Example
Gen Bar A: ['Dhin', 'Dha', 'Re', 'Dhin', 'Na', 'Tin', 'T', 'T', ...]
Gen Bar B: ['Dhin', 'Na', 'Dhin', 'Ti', 'Na', 'Dhin', 'Dhin', 'Tin', ...]
```

**Bar-level blending:**

```python
for each bar in input_bars:
    random_value = random()

    if random_value < blend_ratio:
        # USE GENERATED BAR
        selected_bar = random.choice([gen_bar_A, gen_bar_B])

        # Apply two-level swing to generated bar
        output_notes.extend(selected_bar.notes)
        output_durations.extend(apply_two_level_swing(
            selected_bar.duration_bins,
            beat_patterns,
            swing_beat_iois
        ))
    else:
        # KEEP INPUT BAR (exact durations)
        output_notes.extend(bar.notes)
        output_durations.extend(bar.durations)

# Always keep leftover as input
output_notes.extend(leftover.notes)
output_durations.extend(leftover.durations)
```

**Example at blend=0.5 (50%):**

```
Blending Decisions:
  Bar 1: ✨ Gen Bar B (roll: 0.48 < 0.5)  ← AI variation
  Bar 2: ✨ Gen Bar B (roll: 0.09 < 0.5)  ← AI variation
  Bar 3: ✨ Gen Bar B (roll: 0.27 < 0.5)  ← AI variation
  Bar 4: ✨ Gen Bar A (roll: 0.16 < 0.5)  ← AI variation
  Bar 5: 📥 INPUT    (roll: 0.78 > 0.5)  ← Keep original
  Bar 6: ✨ Gen Bar B (roll: 0.26 < 0.5)  ← AI variation
  Bar 7: ✨ Gen Bar B (roll: 0.07 < 0.5)  ← AI variation
  Bar 8: ✨ Gen Bar A (roll: 0.09 < 0.5)  ← AI variation
  Bar 9: ✨ Gen Bar A (roll: 0.33 < 0.5)  ← AI variation
  Leftover: 📥 INPUT (always)            ← Keep original

Result: 8/9 bars (88.9%) generated
```

---

### Stage 6: Audio Synthesis

**Synthesize with exact onset timing:**

```python
# Calculate exact onset times from durations
onset_times = np.insert(np.cumsum(output_durations), 0, 0.0)

# Place samples at exact positions
for note, onset_time in zip(output_notes, onset_times):
    onset_sample = int(onset_time * sample_rate)
    audio_data = load_sample(f"tabla/{note}.wav")

    # Apply crossfades where samples overlap
    full_audio[onset_sample:onset_sample+len(audio_data)] += audio_data

# Normalize and save
tabla_audio = normalize(full_audio)
drums_audio = synthesize_with_drum_mapping(output_notes, onset_times)

save("output_tabla.wav", tabla_audio)
save("output_drums.wav", drums_audio)
```

**Result:**
- Both tabla and drums use **identical onset times**
- Two-level swing preserved in final audio
- Natural timing variation maintained

---

## Musical Benefits

### What This System Preserves

✅ **Beat-Level Swing (MACRO)**
```
Input beats:  0.64s, 0.61s, 0.63s, 0.62s, 0.64s
Output beats: 0.64s, 0.61s, 0.63s, 0.62s, 0.64s
↑ Same groove, same feel
```

✅ **Subdivision Structure (MICRO)**
```
Input:  |Ti|Ki|Kat|Tun| (4 fast notes in one beat)
        0.16, 0.17, 0.17, 0.17

Generated with same bins (0,0,0,0):
Output: |Re|Ti|Ki|T|
        0.16, 0.17, 0.17, 0.17
↑ Different notes, same rhythmic density
```

✅ **Model's Rhythmic Intent**
```
Model predicts: Tin (bin 0 = fast)
Without swing template: 0.15s (bin midpoint) ❌ Mechanical
With swing template: 0.16s (sampled from input) ✅ Natural
```

✅ **Natural Variation**
```
Three "Bin 1" notes might get:
  0.35s, 0.41s, 0.38s (varied, natural)
Instead of:
  0.40s, 0.40s, 0.40s (mechanical, robotic)
```

---

## Usage Examples

### Basic Usage

```bash
# 50% blend with Ektaal (12 beats per bar)
python blend_knob_v2_with_swing.py Ektaal.wav 0.5 --beats_per_bar 12

# 20% blend with Teental (16 beats per bar)
python blend_knob_v2_with_swing.py Teentaal.wav 0.2 --beats_per_bar 16

# 100% AI generation
python blend_knob_v2_with_swing.py input.wav 1.0 --beats_per_bar 12
```

### Advanced Options

```bash
# Higher temperature for more creativity
python blend_knob_v2_with_swing.py Ektaal.wav 0.5 \
    --beats_per_bar 12 \
    --temperature 1.5 \
    --output_dir my_variations

# Exact reproduction (testing)
python blend_knob_v2_with_swing.py Ektaal.wav 0.0 --beats_per_bar 12
```

---

## Technical Details

### Duration Bins

```python
DURATION_BINS = [0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]
BIN_MIDPOINTS = [0.15, 0.4, 0.65, 1.15, 2.0]

Bin 0: 0.0 - 0.3s   (Very fast, 16th note feel)
Bin 1: 0.3 - 0.5s   (Fast, 8th note feel)
Bin 2: 0.5 - 0.8s   (Medium, quarter note feel)
Bin 3: 0.8 - 1.5s   (Slow, half note feel)
Bin 4: 1.5s+        (Very slow, whole note feel)
```

### Pattern Matching Priority

1. **Exact Match**: `(num_notes, duration_bins)` matches exactly
2. **Note Count Match**: Same `num_notes`, different bins
3. **Synthesized**: Use bin midpoints as equal proportions

### Beat Detection Threshold

```python
beat_threshold = 0.4s  # IOIs above this are beat-level notes

# Example classification:
0.64s → beat-level ✓
0.32s → subdivision ✗
0.16s → subdivision ✗
```

---

## Implementation Files

### Core Script

**`blend_knob_v2_with_swing.py`** (877 lines)
- Complete pipeline implementation
- Command-line interface
- All 6 stages integrated

### Key Functions

```python
# Stage 1
classify_input(file_path, cnn_model)

# Stage 2
detect_tempo_and_beats(durations, onset_samples, sr)

# Stage 3
extract_beat_patterns(notes, durations, onset_samples, beat_positions, sr)

# Stage 4
group_notes_into_beats(gen_duration_indices, avg_beat_duration)
match_and_apply_pattern(gen_chunk_bins, beat_patterns, swing_beat_duration)

# Stage 5
segment_into_bars(notes, durations, beats_per_bar)
blend_bars_with_two_level_swing(input_bars, gen_bars_pool, leftover, blend_ratio, ...)

# Stage 6
synthesize_audio(notes, durations, folder='tabla')
```

---

## Example Output

### Test: Ektaal.wav at blend=0.5

**Input:**
- 115 notes, 38.62s
- Teental pattern
- Natural swing and subdivisions

**Processing:**
- 29 beats detected
- 4 unique subdivision patterns extracted
- Beat IOI range: 0.592s - 1.811s (41% CV)
- 9 complete bars + 7 leftover notes

**Generation:**
- 2-bar variation pool created (Bar A, Bar B)
- Each bar: 12 notes matching Ektaal meter

**Blending:**
- 8/9 bars (88.9%) generated (lucky rolls at 50% blend!)
- 1/9 bars kept as input
- Leftover always preserved

**Output:**
- 115 notes, 31.26s
- Two-level swing preserved
- Both tabla and drum versions generated

**Files:**
```
generated_blend_v2/
├── Ektaal_blend_0.50_tabla.wav
└── Ektaal_blend_0.50_drums.wav
```

---

## Future Enhancements

### Possible Improvements

1. **Adaptive Beat Detection**: Use autocorrelation or spectral methods for more robust tempo tracking

2. **Taal-Specific Pattern Libraries**: Build separate pattern libraries for each taal

3. **Constraint-Based Generation**: Ensure generated bars follow taal rules (sam placement, theka structure)

4. **Multi-Bar Generation**: Generate longer sequences (4-8 bars) for more variety

5. **Dynamic Blending**: Vary blend ratio over time (start conservative, get wilder)

6. **Pattern Similarity Matching**: Use edit distance or DTW for better pattern matching

7. **Subdivision Classification**: Explicitly detect subdivision levels (quarter, eighth, sixteenth)

---

## Comparison: Old vs New System

### Old System (blend_knob_generator.py) ❌

```python
# Wrong approach
if use_generated:
    output_notes = gen_notes
    output_durations = input_durations  # ← Ignored rhythmic intent!
```

**Problems:**
- Fast generated notes forced to be slow
- Slow generated notes forced to be fast
- No understanding of subdivision structure
- Mechanical timing (no natural variation)

### New System (blend_knob_v2_with_swing.py) ✅

```python
# Correct approach
if use_generated:
    output_notes = gen_notes

    # Map generated bins to swing template
    for beat_chunk in gen_beats:
        match pattern from input
        apply swing-adjusted durations based on bins

    output_durations = swing_mapped_durations
```

**Benefits:**
- Generated bins mapped to natural swing IOIs
- Subdivision structure preserved
- Beat-level groove maintained
- Natural timing variation
- Musically coherent results

---

## Conclusion

The **Blend Knob V2 with Two-Level Swing** system successfully addresses the hierarchical timing structure of tabla music by preserving both:

1. **MACRO timing**: Beat-level swing and groove
2. **MICRO timing**: Subdivision patterns and rhythmic density

This approach ensures that AI-generated variations maintain the natural feel and rhythmic intent of the original performance while allowing for creative exploration through the blend control.

---

*Documentation created: October 2024*
*Feature: Two-Level Swing Preservation*
*Project: Tabla-to-Drumset AI System*
*Researcher: Shreya Gupta*