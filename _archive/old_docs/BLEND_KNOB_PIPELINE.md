# Blend Knob Pipeline Documentation

## Overview

The **Blend Knob** feature allows users to control the amount of AI variation applied to an input tabla performance while preserving the original timing, swing, and groove. This document explains how duration bins, swing templates, and subdivisions work together to create musically coherent variations.

---

## The Blend Knob Concept

**Blend Parameter Range: 0.0 to 1.0**

- **0.0 (0%)**: Output is 100% original input (exact reproduction)
- **1.0 (100%)**: Output is 100% AI-generated variations (new note patterns)
- **0.2 (20%)**: Output is 20% generated + 80% input (subtle variations)

### Design Decisions

✅ **Blend notes only, keep input durations** - Preserves timing/groove
✅ **Bar-aware blending** - Switches complete bars to preserve musical structure
✅ **Generate 2 complete bars** - Creates a reusable variation pool
✅ **Random selection** - Different variations each run
✅ **Bar-length matching** - Generated bars match input meter (16 beats for Teental, 12 for Ektaal, etc.)

---

## Pipeline Architecture

### Stage 1: Input Analysis

**Input:** `Teentaal.wav` (51 notes, 19.10 seconds)

#### 1.1 Onset Detection & Classification
```
CNN Classifier detects tabla strokes:
Notes:     Dha   Dhin  Dhin  Dha   Dha   Tin   Tin   Ta    Na   ...
Onsets:    0.00s 0.35s 0.76s 1.14s 1.53s 1.91s 2.33s 2.72s 3.11s ...
IOIs:      0.35  0.41  0.38  0.39  0.38  0.42  0.39  ...
           ↑ Inter-Onset Intervals (time between strokes)
```

#### 1.2 Meter Detection
```
Detect taal: Teental (16 beats per bar)
Tempo: ~140 BPM
```

#### 1.3 Bar Segmentation
```
Input segmentation:
  Bar 1: Notes 0-15   (16 notes) - Complete bar
  Bar 2: Notes 16-31  (16 notes) - Complete bar
  Bar 3: Notes 32-47  (16 notes) - Complete bar
  Leftover: Notes 48-50 (3 notes) - Incomplete bar
```

---

### Stage 2: Duration Quantization

The LSTM Model C predicts **discrete duration bins**, not exact durations.

#### 2.1 Duration Bin Definitions
```python
DURATION_BINS = [0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]

Bin 0: 0.0 - 0.3s   (Very fast subdivisions - 16th note feel)
Bin 1: 0.3 - 0.5s   (Fast subdivisions - 8th note feel)
Bin 2: 0.5 - 0.8s   (Medium subdivisions - quarter note feel)
Bin 3: 0.8 - 1.5s   (Slow subdivisions - half note feel)
Bin 4: 1.5s+        (Very slow/held notes - whole note feel)
```

#### 2.2 Quantize Input Durations
```
Input IOI: 0.35s → Bin 1 (fast subdivision)
Input IOI: 0.41s → Bin 1 (fast subdivision)
Input IOI: 0.38s → Bin 1 (fast subdivision)
Input IOI: 0.39s → Bin 1 (fast subdivision)
Input IOI: 0.16s → Bin 0 (very fast subdivision)
Input IOI: 0.42s → Bin 1 (fast subdivision)
```

**Key Insight:** Even within the same bin, durations vary (0.35s vs 0.41s = 17% difference). This variation creates **swing/groove**.

---

### Stage 3: Swing Template Extraction

The **swing template** captures the natural timing variations within each duration bin.

#### 3.1 Cluster IOIs by Bin
```python
duration_clusters = {
    0: [0.16, 0.15, 0.19, 0.21],           # Very fast IOIs from input
    1: [0.35, 0.41, 0.38, 0.39, 0.38,      # Fast IOIs (most common)
        0.42, 0.39, 0.41, 0.41, 0.39, ...],
    2: [0.63, 0.61, 0.59, 0.65],           # Medium IOIs
    3: [1.15, 1.20, 1.10],                 # Slow IOIs
    4: []                                   # Very slow (if any)
}
```

#### 3.2 Swing Template Statistics
```
Bin 1 statistics:
  - Count: 40 IOIs
  - Mean: 0.396s
  - Std Dev: 0.023s
  - Range: 0.35s - 0.42s
  - CV (Coefficient of Variation): 0.058

This captures the "swing" - the natural human timing variation!
```

---

### Stage 4: AI Generation (Model C)

#### 4.1 Generate Variation Pool
```
Generate 2 complete bars matching input meter:
  - Taal: Teental (16 beats per bar)
  - Seed: Last 16 notes from input
  - Temperature: 1.0 (balanced creativity)

Generated Bar A (16 notes):
  Notes: Dhin T Tin Tin Dha Dha Dhin Na Dhin Na Na Tin Dhin Dhin Dhin T
  Bins:  1    0  1   0   1   1   1    1  1    1  1  1   1    1    1    0

Generated Bar B (16 notes):
  Notes: T Dhin Tin T T Dhin Dhin Dhin Dha Dhin Dhin T Tin Dhin Dhin Dhin
  Bins:  0 1    1   0 0 1    1    1    1   1    1    0 1   1    1    1
```

#### 4.2 Map Generated Bins → Swing Durations (for reference only)
```
Model C predicts bins, but we will DISCARD these durations during blending!

Generated Bar A with swing-mapped durations:
  Dhin(1) → Sample from Bin 1 cluster → 0.41s
  T(0)    → Sample from Bin 0 cluster → 0.15s
  Tin(1)  → Sample from Bin 1 cluster → 0.38s
  Tin(0)  → Sample from Bin 0 cluster → 0.16s
  ...

Note: These are used ONLY if blend = 1.0 and we want generated timing too.
For blend knob feature, we use INPUT durations instead!
```

---

### Stage 5: Bar-Level Blending

#### 5.1 Blend Algorithm

**For each complete input bar:**
```python
for bar_index in range(num_complete_bars):
    # Roll the dice
    random_value = random.uniform(0, 1)

    if random_value < blend_ratio:
        # Use generated bar (randomly pick Bar A or Bar B)
        selected_gen_bar = random.choice([gen_bar_A, gen_bar_B])
        output_notes[bar_index] = selected_gen_bar.notes
        output_durations[bar_index] = input_bar[bar_index].durations  # ← Always input!
    else:
        # Use input bar
        output_notes[bar_index] = input_bar[bar_index].notes
        output_durations[bar_index] = input_bar[bar_index].durations
```

**Leftover/incomplete bars:** Always use input (no blending)

#### 5.2 Example: Blend = 0.2 (20%)

**Input:** 3 complete bars + 3 leftover notes

**Blending Process:**
```
Bar 1 (16 notes): random() = 0.65 → 0.65 > 0.2 → Keep INPUT
  Notes: Dha Dhin Dhin Dha Dha Dhin Dhin Dha Dha Tin Tin Ta Na Dhin Dhin Dha
  Durations: [0.35, 0.41, 0.38, 0.38, 0.39, 0.39, 0.37, 0.42, ...]

Bar 2 (16 notes): random() = 0.12 → 0.12 < 0.2 → Use GENERATED Bar A ✨
  Notes: Dhin T Tin Tin Dha Dha Dhin Na Dhin Na Na Tin Dhin Dhin Dhin T
  Durations: [0.41, 0.39, 0.39, 0.39, 0.41, 0.41, 0.38, 0.39, ...]  ← INPUT durations!

Bar 3 (16 notes): random() = 0.89 → 0.89 > 0.2 → Keep INPUT
  Notes: Dha Dhin Dhin Dha Dha Dhin Dhin Dha Dha Tin Tin Ta Na Dhin Dhin Dha
  Durations: [0.39, 0.41, 0.41, 0.39, 0.41, 0.41, 0.38, 0.41, ...]

Leftover (3 notes): Always INPUT
  Notes: Na Ta Dhin
  Durations: [0.21, 0.19, 0.25]
```

**Result:** ~20% of the bars have AI variations, but ALL bars preserve original timing!

---

### Stage 6: Audio Synthesis

#### 6.1 Calculate Onset Times
```python
# Use input durations to calculate exact onset times
onset_times = np.cumsum([0] + output_durations)

Example:
  onset_times = [0.00, 0.35, 0.76, 1.14, 1.53, 1.91, 2.33, ...]
```

#### 6.2 Place Samples at Exact Times
```python
for note, onset_time in zip(output_notes, onset_times):
    onset_sample = int(onset_time * sample_rate)
    sample_audio = load_audio(f"tabla/{note}.wav")

    # Place at exact position in output buffer
    output_audio[onset_sample : onset_sample + len(sample_audio)] += sample_audio

    # Apply crossfades where samples overlap
```

#### 6.3 Output Files
```
Generated files:
  - Teentaal_blend_0.2_tabla.wav  (Tabla sounds)
  - Teentaal_blend_0.2_drums.wav  (Drum kit mapping)

Both use IDENTICAL onset times (swing preserved)
```

---

## Subdivisions Explained

**Subdivision** = How notes divide the rhythmic pulse

### Fast Subdivisions (Bin 0, 1)
```
Beat:    |----1----|----2----|----3----|----4----|
Notes:   Ti Re Ki T Dha Dhin Ta Na Ti Re Ki T Dha ...
         ↑ 16th note feel (many notes per beat)
IOIs:    0.15 0.15 0.16 0.35 0.38 0.41 0.39 0.35 ...
Bins:    0    0    0    1    1    1    1    1    ...
```

### Medium Subdivisions (Bin 2)
```
Beat:    |----1----|----2----|----3----|----4----|
Notes:   Dha       Tin       Ta        Na
         ↑ Quarter note feel (one note per beat)
IOIs:    0.63      0.61      0.59      0.65
Bins:    2         2         2         2
```

### Slow Subdivisions (Bin 3, 4)
```
Beat:    |----1----|----2----|----3----|----4----|
Notes:   Dhaaaaaaaaaaaa      Tiiiiiiin
         ↑ Held notes (sustained across beats)
IOIs:    1.15                1.20
Bins:    3                   3
```

### Mixed Subdivisions (Natural Tabla Phrases)
```
Beat:    |----1----|----2----|----3----|----4----|
Notes:   Ti Re Ki T   Dhaaaaaa   Ta Kat     Na
         ↑ fast       ↑ slow     ↑ fast     ↑ fast
IOIs:    0.15 0.15 0.16  1.20     0.35 0.38  0.39
Bins:    0    0    0     3        1    1     1
```

**Key Insight:** The duration bins capture subdivision density, and the swing template captures natural human timing variation within each density level.

---

## How Swing is Preserved

### Traditional Approach (Without Swing Template)
```
Model predicts: Bin 1
Converted to:   0.40s (bin midpoint)
Every Bin 1 note gets exactly 0.40s → Mechanical, no groove ❌
```

### Our Approach (With Swing Template)
```
Model predicts: Bin 1
Swing template has: [0.35, 0.41, 0.38, 0.39, 0.38, 0.42, 0.39, ...]
Randomly sample:    0.41s (this time)
Next Bin 1 note:    0.38s (different!)
Next Bin 1 note:    0.39s (different again!)

Result: Natural timing variation preserved ✅
```

### With Blend Knob
```
We don't even use the generated bins!
We directly use input durations → 100% original swing preserved ✅✅✅
```

---

## Complete Example: Blend = 0.3

**Input:** Teentaal.wav (51 notes, 3.2 bars)

### Step-by-Step Process

**1. Classify input**
```
51 notes detected: Dha Dhin Dhin Dha Dha Tin Tin Ta Na Dhin ...
51 durations: 0.35 0.41 0.38 0.38 0.39 0.38 0.42 0.39 0.41 ...
```

**2. Detect meter & segment**
```
Taal: Teental (16 beats)
Bar 1: Notes 0-15
Bar 2: Notes 16-31
Bar 3: Notes 32-47
Leftover: Notes 48-50 (keep as input)
```

**3. Generate 2 variation bars**
```
Gen Bar A: Dhin T Tin Tin Dha Dha Dhin Na Dhin Na Na Tin Dhin Dhin Dhin T
Gen Bar B: T Dhin Tin T T Dhin Dhin Dhin Dha Dhin Dhin T Tin Dhin Dhin Dhin
```

**4. Blend at 30%**
```
Bar 1: random() = 0.85 → Keep INPUT
Bar 2: random() = 0.22 → Use Gen Bar B ✨
Bar 3: random() = 0.56 → Keep INPUT
```

**5. Apply input durations**
```
Bar 1: INPUT notes + INPUT durations
Bar 2: Gen Bar B notes + INPUT Bar 2 durations ← AI variation with original groove!
Bar 3: INPUT notes + INPUT durations
Leftover: INPUT
```

**6. Synthesize audio**
```
Place samples at exact onset times calculated from input durations
Export: Teentaal_blend_0.3_tabla.wav, Teentaal_blend_0.3_drums.wav
```

**Result:** ~30% of bars have AI creativity, 100% has original timing/groove!

---

## Implementation Notes

### Key Functions

1. **`classify_input(audio_file)`**
   - CNN classification
   - Returns: notes, durations

2. **`detect_meter_and_segment(notes, durations)`**
   - Meter detection
   - Bar segmentation
   - Returns: bars, taal_id, beats_per_bar

3. **`generate_variation_pool(seed_notes, seed_durations, taal_id, beats_per_bar)`**
   - Generate 2 complete bars using Model C
   - Returns: gen_bar_A, gen_bar_B

4. **`blend_bars(input_bars, gen_bars, blend_ratio)`**
   - Bar-level random blending
   - Always uses input durations
   - Returns: output_notes, output_durations

5. **`synthesize_audio(notes, durations, folder='tabla')`**
   - Place samples at exact onset times
   - Apply crossfades
   - Returns: audio_array

### Configuration

```python
BLEND_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # Presets
TEMPERATURE = 1.0  # Model C creativity (0.8-1.2 recommended)
NUM_GEN_BARS = 2   # Size of variation pool
```

---

## Musical Benefits

### Why This Design Works

✅ **Preserves Groove**: Original swing/timing always maintained
✅ **Bar Coherence**: Complete phrases swapped, not random notes
✅ **Timbral Variation**: New note choices create interest
✅ **Structural Integrity**: Meter and subdivision patterns preserved
✅ **Efficient Generation**: Only generate 2 bars regardless of input length
✅ **Replayability**: Random selection creates different variations each time

### Use Cases

**Blend = 0.0**: Exact reproduction (useful for testing/validation)
**Blend = 0.1-0.2**: Subtle variations (practice with slight changes)
**Blend = 0.3-0.5**: Moderate variations (creative jugalbandi)
**Blend = 0.7-0.9**: Heavy AI influence (experimental compositions)
**Blend = 1.0**: Full AI generation (maximum creativity)

---

## Future Extensions

### Possible Enhancements

1. **Dynamic Blend**: Blend ratio changes over time (start conservative, get wilder)
2. **Phrase-Aware Blend**: Blend at phrase level, not just bar level
3. **Constrained Generation**: Ensure generated bars follow taal rules (sam placement, theka patterns)
4. **Multiple Models**: Blend between different Model C checkpoints
5. **User-Guided Generation**: Let user mark which bars should vary
6. **Blend Durations Too**: Add a second knob for duration blending (preserve swing template but allow variation)

---

## References

- **MODEL_C_DOCUMENTATION.md** - Model C architecture and training
- **SWING_DRUM_MAPPING_SUMMARY.md** - Swing template extraction
- **TIMING_VERIFICATION_SUMMARY.md** - Onset time preservation
- **BAR_AWARE_PIPELINE.md** - Bar segmentation methodology

---

*Documentation created: October 2024*
*Feature: Blend Knob for Controlled AI Variation*
*Project: Tabla-to-Drumset AI System*
*Researcher: Shreya Gupta*
