# Bar-Aware LSTM Pipeline - Complete Implementation Plan

## Overview

Transform 12 long tabla files into a bar-aware LSTM training dataset with meter-conditional generation.

**Key Innovation**: Simplified bar segmentation (no beat alignment) + meter-conditional LSTM (Option A for unknown meters).

---

## Pipeline Phases

### Phase 1: Batch Classification
**Goal**: Classify all 12 long files into note sequences

**Input**:
- 12 wav files from `/Taal Variations – 12-01-2025/Bounces/`
- Each file: ~140 seconds, contains complete taal cycles

**Process**:
```python
For each file:
1. Load audio (140s)
2. Detect onsets using RCD method
3. Classify each stroke with CNN (ConvNet_SNFPR_model.pth)
4. Save: notes, durations, onset_samples
```

**Output**: `classified_long_files/`
```json
{
  "file": "Teental_Dugun.wav",
  "notes": ["Dha", "Dhin", "Dhin", "Dha", ...],  // 500-700 notes
  "durations": [0.4, 0.4, 0.4, 0.4, ...],
  "onset_samples": [145000, 180000, 215000, ...],
  "num_notes": 493,
  "total_duration_sec": 138.0
}
```

**Files to Process**:
- Teental files (7): Basic Teental, Teental Dugun Variation 2, Teental Dugun Variation#02, Teental Variation 1/2/3
- Ektaal files (5): Basic Ektaal, Ektaal Dugun Variation, Ektaal Dugun, Ektaal Variation 1/2

**Expected Output**:
- 12 JSON files with classified notes
- Total: ~5000-7000 classified notes across all files

---

### Phase 2: Meter Detection & Bar Segmentation
**Goal**: Segment classified notes into complete bars using meter detection

**Input**:
- Original wav files
- Classified notes + onset_samples from Phase 1

**Process**:
```python
For each file:
1. Run hybrid_meter() on audio
   → Get: meter (16 or 12), bar_start_samples, tempo

2. Segment notes by bar boundaries (SIMPLIFIED APPROACH):
   For each bar i:
     - Bar start: bar_start_samples[i]
     - Bar end: bar_start_samples[i+1]
     - Find all notes with onset_samples in [start, end)
     - Group into bar

3. Discard incomplete bars at file start/end
```

**Key Simplification**:
```python
# NO beat-level alignment needed!
# Just use bar boundaries directly

def segment_by_bars(notes, durations, onset_samples, bar_start_samples):
    bars = []

    for i in range(len(bar_start_samples) - 1):
        bar_start = bar_start_samples[i]
        bar_end = bar_start_samples[i + 1]

        # Find all notes in this bar
        bar_notes = []
        bar_durs = []

        for j, onset in enumerate(onset_samples):
            if bar_start <= onset < bar_end:
                bar_notes.append(notes[j])
                bar_durs.append(durations[j])

        bars.append({
            'notes': bar_notes,
            'durations': bar_durs,
            'num_notes': len(bar_notes)
        })

    return bars
```

**Output**: `segmented_bars/`
```json
{
  "file": "Teental_Dugun.wav",
  "taal": "Teental",
  "meter": 16,
  "tempo": 162.2,
  "bars": [
    {
      "bar_num": 0,
      "notes": ["Dha", "Dhin", "Dhin", "Dha", "Na", "Tin", ...],
      "durations": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, ...],
      "num_notes": 15  // Variable! (12-20 for Teental)
    },
    {
      "bar_num": 1,
      "notes": ["Dha", "Tin", "Tin", "Ta", "Dhin", ...],
      "durations": [0.4, 0.4, 0.4, 0.4, 0.4, ...],
      "num_notes": 17  // Different from bar 0!
    },
    ...
  ],
  "total_bars": 23,
  "bar_start_samples": [143520, 715200, 1295040, ...]
}
```

**Expected Output**:
- Teental files: ~110 complete bars (7 files × ~15 bars each)
- Ektaal files: ~120 complete bars (5 files × ~24 bars each)
- Total: ~230 complete bars

---

### Phase 3: Training Dataset Preparation
**Goal**: Create meter-specific training sequences

**Input**:
- Segmented bars from Phase 2

**Process**:

#### 3A: Separate by Taal
```python
teental_bars = []  # All bars from Teental files (meter=16)
ektaal_bars = []   # All bars from Ektaal files (meter=12)

for file_data in segmented_files:
    if file_data['meter'] == 16:
        teental_bars.extend(file_data['bars'])
    elif file_data['meter'] == 12:
        ektaal_bars.extend(file_data['bars'])
```

#### 3B: Create Taal-Specific Sequences
```python
def create_sequences(bars, taal_name, meter):
    """
    Create training sequences: 2 bars context → 1 bar target
    """
    sequences = []

    for i in range(len(bars) - 2):
        # Context: 2 consecutive bars
        context_notes = bars[i]['notes'] + bars[i+1]['notes']
        context_durs = bars[i]['durations'] + bars[i+1]['durations']

        # Target: Next bar
        target_notes = bars[i+2]['notes']
        target_durs = bars[i+2]['durations']

        sequences.append({
            'input_notes': context_notes,
            'input_durations': context_durs,
            'input_length': len(context_notes),  // Track actual length
            'target_notes': target_notes,
            'target_durations': target_durs,
            'target_length': len(target_notes),
            'taal': taal_name,
            'meter': meter,
            'taal_id': 0 if meter == 16 else 1  // Teental=0, Ektaal=1
        })

    return sequences

# Create sequences for both taals
teental_sequences = create_sequences(teental_bars, 'Teental', 16)
ektaal_sequences = create_sequences(ektaal_bars, 'Ektaal', 12)

# Combine
all_sequences = teental_sequences + ektaal_sequences
```

**Expected Sequences**:
```python
# Teental (meter=16):
# ~110 bars → ~108 sequences (110 - 2)
# Context length: ~28-34 notes (2 bars)
# Target length: ~14-17 notes (1 bar)

Example Teental sequence:
{
  'input_notes': ['Dha', 'Dhin', 'Dhin', ..., 'Dha'],  // 30 notes
  'input_durations': [0.4, 0.4, 0.4, ..., 0.4],
  'input_length': 30,
  'target_notes': ['Na', 'Tin', 'Tin', ...],  // 16 notes
  'target_durations': [0.4, 0.4, 0.4, ...],
  'target_length': 16,
  'taal': 'Teental',
  'meter': 16,
  'taal_id': 0
}

# Ektaal (meter=12):
# ~120 bars → ~118 sequences
# Context length: ~22-28 notes (2 bars)
# Target length: ~11-14 notes (1 bar)

Example Ektaal sequence:
{
  'input_notes': ['Dhin', 'Dhin', 'Dha', ..., 'Ge'],  // 25 notes
  'input_durations': [0.4, 0.4, 0.4, ..., 0.4],
  'input_length': 25,
  'target_notes': ['Tin', 'Tin', 'Ta', ...],  // 12 notes
  'target_durations': [0.4, 0.4, 0.4, ...],
  'target_length': 12,
  'taal': 'Ektaal',
  'meter': 12,
  'taal_id': 1
}
```

#### 3C: Train/Validation Split
```python
# Split: 80% train, 20% validation
# Split by files to avoid data leakage

import random
random.shuffle(all_sequences)

split_idx = int(0.8 * len(all_sequences))
train_sequences = all_sequences[:split_idx]
val_sequences = all_sequences[split_idx:]

# ~226 total sequences → ~181 train, ~45 validation
```

**Output**: `training_data/bar_aware_dataset.pkl`
```python
{
  'train': [
    {'input_notes': [...], 'target_notes': [...], 'taal_id': 0, ...},
    ...
  ],  // ~181 sequences
  'validation': [
    {'input_notes': [...], 'target_notes': [...], 'taal_id': 1, ...},
    ...
  ],  // ~45 sequences
  'note_to_idx': {'Dha': 0, 'Dhin': 1, ..., 'Tin': 11},
  'idx_to_note': {0: 'Dha', 1: 'Dhin', ..., 11: 'Tin'},
  'taal_to_idx': {'Teental': 0, 'Ektaal': 1},
  'idx_to_taal': {0: 'Teental', 1: 'Ektaal'},
  'duration_bins': [0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]
}
```

---

### Phase 4: LSTM Architecture
**Goal**: Meter-conditional LSTM with variable-length sequences

**Architecture**:
```python
class MeterConditionalLSTM(nn.Module):
    def __init__(self, vocab_size, num_taals=2, hidden_size=256):
        super().__init__()

        # Embeddings
        self.note_embedding = nn.Embedding(vocab_size, 64)
        self.duration_embedding = nn.Embedding(5, 16)  # 5 duration bins
        self.taal_embedding = nn.Embedding(num_taals + 1, 32)  # +1 for "any" (taal_id=2)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=64 + 16 + 32,  # note + duration + taal
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.3
        )

        # Output heads
        self.note_head = nn.Linear(hidden_size, vocab_size)
        self.duration_head = nn.Linear(hidden_size, 5)

    def forward(self, notes, durations, taal_id, lengths):
        batch_size = notes.size(0)

        # Embed inputs
        note_emb = self.note_embedding(notes)
        dur_emb = self.duration_embedding(durations)
        taal_emb = self.taal_embedding(taal_id)

        # Broadcast taal embedding to all timesteps
        taal_emb = taal_emb.unsqueeze(1).expand(-1, notes.size(1), -1)

        # Concatenate
        x = torch.cat([note_emb, dur_emb, taal_emb], dim=-1)

        # Pack for variable lengths
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Unpack
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Predict
        note_logits = self.note_head(lstm_out)
        dur_logits = self.duration_head(lstm_out)

        return note_logits, dur_logits
```

**Key Features**:
1. **Variable-length sequences**: Uses pack_padded_sequence for bars with different note counts
2. **Taal embedding**: Learns meter-specific patterns (Teental vs Ektaal vs "any")
3. **Separate heads**: Predicts notes and durations independently

---

### Phase 5: Generation with Meter Conditioning (Option A)
**Goal**: Generate meter-appropriate responses

**Option A Implementation** (use mixed training data for unknown meters):
```python
def generate_response(audio_file, lstm_model, cnn_model, temperature=1.0):
    """
    Generate tabla response conditioned on input meter
    """
    # Step 1: Classify input
    input_notes, input_durations = classify_audio(audio_file, cnn_model)

    # Step 2: Detect meter
    meter_result = hybrid_meter(audio_file)
    detected_meter = meter_result.get('final_meter')

    # Step 3: Map meter to taal_id (OPTION A)
    if detected_meter == 16:
        taal_id = 0  # Teental
        taal_name = 'Teental'
        print(f"✅ Detected Teental (16 beats)")
    elif detected_meter == 12:
        taal_id = 1  # Ektaal
        taal_name = 'Ektaal'
        print(f"✅ Detected Ektaal (12 beats)")
    else:
        # OPTION A: Use mixed training (special "any" embedding)
        taal_id = 2  # "any" - uses both Teental and Ektaal patterns
        taal_name = 'Mixed'
        print(f"⚠️  Unknown meter ({detected_meter} beats), using mixed patterns")

    # Step 4: Extract seed (last 2 bars worth of notes)
    # For Teental: ~30 notes, for Ektaal: ~25 notes, for unknown: use last 30
    seed_length = 30
    seed_notes = input_notes[-seed_length:]
    seed_durations = input_durations[-seed_length:]

    # Step 5: Generate
    generated_notes, generated_durations = lstm_model.generate(
        seed_notes=seed_notes,
        seed_durations=seed_durations,
        taal_id=taal_id,
        num_generate=30,  # Generate ~2 bars worth
        temperature=temperature
    )

    return generated_notes, generated_durations, taal_name, detected_meter
```

**Behavior by Meter**:
- **Meter = 16**: Uses Teental patterns only (taal_id=0)
- **Meter = 12**: Uses Ektaal patterns only (taal_id=1)
- **Meter = 7, 10, or other**: Uses mixed patterns (taal_id=2, trained on both taals)

**Why Option A Works**:
```python
# During training, taal_id=2 never appears
# But the embedding layer learns:
# - taal_id=0 → Teental patterns
# - taal_id=1 → Ektaal patterns
# - taal_id=2 → Initialized randomly, will activate mixed patterns

# The LSTM learns to use taal embedding as conditioning
# When taal_id=2, it falls back to general tabla patterns
# that work across both taals
```

---

## Expected Results

### Dataset Statistics
```
Training Data:
- Total sequences: ~226
- Train split: ~181 (80%)
- Validation split: ~45 (20%)

Sequence Breakdown:
- Teental sequences: ~108 (47%)
  - Context: 28-34 notes (2 bars)
  - Target: 14-17 notes (1 bar)
- Ektaal sequences: ~118 (53%)
  - Context: 22-28 notes (2 bars)
  - Target: 11-14 notes (1 bar)
```

### Generation Examples

**Input: Teental audio (16 beats)**
```
Input: Dha Dhin Dhin Dha | Dha Dhin Dhin Dha | Na Tin Tin Na | Dhin Dhin Dha
Detected: Teental (16 beats)
Generated: Dha Tin Tin Ta | Na Dhin Dhin Dha | Ti Re Ki T | Dhin Dhin Dha
✅ Uses Teental-specific patterns
```

**Input: Ektaal audio (12 beats)**
```
Input: Dhin Dhin Dha Ge | Tin Tin Ta Dhin | Dhin Dha Dha Tin
Detected: Ektaal (12 beats)
Generated: Tin Ta Dha Ge | Dhin Dhin Dha Tin | Tin Ta Dha Ge
✅ Uses Ektaal-specific patterns
```

**Input: Rupak audio (7 beats) - Unknown meter**
```
Input: Tin Tin Na | Dhin Na | Dhin Na
Detected: Unknown (7 beats)
Generated: Dha Dhin Na | Ta Tin | Dha Dhin
⚠️  Uses mixed patterns (general tabla vocabulary)
```

---

## Implementation Timeline

### Week 1
- [x] Phase 1: Batch classification (1-2 days)
- [ ] Phase 2: Meter detection & segmentation (1 day)
- [ ] Phase 3: Dataset preparation (1 day)

### Week 2
- [ ] Phase 4: LSTM implementation (2 days)
- [ ] Phase 5: Training (2 days)
- [ ] Phase 6: Evaluation & comparison (1 day)

---

## Success Criteria

### Minimum Viable
- [ ] Successfully classify all 12 files
- [ ] Segment into ~230 complete bars
- [ ] Create ~226 training sequences
- [ ] Train LSTM to convergence
- [ ] Generate bar-aligned output (no mid-bar termination)

### Stretch Goals
- [ ] Less looping than current model (max 3 consecutive repeats)
- [ ] Meter-appropriate generation (Teental sounds like Teental)
- [ ] Embedding regularization works (Na/Ta, Tin/Tun variation)
- [ ] Better than current model on 5+ metrics

---

## Files to Create

```
tabla-to-drumset_SG/
├── batch_classify_long_files.py       # Phase 1
├── segment_by_bars.py                 # Phase 2
├── prepare_training_data.py           # Phase 3
├── meter_conditional_lstm.py          # Phase 4 (new architecture)
├── train_bar_aware_lstm.py            # Phase 5
├── generate_with_meter.py             # Phase 5 (generation)
├── compare_models.py                  # Evaluation
│
├── classified_long_files/             # Phase 1 output
│   ├── Teental_Dugun_classified.json
│   └── ...
│
├── segmented_bars/                    # Phase 2 output
│   ├── Teental_Dugun_bars.json
│   └── ...
│
└── training_data/                     # Phase 3 output
    └── bar_aware_dataset.pkl
```

---

## Next Steps

Ready to implement Phase 1: Batch Classification?
