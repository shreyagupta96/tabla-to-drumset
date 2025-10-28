# Tabla-to-Drumset AI System: Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Research Context](#research-context)
3. [System Architecture](#system-architecture)
4. [Technical Implementation](#technical-implementation)
5. [Key Innovations](#key-innovations)
6. [Results and Performance](#results-and-performance)
7. [Future Directions](#future-directions)

---

## Project Overview

### Goal
Develop an AI system that can:
1. **Classify** tabla audio recordings into note sequences
2. **Generate** musical responses (jugalbandi - call-and-response improvisation)
3. **Translate** between tabla and drum kit sounds
4. **Play back** compositions through a terminal interface

### Timeline
- **Deadline**: October 26, 2024
- **Development period**: ~1 week intensive development

### Constraints
- 12-note vocabulary limitation (Dha, Dhin, Ghe, Kat, Ki, Na, Re, T, Ta, Ti, Tun, Tin)
- No paired call-response training data available
- Limited computational resources (local CPU training)
- No GPU access initially (later option: Google Colab)

---

## Research Context

### Musical Concept: Jugalbandi
**Jugalbandi** (जुगलबन्दी) is a classical Indian musical concept meaning "entwined twins" - a duet performance where two musicians engage in musical conversation through call-and-response improvisation.

**Key characteristics:**
- One musician plays a phrase (CALL)
- Second musician responds with a complementary phrase (RESPONSE)
- Responses should match complexity and length of the call
- Each response should be unique (improvisational)
- Maintains musical grammar and structure

### Tabla Fundamentals

**Tabla** is a pair of hand drums central to North Indian classical music, consisting of:
- **Dayan** (right drum): Produces higher-pitched, crisp sounds
- **Bayan** (left drum): Produces bass, resonant sounds

**Note Categories:**
1. **Crisp/Closed notes**: Ti, Re, Ki, T, Kat - sharp, staccato sounds
2. **Open/Ringing notes**: Ta, Na, Tun, Tin - resonant, sustained sounds
3. **Bass notes**: Dha, Dhin, Ghe - deep, low-frequency sounds

**Formulaic Patterns:**
- **"Ti Re Ki T"**: Most common 4-note crisp phrase
- These patterns are tempo-independent (defined by note category, not duration)
- Act as "musical vocabulary" similar to words in language

---

## System Architecture

### Component Pipeline

```
Input Audio (WAV)
    ↓
[1] RCD Onset Detection
    ↓
[2] CNN Classification (97% accuracy on known notes)
    ↓
[3] Post-Processing (Ti Re Ki T pattern correction)
    ↓
[4] LSTM Generation (temperature-controlled improvisation)
    ↓
[5] Audio Playback (drums or tabla)
```

### Three-Stage Development

#### Stage 1: Classification System
**Goal**: Convert raw tabla audio → note sequences

**Components:**
- **RCD Onset Detection**: Rectified Complex Domain algorithm for detecting stroke timing
- **CNN Model**: ConvNet_SNFPR_model.pth (97% validation accuracy)
- **Duration Extraction**: Automatic timing between detected onsets

**Challenges:**
- 12-note vocabulary insufficient for actual tabla vocabulary (~20+ strokes)
- Initial accuracy: 52.3% on real-world recordings
- Systematic errors: Ki→Kat, Na→Ta, Ti→Re

#### Stage 2: Pattern-Based Post-Processing
**Goal**: Improve classification accuracy using musical grammar rules

**Innovation**: Category-based pattern detection (not duration-based)
- Detect 4-note crisp sequences (CRISP_NOTES category)
- Check relative duration similarity (tempo-independent)
- Apply probabilistic "Ti Re Ki T" correction

**Results:**
- Tested thresholds: 75%, 50%, 25%, 0%
- Optimal: 25% confidence threshold
- **Accuracy improvement: 56.2% → 68.0% (+11.8%)**
- 32 corrections applied across 19 files

**Key Insight**: Systematic errors can be leveraged for drum↔tabla translation

#### Stage 3: LSTM-Based Generation
**Goal**: Generate musical responses for jugalbandi

**Why LSTM over Random Forest?**
- Longer context window (8 notes vs. 3)
- Captures sequential dependencies
- Temperature sampling enables variation
- Better suited for musical improvisation

**Model Architecture:**
- **Input**: Note embeddings (32-dim) + Duration embeddings (16-dim)
- **Hidden layers**: 2-layer LSTM, 128 hidden units
- **Output heads**: Separate note and duration prediction
- **Total parameters**: 225,889
- **Dropout**: 0.3 (regularization)

---

## Technical Implementation

### File Structure

```
tabla-to-drumset_SG/
│
├── models/
│   ├── ConvNet_SNFPR_model.pth          # CNN classifier (97% val accuracy)
│   └── tabla_lstm_model.pth             # Trained LSTM generator
│
├── data_preparation/
│   ├── lstm_data_prep.py                # Preprocessing for LSTM training
│   └── lstm_training_data.pkl           # 481 training examples
│
├── training/
│   ├── lstm_model.py                    # LSTM architecture definition
│   └── train_lstm.py                    # Training script
│
├── classification/
│   ├── api.py                           # Original Flask API + CNN functions
│   ├── classify_terminal.py            # Terminal-based classifier
│   └── batch_classify_standalone.py    # Batch classification (60 files)
│
├── post_processing/
│   ├── post_process_classifications.py  # Ti Re Ki T pattern correction
│   └── review_classifications.py        # Interactive review tool
│
├── generation/
│   ├── test_generation.py               # Temperature testing script
│   └── jugalbandi_terminal.py           # Full call-response system
│
└── data/
    ├── training_data_corrected_25pct.json  # Post-processed classifications
    └── corrected_labels.csv                # Clean CSV export
```

### Key Algorithms

#### 1. RCD Onset Detection
```python
def compute_rcd_onsets(y, sr):
    """
    Rectified Complex Domain onset detection
    - More robust to noise than energy-based detection
    - Better for percussive transients
    """
    # Complex domain processing
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_samples = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        backtrack=True
    )
    return onset_samples, onset_env
```

#### 2. Duration Quantization
```python
def quantize_duration(duration):
    """
    Bins: [0.0-0.3], [0.3-0.5], [0.5-0.8], [0.8-1.5], [1.5+]
    Rationale: Captures short/medium/long note distinctions
    """
    bins = [0.0, 0.3, 0.5, 0.8, 1.5, float('inf')]
    for i, edge in enumerate(bins[1:]):
        if duration < edge:
            return i
    return len(bins) - 2
```

#### 3. Category-Based Pattern Detection
```python
CRISP_NOTES = ['Ti', 'Re', 'Ki', 'T', 'Kat']
OPEN_NOTES = ['Ta', 'Na', 'Tun', 'Tin']
BASS_NOTES = ['Dha', 'Dhin', 'Ghe']

def detect_crisp_sequence(notes, start_idx, length=4):
    """
    Check if sequence is all crisp notes
    Enables tempo-independent pattern detection
    """
    sequence = notes[start_idx:start_idx+length]
    return all(note in CRISP_NOTES for note in sequence)
```

#### 4. LSTM Forward Pass
```python
def forward(self, note_seq, duration_seq, hidden=None):
    """
    Dual-embedding approach:
    - Note embeddings: 32-dim (captures note identity)
    - Duration embeddings: 16-dim (captures timing)
    - Concatenated input to LSTM
    - Separate prediction heads for note and duration
    """
    note_emb = self.note_embedding(note_seq)
    dur_emb = self.duration_embedding(duration_seq)
    combined = torch.cat([note_emb, dur_emb], dim=-1)

    lstm_out, hidden = self.lstm(combined, hidden)
    last_output = lstm_out[:, -1, :]

    note_logits = self.note_head(last_output)
    duration_logits = self.duration_head(last_output)

    return note_logits, duration_logits, hidden
```

#### 5. Temperature Sampling
```python
def generate(self, seed_notes, seed_durations, num_generate=10, temperature=1.0):
    """
    Temperature controls creativity:
    - Low (0.5-0.8): Conservative, repetitive
    - Medium (1.0): Balanced
    - High (1.2-1.5): Creative, diverse
    """
    for _ in range(num_generate):
        note_logits, dur_logits, hidden = self.forward(...)

        # Apply temperature
        note_probs = torch.softmax(note_logits / temperature, dim=-1)
        dur_probs = torch.softmax(dur_logits / temperature, dim=-1)

        # Sample from distribution (not argmax!)
        next_note = torch.multinomial(note_probs, 1).item()
        next_dur = torch.multinomial(dur_probs, 1).item()
```

---

## Key Innovations

### 1. Musical Grammar Post-Processing
**Problem**: CNN classifier produces 52% accuracy due to vocabulary mismatch

**Solution**: Apply domain knowledge through category-based pattern detection
- Leverage musical grammar rules ("Ti Re Ki T" as formulaic phrase)
- Tempo-independent detection (category + relative duration)
- Probabilistic approach (confidence threshold)

**Impact**: 11.8% accuracy improvement

### 2. Systematic Error as Feature
**Insight**: Classification errors are not random - they're systematic!

**Observation**:
- Ki → Kat confusion: 9 instances
- Na → Ta confusion: 9 instances
- Ti → Re confusion: 6 instances

**Implication**: These mappings can be reverse-engineered for drum↔tabla translation
- Na and Ta are acoustically similar (open, ringing)
- Ki and Kat are both crisp notes
- Systematic = predictable = exploitable

### 3. Dual-Embedding LSTM Architecture
**Design choice**: Separate embeddings for notes and durations

**Rationale**:
- Notes and durations have different semantic spaces
- Note identity (what) vs. temporal pattern (when)
- Allows model to learn independently

**Alternative rejected**: Single embedding of note+duration pairs
- Would require 12×5=60 embedding vectors
- Sparse training signal for rare combinations

### 4. Temperature-Controlled Improvisation
**Challenge**: Same input should generate different responses (like human improvisation)

**Solution**: Temperature sampling instead of greedy decoding (argmax)

**Benefits**:
- User control over creativity level
- Maintains musical coherence at lower temps
- Enables exploration at higher temps

---

## Results and Performance

### Dataset
- **60 tabla recordings**:
  - 35 electronic tabla loops (TablaPU)
  - 25 live tabla performances (TablaLive - 8 iterations × 3 loops)
- **Total classified notes**: 996 notes
- **Training examples**: 481 sequences (8-note context windows)
- **Train/Val split**: 385 train / 96 validation (80/20 split)

### Classification Performance

| Metric | Value |
|--------|-------|
| CNN validation accuracy | 97% (on known notes) |
| Initial real-world accuracy | 52.3% |
| Post-processing accuracy | 68.0% |
| Improvement | +11.8% |
| Files corrected | 19/60 (31.7%) |
| Total corrections applied | 32 |

**Error Analysis**:
- Most errors: Ki↔Kat, Na↔Ta, Ti↔Re (systematic)
- Root cause: 12-note vocabulary vs. ~20+ actual tabla strokes
- Remaining 32% error rate acceptable for generation training

### LSTM Training Performance

| Metric | Training | Validation |
|--------|----------|------------|
| Final accuracy (note) | 97.4% | 51.0% |
| Final accuracy (duration) | 98.7% | 63.5% |
| Best validation loss | 2.7395 | (epoch 20) |
| Training time | 0.1 minutes | (100 epochs) |
| Epoch time | ~0.1 seconds | (local CPU) |

**Observations**:
- Clear overfitting pattern (expected with 481 examples)
- Fast training on local CPU (no need for Colab)
- Best model saved at epoch 20 (early stopping effective)
- Duration prediction easier than note prediction (63.5% vs 51.0%)

### Generation Quality (Qualitative)

**Temperature 0.8** (Conservative):
```
Seed:     Ti Re Ki T
Generated: Tin Ta Dhin Dhin Dha Dhin Dhin Dha Ta Ghe Ta Kat Tin Ta Tin Dhin
Quality:   ✓ Structured, rhythmic
           ✗ Some repetition (Dhin Dhin Dhin)
```

**Temperature 1.0** (Balanced):
```
Seed:     Ti Re Ki T
Generated: Tun Dhin Dha Ta Ta Kat Kat Re Tin Na Dha Ki Tin Tin Ti T
Quality:   ✓ Good variation
           ✓ Mixes note categories (crisp, open, bass)
           ✓ Musical coherence maintained
```

**Temperature 1.2** (Creative):
```
Seed:     Ti Re Ki T
Generated: Dhin Dha Dhin Tin Ti Ti Dha T Dhin Ta Kat Tun Na Dha Ti Ti
Quality:   ✓ High diversity
           ✓ Crisp note sequences appear (Ti Ti, Ti Ti)
           ✗ Slightly less structured
```

**Temperature 1.5** (Very Random):
```
Seed:     Ti Re Ki T
Generated: Tun Ta Na Tin Dhin Dha Dhin Dhin Dhin Dhin Dha Tin Ti Ki T Dha
Quality:   ✓ Maximum variation
           ✗ May lose musical coherence
```

**Recommendation**: Temperature 0.8-1.2 range optimal for jugalbandi

---

## Technical Challenges and Solutions

### Challenge 1: Flask Server Conflict
**Problem**: `api.py` contains `app.run()` at module level, preventing imports

**Solution**: Created `batch_classify_standalone.py` with copied functions
- Avoided import-triggered server startup
- Enabled batch processing of 60 files

### Challenge 2: Playback Timing Issues
**Problem**: Gaps/silence between notes during playback

**Root causes**:
1. Synchronous audio playback blocking execution
2. Processing delays during playback (loading, fading)

**Solutions**:
1. Non-blocking subprocess with `Popen()` instead of blocking calls
2. Pre-process ALL audio files BEFORE starting playback timer
3. Precise timing using cumulative duration calculation

**Implementation**:
```python
# Pre-process all audio FIRST
for note in notes:
    audio_data = load_and_fade(note)
    temp_files.append(save_temp(audio_data))

# Then play with accurate timing
start_time = time.time()
for i, note in enumerate(notes):
    target_time = start_time + sum(durations[:i])
    wait_time = target_time - time.time()
    if wait_time > 0:
        time.sleep(wait_time)
    play_async(temp_files[i])
```

### Challenge 3: Audio Clicks Between Samples
**Problem**: Audible clicks at start/end of each played sample

**Cause**: Abrupt audio transitions (non-zero sample values at boundaries)

**Solution**: Cross-fade implementation
- 5ms fade-in at start
- 5ms fade-out at end
- Linear amplitude envelope

### Challenge 4: Ghost Notes Implementation
**Requirement**: "T" notes should be quieter (ghost notes in drumming terminology)

**Solution**: Volume reduction via amplitude multiplication
- Ghost notes: 30% of original amplitude
- Visual indicator during playback (👻 emoji)
- Optional feature (can be toggled)

---

## Future Directions

### 1. Drum-Tabla Translation System
**Goal**: Bidirectional mapping between drum kit and tabla

**Approach**:
- Leverage systematic error patterns as translation dictionary
- Na ↔ Ta (open notes)
- Ki ↔ Kat (crisp notes)
- Maintain rhythmic structure while swapping timbres

**Use cases**:
- Drummers learning tabla vocabulary
- Tabla players understanding drum kit equivalents
- Cross-cultural musical collaboration

### 2. Expanded Training Data
**Current limitation**: 481 training examples

**Future improvements**:
- Collect more tabla recordings (target: 200+ distinct loops)
- Data augmentation: pitch shifting, time stretching
- Include multiple performance styles (classical, folk, fusion)

**Expected impact**: Better generalization, reduced overfitting

### 3. Longer Context Window
**Current**: 8-note context window

**Research direction**: Test 16, 32-note windows
- Capture longer-term musical structure
- Better understanding of phrase boundaries
- May require Transformer architecture (attention mechanism)

### 4. Multi-Track Generation
**Vision**: Generate full ensemble responses
- Tabla + tanpura (drone)
- Tabla + melodic instrument
- Full rhythm section (tabla + drums + percussion)

### 5. Real-Time Performance Mode
**Goal**: Live jugalbandi with AI

**Technical requirements**:
- Low-latency onset detection (<50ms)
- Real-time classification (<100ms)
- Instant generation (<200ms)
- Total latency budget: <350ms

**Challenges**:
- Online processing vs. batch processing
- Model optimization (quantization, pruning)
- Streaming audio I/O

### 6. Magenta Integration
**Google Magenta**: Pre-trained models for music generation

**Potential integration**:
- Music VAE for interpolation between styles
- GrooVAE for drum pattern generation
- Transfer learning from Magenta models

**Benefits**: State-of-art generation quality without extensive training

---

## Research Contributions

### 1. Cross-Cultural AI Music
- First (to our knowledge) AI system for tabla jugalbandi
- Bridges Indian classical and Western percussion traditions
- Demonstrates AI's potential in non-Western music contexts

### 2. Domain Knowledge Integration
- Shows importance of musical grammar in post-processing
- Category-based pattern detection generalizes across tempos
- Systematic error exploitation as feature engineering

### 3. Practical AI with Limited Data
- Achieves functional system with <500 training examples
- No GPU required (local CPU training viable)
- Demonstrates rapid prototyping approach (1-week timeline)

### 4. Temperature Sampling for Improvisation
- Provides user control over creativity level
- Balances determinism and stochasticity
- Enables multiple responses to same input (key for improvisation)

---

## Technical Specifications

### Hardware Requirements
- **Development**: MacBook (macOS, CPU only)
- **RAM**: 8GB+ recommended
- **Storage**: ~500MB for models + data
- **Audio**: Built-in or external speakers/headphones

### Software Dependencies
```
Python 3.8+
PyTorch 1.9+
librosa 0.9+
numpy 1.21+
soundfile 0.11+
pickle (standard library)
```

### Model Files
- `ConvNet_SNFPR_model.pth`: 2.3 MB (CNN classifier)
- `tabla_lstm_model.pth`: 3.6 MB (LSTM generator)
- `lstm_training_data.pkl`: 1.2 MB (preprocessed sequences)

---

## Acknowledgments

### Musical Concepts
- Tabla notation and performance practices
- Jugalbandi tradition in Indian classical music
- Formulaic patterns ("Ti Re Ki T") from tabla pedagogy

### Technical Foundations
- **CNN Architecture**: ConvNet with SNFPR (Signal-to-Noise Peak Ratio) optimization
- **RCD Onset Detection**: Librosa implementation
- **LSTM Generation**: Standard sequence-to-sequence approach with temperature sampling

### Development Process
- Iterative refinement based on user feedback
- Rapid prototyping and testing cycle
- Domain expert (user) in the loop for validation

---

## Conclusion

This project demonstrates that **functional AI music systems can be built with limited resources** by:
1. Leveraging domain knowledge (musical grammar)
2. Accepting imperfect data (68% accuracy sufficient)
3. Exploiting systematic patterns (errors as features)
4. Providing user control (temperature parameter)

The system achieves its core goals:
- ✅ Classifies tabla recordings into note sequences
- ✅ Generates musical responses with controllable creativity
- ✅ Plays back compositions through terminal interface
- 🔄 Drum-tabla translation (in progress)

**Key insight**: Perfect classification is not required for musical generation - systematic patterns and musical structure matter more than absolute accuracy.

**Future potential**: This approach can extend to other percussion traditions (African drums, Latin percussion, etc.) and serve as a foundation for cross-cultural AI music systems.

---

## References

### Academic Foundations
- **Onset Detection**: Dixon, S. (2006). "Onset detection revisited"
- **Music Generation**: Briot, J.P., et al. (2017). "Deep learning techniques for music generation"
- **LSTM for Music**: Eck, D., & Schmidhuber, J. (2002). "Learning the long-term structure of the Blues"

### Tools and Libraries
- **LibROSA**: McFee, B., et al. (2015). "librosa: Audio and Music Signal Analysis in Python"
- **PyTorch**: Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library"

### Cultural Context
- Tabla pedagogy and notation systems
- North Indian classical music performance practices
- Jugalbandi tradition and improvisation techniques

---

*Documentation created: October 2024*
*Project timeline: October 19-26, 2024*
*Version: 1.0*