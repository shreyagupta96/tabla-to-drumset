# Final Model: 4-Taal Bar-Aware LSTM

**Quick Reference Guide for the Production Model**

---

## Model File

**Path**: `models/best_bar_aware_lstm.pth` (11 MB)
**Architecture**: Meter-Conditional LSTM with Embedding Regularization
**Best Epoch**: 49
**Validation Loss**: 2.640

---

## Capabilities

### Supported Taals (4)

| Taal ID | Name | Beats per Bar | Training Sequences |
|---------|------|---------------|-------------------|
| 0 | Teental | 16 | 105 |
| 1 | Ektaal | 12 | 83 |
| 2 | Jhaptaal | 10 | 18 |
| 3 | Rupak | 7 | 42 |

### Note Vocabulary (12)
`Dha`, `Dhin`, `Ghe`, `Kat`, `Ki`, `Na`, `Re`, `T`, `Ta`, `Ti`, `Tun`, `Tin`

### Duration Categories (5)

| Bin | Range | Type | Example Use |
|-----|-------|------|-------------|
| 0 | 0.0-0.3s | Very Short | Chougun (4x speed) |
| 1 | 0.3-0.5s | Short | Dugun (2x speed) |
| 2 | 0.5-0.8s | Medium | Regular tempo |
| 3 | 0.8-1.5s | Long | Sustained notes |
| 4 | 1.5s+ | Very Long | Final notes, pauses |

---

## Model Specifications

### Architecture
- **Type**: 2-layer LSTM with triple embedding (notes + durations + taal)
- **Parameters**: 974,177
- **Hidden Size**: 256
- **Dropout**: 0.3
- **Embedding Regularization**: λ = 0.1

### Training Data
- **Total Sequences**: 248 (188 train / 60 val)
- **Format**: 2-bar context → 1-bar target prediction
- **Training Files**: 15 .wav files (Teental, Ektaal, Jhaptaal, Rupak variations)
- **Validation Files**: 5 .wav files

### Performance
- **Validation Loss**: 2.640 (best at epoch 49)
- **Training Time**: ~120 seconds (100 epochs, CPU)
- **Inference Speed**: Real-time (<50ms per generation)

---

## Quick Start

### Load Model

```python
import torch
from meter_conditional_lstm import create_model

# Load checkpoint
checkpoint = torch.load('models/best_bar_aware_lstm.pth', map_location='cpu')
metadata = checkpoint['metadata']

# Create model
model, regularizer = create_model(
    vocab_size=12,
    num_duration_bins=5,
    num_taals=4,  # 4 taals supported
    hidden_size=256,
    num_layers=2,
    dropout=0.3,
    note_labels=metadata['note_labels']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded! Val loss: {checkpoint['val_loss']:.3f}")
```

### Generate Tabla Sequence

```python
# Taal mapping
TAAL_IDS = {
    'Teental': 0,   # 16 beats
    'Ektaal': 1,    # 12 beats
    'Jhaptaal': 2,  # 10 beats
    'Rupak': 3      # 7 beats
}

# Note to index mapping
note_to_idx = {note: i for i, note in enumerate(metadata['note_labels'])}

# Example: Generate 32 notes in Teental
seed_notes = [note_to_idx['Ti'], note_to_idx['Re'],
              note_to_idx['Ki'], note_to_idx['T']]  # Ti Re Ki T
seed_durations = [2, 2, 2, 2]  # Medium duration (bin 2)

# Generate
generated_notes, generated_durations = model.generate(
    seed_notes=seed_notes,
    seed_durations=seed_durations,
    taal_id=TAAL_IDS['Teental'],
    num_generate=32,
    temperature=1.0,  # 0.8-1.2 recommended
    device='cpu'
)

# Convert indices to note names
note_labels = metadata['note_labels']
generated_sequence = [note_labels[idx] for idx in generated_notes]
print("Generated:", ' '.join(generated_sequence))
```

---

## Key Features

### 1. Meter-Conditional Generation
Model understands and respects taal structure. Generates taal-appropriate patterns for all 4 supported taals.

### 2. Embedding Regularization
Learns acoustic similarity between notes:
- **Na ↔ Ta**: Both open, ringing
- **Tin ↔ Tun**: Both mid-range
- **Ki ↔ Kat**: Both crisp, closed

### 3. Variable-Length Sequences
Handles different phrase lengths using packed sequences (16, 32, 48+ notes).

### 4. Speed-Aware Duration Modeling
5 duration bins capture different laykaari (tempo variations): Vilambit, Dugun, Chougun.

---

## Usage Recommendations

### Temperature Settings

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.8 | Conservative, structured | Practice patterns, learning |
| 1.0 | **Balanced (recommended)** | Jugalbandi, performance |
| 1.2 | Creative, diverse | Improvisation, exploration |

### Taal Selection

**For best results**:
- Teental (105 sequences) → Most robust
- Ektaal (83 sequences) → Well-trained
- Rupak (42 sequences) → Good coverage
- Jhaptaal (18 sequences) → Limited training data

---

## Training Details

### Dataset Composition

**Training Files (15)**:
- 7 Teental variations (basic, dugun, chougun, variations)
- 4 Ektaal variations (basic, dugun, variations)
- 2 Jhaptaal files (basic, chougun)
- 2 Rupak files (basic, chougun)

**Validation Files (5)**:
- 2 Teental Variation 3
- 1 Ektaal Variation 2
- 1 Jhap Dugun
- 1 Rupak #02

### Hyperparameters

```python
{
    'learning_rate': 0.001,
    'batch_size': 16,
    'hidden_size': 256,
    'num_layers': 2,
    'dropout': 0.3,
    'reg_weight': 0.1,  # Embedding regularization
    'optimizer': 'Adam',
    'scheduler': 'ReduceLROnPlateau'
}
```

---

## File Dependencies

### Required Files
- `models/best_bar_aware_lstm.pth` (11 MB model checkpoint)
- `meter_conditional_lstm.py` (model architecture)
- `taal_utils.py` (taal ID mappings)

### Optional Files
- `tabla/*.wav` (12 note samples for playback)
- `drums/*.wav` (12 drum kit samples for translation)

---

## Integration Points

### 1. Terminal Interface
```bash
python classify_terminal.py input/audio.wav --generate
```

### 2. Flask API
```python
@app.route('/generate', methods=['POST'])
@require_api_key
def generate_tabla():
    # Load model once at startup
    # Generate on demand
    pass
```

### 3. Batch Generation
```bash
python export_model_audio_improved.py
```

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| **Validation Loss** | 2.640 |
| **Training Time** | ~120s (100 epochs, CPU) |
| **Generation Speed** | <50ms per note |
| **Model Size** | 11 MB |
| **RAM Usage** | ~500 MB |
| **Device** | CPU-compatible |

---

## Limitations

1. **Jhaptaal**: Only 18 training sequences (may produce less varied output)
2. **Dataset Size**: 248 total sequences (small for deep learning)
3. **Repetition**: Can generate consecutive repeats (avg ~17 per sequence)
4. **Fixed Vocabulary**: Only 12 notes (tabla has 20+ strokes in practice)

---

## Version History

- **v1.0** (Oct 2024): 4-taal meter-conditional LSTM
- Best model saved at epoch 49
- Trained on corrected data with embedding regularization

---

## Citation

```
Tabla-to-Drumset AI System
4-Taal Bar-Aware LSTM with Embedding Regularization
Shreya Gupta, 2024-2025

Model: models/best_bar_aware_lstm.pth
Architecture: MeterConditionalLSTM (974,177 parameters)
Taals: Teental (16), Ektaal (12), Jhaptaal (10), Rupak (7)
```

---

## Related Documentation

- `MODEL_C_DOCUMENTATION.md` - Detailed research documentation
- `PROJECT_DOCUMENTATION.md` - Full system overview
- `BAR_AWARE_PIPELINE.md` - Training pipeline
- `taal_utils.py` - Taal mapping utilities

---

*Last Updated: October 2024*
*Model Version: 1.0 (4-Taal Final)*