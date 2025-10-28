"""
Compare Models B, C, and D on the same Ektaal file

Model Matrix (we have 3/4):
                   Original Data    Corrected Data (69 fixes)
With Reg:          [Model A missing] Model C ✅
No Reg:            Model B ✅        Model D ✅
"""
import torch
import numpy as np
from test_bar_aware_lstm import test_on_audio_file

print("="*100)
print(" "*25 + "COMPLETE MODEL COMPARISON ON EKTAAL FILE")
print("="*100)

ektaal_file = "/Users/shreyagupta/Desktop/AI_Research_Data/Tabla_files/Ektaal.wav"

# Model B: Original data + no regularization
print("\n" + "="*100)
print("MODEL B: No Regularization (Original Data)")
print("Val Loss: 3.0094")
print("="*100)
result_b = test_on_audio_file(
    audio_file=ektaal_file,
    lstm_model_path='models/best_bar_aware_lstm_no_reg.pth',
    num_generate=32,
    temperature=1.0
)

# Model C: Corrected data + regularization
print("\n" + "="*100)
print("MODEL C: With Regularization (Corrected Data - 69 'Ti Re Ki T' fixes)")
print("Val Loss: 3.0360")
print("="*100)
result_c = test_on_audio_file(
    audio_file=ektaal_file,
    lstm_model_path='models/best_bar_aware_lstm_corrected.pth',
    num_generate=32,
    temperature=1.0
)

# Model D: Corrected data + no regularization
print("\n" + "="*100)
print("MODEL D: No Regularization (Corrected Data - 69 'Ti Re Ki T' fixes)")
print("Val Loss: 3.0375")
print("="*100)
result_d = test_on_audio_file(
    audio_file=ektaal_file,
    lstm_model_path='models/best_bar_aware_lstm_corrected_no_reg.pth',
    num_generate=32,
    temperature=1.0
)

# Final comparison summary
print("\n" + "="*100)
print(" "*30 + "FINAL COMPARISON SUMMARY")
print("="*100)

print("\n" + "="*100)
print("MODEL MATRIX:")
print("="*100)
print(f"                   Original Data      Corrected Data")
print(f"With Reg:          [Not available]    Model C (val_loss: 3.0360)")
print(f"No Reg:            Model B (val_loss: 3.0094)   Model D (val_loss: 3.0375)")
print("="*100)

print(f"\n📊 Model B (Original + No Reg):")
print(f"   Val loss: 3.0094")
print(f"   Max repeats: {result_b['max_consecutive_repeats']}")
print(f"   Unique ratio: {result_b['unique_note_ratio']:.1%}")
print(f"   Generated: {' '.join(result_b['generated_notes'])}")

print(f"\n📊 Model C (Corrected + Reg):")
print(f"   Val loss: 3.0360")
print(f"   Max repeats: {result_c['max_consecutive_repeats']}")
print(f"   Unique ratio: {result_c['unique_note_ratio']:.1%}")
print(f"   Generated: {' '.join(result_c['generated_notes'])}")

print(f"\n📊 Model D (Corrected + No Reg):")
print(f"   Val loss: 3.0375")
print(f"   Max repeats: {result_d['max_consecutive_repeats']}")
print(f"   Unique ratio: {result_d['unique_note_ratio']:.1%}")
print(f"   Generated: {' '.join(result_d['generated_notes'])}")

print("\n" + "="*100)
print("ANALYSIS:")
print("="*100)

# Effect of regularization on corrected data (C vs D)
print(f"\nEffect of Regularization (Corrected Data):")
print(f"   With Reg (C):    Val Loss: 3.0360 | Repeats: {result_c['max_consecutive_repeats']} | Unique: {result_c['unique_note_ratio']:.1%}")
print(f"   No Reg (D):      Val Loss: 3.0375 | Repeats: {result_d['max_consecutive_repeats']} | Unique: {result_d['unique_note_ratio']:.1%}")
if result_c['max_consecutive_repeats'] < result_d['max_consecutive_repeats']:
    print(f"   → Regularization REDUCED max repeats")
elif result_c['max_consecutive_repeats'] > result_d['max_consecutive_repeats']:
    print(f"   → Regularization INCREASED max repeats")
else:
    print(f"   → Regularization had NO EFFECT on max repeats")

# Effect of data correction (B vs D - both no reg)
print(f"\nEffect of Data Correction (No Regularization):")
print(f"   Original (B):    Val Loss: 3.0094 | Repeats: {result_b['max_consecutive_repeats']} | Unique: {result_b['unique_note_ratio']:.1%}")
print(f"   Corrected (D):   Val Loss: 3.0375 | Repeats: {result_d['max_consecutive_repeats']} | Unique: {result_d['unique_note_ratio']:.1%}")
if result_b['max_consecutive_repeats'] < result_d['max_consecutive_repeats']:
    print(f"   → Correction INCREASED max repeats (worse)")
elif result_b['max_consecutive_repeats'] > result_d['max_consecutive_repeats']:
    print(f"   → Correction REDUCED max repeats (better)")
else:
    print(f"   → Correction had NO EFFECT on max repeats")

# Best overall
best_repeats = min(result_b['max_consecutive_repeats'], result_c['max_consecutive_repeats'], result_d['max_consecutive_repeats'])
best_unique = max(result_b['unique_note_ratio'], result_c['unique_note_ratio'], result_d['unique_note_ratio'])

print(f"\nBest Metrics:")
models = {
    'B': {'repeats': result_b['max_consecutive_repeats'], 'unique': result_b['unique_note_ratio']},
    'C': {'repeats': result_c['max_consecutive_repeats'], 'unique': result_c['unique_note_ratio']},
    'D': {'repeats': result_d['max_consecutive_repeats'], 'unique': result_d['unique_note_ratio']}
}
for name, metrics in models.items():
    if metrics['repeats'] == best_repeats:
        print(f"   Lowest repeats: Model {name} ({metrics['repeats']})")
    if metrics['unique'] == best_unique:
        print(f"   Highest diversity: Model {name} ({metrics['unique']:.1%})")

print("\n" + "="*100)
print("✅ COMPARISON COMPLETE!")
print("="*100)
