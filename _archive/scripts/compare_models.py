"""
Compare all 3 models (A, B, C) on the same Ektaal file
"""
import torch
import numpy as np
from test_bar_aware_lstm import test_on_audio_file

print("="*100)
print(" "*30 + "MODEL COMPARISON ON EKTAAL FILE")
print("="*100)

ektaal_file = "/Users/shreyagupta/Desktop/AI_Research_Data/Tabla_files/Ektaal.wav"

# Model A: Original data + regularization
print("\n" + "="*100)
print("MODEL A: With Regularization (Original Data)")
print("="*100)
result_a = test_on_audio_file(
    audio_file=ektaal_file,
    lstm_model_path='models/best_bar_aware_lstm.pth',
    num_generate=32,
    temperature=1.0
)

# Model B: Original data + no regularization
print("\n" + "="*100)
print("MODEL B: No Regularization (Original Data)")
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
print("="*100)
result_c = test_on_audio_file(
    audio_file=ektaal_file,
    lstm_model_path='models/best_bar_aware_lstm_corrected.pth',
    num_generate=32,
    temperature=1.0
)

# Final comparison summary
print("\n" + "="*100)
print(" "*30 + "FINAL COMPARISON SUMMARY")
print("="*100)

print(f"\n📊 Model A (Original + Reg):")
print(f"   Max repeats: {result_a['max_consecutive_repeats']}")
print(f"   Unique ratio: {result_a['unique_note_ratio']:.1%}")
print(f"   Generated: {' '.join(result_a['generated_notes'])}")

print(f"\n📊 Model B (Original + No Reg):")
print(f"   Max repeats: {result_b['max_consecutive_repeats']}")
print(f"   Unique ratio: {result_b['unique_note_ratio']:.1%}")
print(f"   Generated: {' '.join(result_b['generated_notes'])}")

print(f"\n📊 Model C (Corrected + Reg):")
print(f"   Max repeats: {result_c['max_consecutive_repeats']}")
print(f"   Unique ratio: {result_c['unique_note_ratio']:.1%}")
print(f"   Generated: {' '.join(result_c['generated_notes'])}")

print("\n" + "="*100)
print("✅ COMPARISON COMPLETE!")
print("="*100)
