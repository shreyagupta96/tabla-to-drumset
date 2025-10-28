"""
Diagnose the actual durations being passed to tabla and drum synthesis
to understand why drums still show less swing variation
"""

import os
import numpy as np
import torch
import sys

sys.path.insert(0, "/Users/shreyagupta/Desktop/Research MFP/B264026_Masters_Final_Project_SG/Submission/Code/HybridApproach")

from export_drum_mapping import generate_with_swing

print("="*80)
print("DIAGNOSING DURATION VARIATION")
print("="*80)

ektaal_file = "/Users/shreyagupta/Desktop/AI_Research_Data/Tabla_files/Ektaal.wav"
model_path = "models/best_bar_aware_lstm.pth"

# Generate with swing
context_notes, context_durations, gen_notes, gen_durations, swing_stats = generate_with_swing(
    ektaal_file,
    model_path,
    num_generate=32,
    temperature=1.0
)

print("\n" + "="*80)
print("CONTEXT (Tabla) DURATIONS")
print("="*80)
print(f"Number of notes: {len(context_durations)}")
print(f"Durations (first 20): {[f'{d:.3f}' for d in context_durations[:20]]}")
print(f"\nStatistics:")
print(f"  Mean: {np.mean(context_durations):.3f}s")
print(f"  Std:  {np.std(context_durations):.3f}s")
print(f"  CV:   {np.std(context_durations) / np.mean(context_durations):.3f}")
print(f"  Min:  {np.min(context_durations):.3f}s")
print(f"  Max:  {np.max(context_durations):.3f}s")

print("\n" + "="*80)
print("GENERATION (Drums) DURATIONS")
print("="*80)
print(f"Number of notes: {len(gen_durations)}")
print(f"Durations (first 20): {[f'{d:.3f}' for d in gen_durations[:20]]}")
print(f"\nStatistics:")
print(f"  Mean: {np.mean(gen_durations):.3f}s")
print(f"  Std:  {np.std(gen_durations):.3f}s")
print(f"  CV:   {np.std(gen_durations) / np.mean(gen_durations):.3f}")
print(f"  Min:  {np.min(gen_durations):.3f}s")
print(f"  Max:  {np.max(gen_durations):.3f}s")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"Context CV:    {np.std(context_durations) / np.mean(context_durations):.3f}")
print(f"Generation CV: {np.std(gen_durations) / np.mean(gen_durations):.3f}")
print(f"\nContext has {len(context_durations)} notes with mean duration {np.mean(context_durations):.3f}s")
print(f"Generation has {len(gen_durations)} notes with mean duration {np.mean(gen_durations):.3f}s")

if np.mean(gen_durations) > np.mean(context_durations):
    print(f"\n⚠️  Generated notes are LONGER on average ({np.mean(gen_durations):.3f}s vs {np.mean(context_durations):.3f}s)")
    print(f"This means fewer notes per second, which could reduce perceived swing variation")

print("\n" + "="*80)
print("SWING TEMPLATE STATISTICS")
print("="*80)
for i in range(5):
    cluster_key = f'bin_{i}'
    cluster_size = swing_stats['cluster_sizes'][cluster_key]
    cluster_mean = swing_stats['cluster_means'][cluster_key]
    print(f"Duration bin {i}: {cluster_size} samples, mean IOI: {cluster_mean:.3f}s")
