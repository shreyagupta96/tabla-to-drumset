"""
Train Bar-Aware LSTM with 4 Taals
Extends the 2-taal system to support:
- Teental (16 beats) → taal_id = 0
- Ektaal (12 beats) → taal_id = 1
- Jhaptaal (10 beats) → taal_id = 2
- Rupak (7 beats) → taal_id = 3
"""

import sys
import os

# Just import and run the existing training function with updated parameters
from train_bar_aware_lstm import train

if __name__ == "__main__":
    print("=" * 100)
    print(" " * 25 + "TRAINING 4-TAAL METER-CONDITIONAL LSTM")
    print("=" * 100)
    print("\nTaal mapping:")
    print("  0 = Teental (16 beats)")
    print("  1 = Ektaal (12 beats)")
    print("  2 = Jhaptaal (10 beats)")
    print("  3 = Rupak (7 beats)")
    print()

    train(
        dataset_path='training_data/bar_aware_dataset_4_taals.pkl',
        model_save_path='models',
        num_epochs=100,
        batch_size=16,
        learning_rate=0.001,
        reg_weight=0.1,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        device=None
    )
