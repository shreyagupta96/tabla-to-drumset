"""
Train Model D: Bar-Aware LSTM with Corrected Data WITHOUT Embedding Regularization
Completes the 2x2 comparison matrix:
  - Model A: Original + Reg
  - Model B: Original + No Reg
  - Model C: Corrected + Reg
  - Model D: Corrected + No Reg (THIS MODEL)
"""

from train_bar_aware_lstm import train

if __name__ == "__main__":
    print("\n" + "="*100)
    print(" " * 20 + "TRAINING MODEL D: CORRECTED DATA + NO REGULARIZATION")
    print("="*100 + "\n")

    train(
        dataset_path='training_data/bar_aware_dataset.pkl',  # Corrected dataset
        model_save_path='models',
        num_epochs=100,
        batch_size=16,
        learning_rate=0.001,
        reg_weight=0.0,  # NO REGULARIZATION
        hidden_size=256,
        num_layers=2,
        dropout=0.3
    )

    print("\n" + "="*100)
    print("Model D saved as: models/best_bar_aware_lstm.pth")
    print("Please rename to: models/best_bar_aware_lstm_corrected_no_reg.pth")
    print("="*100 + "\n")
