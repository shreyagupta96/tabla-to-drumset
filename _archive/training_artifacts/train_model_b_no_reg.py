"""
Train Model B: Bar-Aware LSTM WITHOUT Embedding Regularization
For comparison with Model A (with regularization)
"""

from train_bar_aware_lstm import train

if __name__ == "__main__":
    print("\n" + "="*100)
    print(" " * 25 + "TRAINING MODEL B: NO EMBEDDING REGULARIZATION")
    print("="*100 + "\n")

    train(
        dataset_path='training_data/bar_aware_dataset.pkl',
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
    print("Model B saved as: models/best_bar_aware_lstm.pth")
    print("Please rename to: models/best_bar_aware_lstm_no_reg.pth")
    print("="*100 + "\n")
