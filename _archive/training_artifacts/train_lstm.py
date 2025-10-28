"""
Training script for Tabla LSTM model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from lstm_model import create_model
import time

class TablaDataset(Dataset):
    """PyTorch Dataset for tabla sequences"""

    def __init__(self, X_notes, X_durations, y_notes, y_durations):
        self.X_notes = torch.tensor(X_notes, dtype=torch.long)
        self.X_durations = torch.tensor(X_durations, dtype=torch.long)
        self.y_notes = torch.tensor(y_notes, dtype=torch.long)
        self.y_durations = torch.tensor(y_durations, dtype=torch.long)

    def __len__(self):
        return len(self.X_notes)

    def __getitem__(self, idx):
        return (
            self.X_notes[idx],
            self.X_durations[idx],
            self.y_notes[idx],
            self.y_durations[idx]
        )

def train_epoch(model, dataloader, criterion_note, criterion_dur, optimizer, device, lambda_similarity=0.1):
    """Train for one epoch with embedding similarity loss"""
    model.train()
    total_loss = 0
    total_note_loss = 0
    total_dur_loss = 0
    total_sim_loss = 0
    correct_notes = 0
    correct_durs = 0
    total = 0

    for batch in dataloader:
        X_notes, X_durs, y_notes, y_durs = [b.to(device) for b in batch]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        note_logits, dur_logits, _ = model(X_notes, X_durs)

        # Calculate prediction losses
        loss_note = criterion_note(note_logits, y_notes)
        loss_dur = criterion_dur(dur_logits, y_durs)

        # Calculate embedding similarity loss
        loss_similarity = model.embedding_similarity_loss()

        # Combined loss (prediction + embedding regularization)
        loss = loss_note + loss_dur + lambda_similarity * loss_similarity

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_note_loss += loss_note.item()
        total_dur_loss += loss_dur.item()
        total_sim_loss += loss_similarity.item()

        # Calculate accuracy
        _, predicted_notes = torch.max(note_logits, 1)
        _, predicted_durs = torch.max(dur_logits, 1)

        correct_notes += (predicted_notes == y_notes).sum().item()
        correct_durs += (predicted_durs == y_durs).sum().item()
        total += y_notes.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_note_loss = total_note_loss / len(dataloader)
    avg_dur_loss = total_dur_loss / len(dataloader)
    avg_sim_loss = total_sim_loss / len(dataloader)
    note_acc = 100 * correct_notes / total
    dur_acc = 100 * correct_durs / total

    return avg_loss, avg_note_loss, avg_dur_loss, avg_sim_loss, note_acc, dur_acc

def validate(model, dataloader, criterion_note, criterion_dur, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct_notes = 0
    correct_durs = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            X_notes, X_durs, y_notes, y_durs = [b.to(device) for b in batch]

            note_logits, dur_logits, _ = model(X_notes, X_durs)

            loss_note = criterion_note(note_logits, y_notes)
            loss_dur = criterion_dur(dur_logits, y_durs)
            loss = loss_note + loss_dur

            total_loss += loss.item()

            _, predicted_notes = torch.max(note_logits, 1)
            _, predicted_durs = torch.max(dur_logits, 1)

            correct_notes += (predicted_notes == y_notes).sum().item()
            correct_durs += (predicted_durs == y_durs).sum().item()
            total += y_notes.size(0)

    avg_loss = total_loss / len(dataloader)
    note_acc = 100 * correct_notes / total
    dur_acc = 100 * correct_durs / total

    return avg_loss, note_acc, dur_acc

def train_model(data_file, num_epochs=100, batch_size=32, learning_rate=0.001,
                val_split=0.2, save_path='tabla_lstm_model.pth'):
    """
    Main training function

    Args:
        data_file: Path to preprocessed data pickle file
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        val_split: Validation split fraction
        save_path: Path to save trained model
    """
    # Load data
    print("="*80)
    print("TRAINING TABLA LSTM MODEL")
    print("="*80)
    print(f"\n📁 Loading data from: {data_file}")

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    X_notes = data['X_notes']
    X_durations = data['X_durations']
    y_notes = data['y_notes']
    y_durations = data['y_durations']
    vocab_size = data['vocab_size']

    print(f"✅ Data loaded: {len(X_notes)} examples")

    # Train/val split
    n_val = int(len(X_notes) * val_split)
    n_train = len(X_notes) - n_val

    indices = np.random.permutation(len(X_notes))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Create datasets
    train_dataset = TablaDataset(
        X_notes[train_idx], X_durations[train_idx],
        y_notes[train_idx], y_durations[train_idx]
    )

    val_dataset = TablaDataset(
        X_notes[val_idx], X_durations[val_idx],
        y_notes[val_idx], y_durations[val_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"📊 Train examples: {n_train}")
    print(f"📊 Validation examples: {n_val}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")

    model = create_model(vocab_size=vocab_size)
    model = model.to(device)

    # Set note mapping for embedding similarity loss
    note_to_idx = data['note_to_idx']
    model.set_note_mapping(note_to_idx)
    print(f"✅ Note mapping set for embedding regularization")

    # Loss and optimizer
    criterion_note = nn.CrossEntropyLoss()
    criterion_dur = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n🎯 Training for {num_epochs} epochs...")
    print(f"📊 Batch size: {batch_size}")
    print(f"📊 Learning rate: {learning_rate}")
    print(f"🔗 Embedding regularization: λ = 0.1 (Na/Ta, Tin/Tun, Ki/Kat)")
    print("-"*80)

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train with embedding similarity loss (λ = 0.1)
        train_loss, train_note_loss, train_dur_loss, train_sim_loss, train_note_acc, train_dur_acc = train_epoch(
            model, train_loader, criterion_note, criterion_dur, optimizer, device, lambda_similarity=0.1
        )

        # Validate
        val_loss, val_note_acc, val_dur_acc = validate(
            model, val_loader, criterion_note, criterion_dur, device
        )

        epoch_time = time.time() - epoch_start

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Loss: {train_loss:.4f} (Sim: {train_sim_loss:.4f}) | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Note Acc: {train_note_acc:.1f}% / {val_note_acc:.1f}% | "
                  f"Dur Acc: {train_dur_acc:.1f}% / {val_dur_acc:.1f}% | "
                  f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab_size': vocab_size,
                'note_to_idx': data['note_to_idx'],
                'idx_to_note': data['idx_to_note']
            }, save_path)

    total_time = time.time() - start_time

    print("-"*80)
    print(f"\n✅ Training complete!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"💾 Best model saved to: {save_path}")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print("="*80)

    return model

if __name__ == "__main__":
    DATA_FILE = "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/lstm_training_data.pkl"
    MODEL_PATH = "/Users/shreyagupta/Desktop/AI_Research_Data/tabla-to-drumset_SG/tabla_lstm_model.pth"

    # Train model
    model = train_model(
        data_file=DATA_FILE,
        num_epochs=100,
        batch_size=32,
        learning_rate=0.001,
        val_split=0.2,
        save_path=MODEL_PATH
    )

    print("\n🎉 Model training complete!")
    print(f"📝 Next step: Test generation with different temperatures")
