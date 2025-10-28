"""
Phase 5: Train Bar-Aware LSTM
Training script for meter-conditional tabla generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
import numpy as np
from collections import Counter
import time

from meter_conditional_lstm import create_model, EmbeddingRegularization

class BarAwareDataset(Dataset):
    """Dataset for bar-aware training sequences"""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'context_notes': torch.tensor(seq['context_notes'], dtype=torch.long),
            'context_durations': torch.tensor(seq['context_durations'], dtype=torch.long),
            'target_notes': torch.tensor(seq['target_notes'], dtype=torch.long),
            'target_durations': torch.tensor(seq['target_durations'], dtype=torch.long),
            'taal_id': torch.tensor(seq['taal_id'], dtype=torch.long),
            'context_length': len(seq['context_notes']),
            'target_length': len(seq['target_notes'])
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences
    Pads sequences to max length in batch
    """
    # Separate fields
    context_notes = [item['context_notes'] for item in batch]
    context_durations = [item['context_durations'] for item in batch]
    target_notes = [item['target_notes'] for item in batch]
    target_durations = [item['target_durations'] for item in batch]
    taal_ids = torch.stack([item['taal_id'] for item in batch])
    context_lengths = torch.tensor([item['context_length'] for item in batch])
    target_lengths = torch.tensor([item['target_length'] for item in batch])

    # Pad sequences
    context_notes_padded = pad_sequence(context_notes, batch_first=True, padding_value=0)
    context_durations_padded = pad_sequence(context_durations, batch_first=True, padding_value=0)
    target_notes_padded = pad_sequence(target_notes, batch_first=True, padding_value=0)
    target_durations_padded = pad_sequence(target_durations, batch_first=True, padding_value=0)

    return {
        'context_notes': context_notes_padded,
        'context_durations': context_durations_padded,
        'target_notes': target_notes_padded,
        'target_durations': target_durations_padded,
        'taal_ids': taal_ids,
        'context_lengths': context_lengths,
        'target_lengths': target_lengths
    }


def compute_looping_metrics(predictions, targets, lengths):
    """
    Compute looping metrics for generated sequences

    Metrics:
    - Max consecutive repeats: longest run of same note
    - Unique note ratio: unique notes / total notes
    """
    max_repeats_list = []
    unique_ratios = []

    for pred, target, length in zip(predictions, targets, lengths):
        # Only consider actual sequence (not padding)
        pred_seq = pred[:length].cpu().numpy()

        # Max consecutive repeats
        if len(pred_seq) > 0:
            max_repeat = 1
            current_repeat = 1
            for i in range(1, len(pred_seq)):
                if pred_seq[i] == pred_seq[i-1]:
                    current_repeat += 1
                    max_repeat = max(max_repeat, current_repeat)
                else:
                    current_repeat = 1
            max_repeats_list.append(max_repeat)

            # Unique ratio
            unique_ratio = len(set(pred_seq)) / len(pred_seq)
            unique_ratios.append(unique_ratio)

    return {
        'max_consecutive_repeats': np.mean(max_repeats_list) if max_repeats_list else 0,
        'unique_note_ratio': np.mean(unique_ratios) if unique_ratios else 0
    }


def train_epoch(model, dataloader, optimizer, criterion_note, criterion_duration, regularizer, reg_weight, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_note_loss = 0
    total_dur_loss = 0
    total_reg_loss = 0

    for batch in dataloader:
        # Move to device
        context_notes = batch['context_notes'].to(device)
        context_durations = batch['context_durations'].to(device)
        target_notes = batch['target_notes'].to(device)
        target_durations = batch['target_durations'].to(device)
        taal_ids = batch['taal_ids'].to(device)
        context_lengths = batch['context_lengths']
        target_lengths = batch['target_lengths']

        # Forward pass
        note_logits, duration_logits = model(
            context_notes,
            context_durations,
            taal_ids,
            context_lengths
        )

        # We predict the target from the context
        # Target is shifted by context_length, so we need to align predictions
        # For simplicity, we'll use the last len(target) predictions from the context
        batch_size = note_logits.size(0)

        # Compute losses only on actual target positions
        note_loss = 0
        dur_loss = 0

        for i in range(batch_size):
            target_len = target_lengths[i].item()

            # Use last target_len predictions from context
            pred_notes = note_logits[i, -target_len:, :]
            pred_durs = duration_logits[i, -target_len:, :]

            # Actual targets
            gt_notes = target_notes[i, :target_len]
            gt_durs = target_durations[i, :target_len]

            # Cross-entropy loss
            note_loss += criterion_note(pred_notes, gt_notes)
            dur_loss += criterion_duration(pred_durs, gt_durs)

        note_loss = note_loss / batch_size
        dur_loss = dur_loss / batch_size

        # Embedding regularization
        reg_loss = torch.tensor(0.0, device=device)
        if regularizer is not None:
            reg_loss = regularizer(model.note_embedding.weight)

        # Total loss
        loss = note_loss + dur_loss + reg_weight * reg_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_note_loss += note_loss.item()
        total_dur_loss += dur_loss.item()
        total_reg_loss += reg_loss.item()

    n_batches = len(dataloader)
    return {
        'total_loss': total_loss / n_batches,
        'note_loss': total_note_loss / n_batches,
        'duration_loss': total_dur_loss / n_batches,
        'reg_loss': total_reg_loss / n_batches
    }


def validate(model, dataloader, criterion_note, criterion_duration, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_note_loss = 0
    total_dur_loss = 0

    all_note_preds = []
    all_note_targets = []
    all_target_lengths = []

    with torch.no_grad():
        for batch in dataloader:
            context_notes = batch['context_notes'].to(device)
            context_durations = batch['context_durations'].to(device)
            target_notes = batch['target_notes'].to(device)
            target_durations = batch['target_durations'].to(device)
            taal_ids = batch['taal_ids'].to(device)
            context_lengths = batch['context_lengths']
            target_lengths = batch['target_lengths']

            # Forward pass
            note_logits, duration_logits = model(
                context_notes,
                context_durations,
                taal_ids,
                context_lengths
            )

            # Compute losses
            batch_size = note_logits.size(0)
            note_loss = 0
            dur_loss = 0

            for i in range(batch_size):
                target_len = target_lengths[i].item()

                pred_notes = note_logits[i, -target_len:, :]
                pred_durs = duration_logits[i, -target_len:, :]

                gt_notes = target_notes[i, :target_len]
                gt_durs = target_durations[i, :target_len]

                note_loss += criterion_note(pred_notes, gt_notes)
                dur_loss += criterion_duration(pred_durs, gt_durs)

                # Collect predictions for metrics
                pred_note_indices = torch.argmax(pred_notes, dim=-1)
                all_note_preds.append(pred_note_indices)
                all_note_targets.append(gt_notes)
                all_target_lengths.append(target_len)

            note_loss = note_loss / batch_size
            dur_loss = dur_loss / batch_size
            loss = note_loss + dur_loss

            total_loss += loss.item()
            total_note_loss += note_loss.item()
            total_dur_loss += dur_loss.item()

    n_batches = len(dataloader)

    # Compute looping metrics
    looping_metrics = compute_looping_metrics(all_note_preds, all_note_targets, all_target_lengths)

    return {
        'total_loss': total_loss / n_batches,
        'note_loss': total_note_loss / n_batches,
        'duration_loss': total_dur_loss / n_batches,
        'max_consecutive_repeats': looping_metrics['max_consecutive_repeats'],
        'unique_note_ratio': looping_metrics['unique_note_ratio']
    }


def train(
    dataset_path='training_data/bar_aware_dataset.pkl',
    model_save_path='models',
    num_epochs=100,
    batch_size=16,
    learning_rate=0.001,
    reg_weight=0.1,
    hidden_size=256,
    num_layers=2,
    dropout=0.3,
    device=None
):
    """Main training loop"""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 100)
    print(" " * 30 + "PHASE 5: BAR-AWARE LSTM TRAINING")
    print("=" * 100)

    # Load dataset
    print(f"\n📂 Loading dataset from: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        dataset_dict = pickle.load(f)

    train_sequences = dataset_dict['train']
    val_sequences = dataset_dict['val']
    metadata = dataset_dict['metadata']

    print(f"   Training sequences: {len(train_sequences)}")
    print(f"   Validation sequences: {len(val_sequences)}")

    # Create datasets
    train_dataset = BarAwareDataset(train_sequences)
    val_dataset = BarAwareDataset(val_sequences)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"\n🔧 Configuration:")
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Regularization weight: {reg_weight}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Num layers: {num_layers}")
    print(f"   Dropout: {dropout}")
    print(f"   Epochs: {num_epochs}")

    # Create model
    print(f"\n🏗️  Creating model...")
    model, regularizer = create_model(
        vocab_size=metadata['num_classes'],
        num_duration_bins=metadata['num_duration_bins'],
        num_taals=len(metadata['taal_mapping']),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        note_labels=metadata['note_labels']
    )
    model = model.to(device)

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss functions and optimizer
    criterion_note = nn.CrossEntropyLoss()
    criterion_duration = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Training loop
    os.makedirs(model_save_path, exist_ok=True)
    best_val_loss = float('inf')

    print(f"\n{'=' * 100}")
    print(" " * 35 + "TRAINING PROGRESS")
    print(f"{'=' * 100}\n")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer,
            criterion_note, criterion_duration,
            regularizer, reg_weight, device
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion_note, criterion_duration, device)

        # Learning rate scheduling
        scheduler.step(val_metrics['total_loss'])

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch+1:3d}/{num_epochs} ({epoch_time:.1f}s) | "
              f"Train Loss: {train_metrics['total_loss']:.4f} "
              f"(N:{train_metrics['note_loss']:.4f} D:{train_metrics['duration_loss']:.4f} R:{train_metrics['reg_loss']:.4f}) | "
              f"Val Loss: {val_metrics['total_loss']:.4f} | "
              f"Repeats: {val_metrics['max_consecutive_repeats']:.2f} | "
              f"Unique: {val_metrics['unique_note_ratio']:.3f}")

        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'metadata': metadata,
                'hyperparameters': {
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'learning_rate': learning_rate,
                    'reg_weight': reg_weight
                }
            }
            torch.save(checkpoint, os.path.join(model_save_path, 'best_bar_aware_lstm.pth'))
            print(f"   ✅ Saved best model (val_loss: {val_metrics['total_loss']:.4f})")

    print(f"\n{'=' * 100}")
    print("✅ Training Complete!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Model saved to: {os.path.join(model_save_path, 'best_bar_aware_lstm.pth')}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    train(
        dataset_path='training_data/bar_aware_dataset.pkl',
        model_save_path='models',
        num_epochs=100,
        batch_size=16,
        learning_rate=0.001,
        reg_weight=0.1,
        hidden_size=256,
        num_layers=2,
        dropout=0.3
    )
