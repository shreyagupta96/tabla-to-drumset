"""
LSTM Model for Tabla Note Generation
Predicts next note and duration given a sequence
"""

import torch
import torch.nn as nn

class TablaLSTM(nn.Module):
    """
    LSTM model for tabla sequence generation

    Predicts both the next note and its duration
    """

    def __init__(self, vocab_size, duration_bins=5, embedding_dim=32,
                 hidden_dim=128, num_layers=2, dropout=0.3):
        """
        Args:
            vocab_size: Number of unique notes in vocabulary
            duration_bins: Number of duration quantization bins
            embedding_dim: Dimension of note embeddings
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(TablaLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.duration_bins = duration_bins
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Store note mapping for embedding similarity loss
        self.note_to_idx = None  # Will be set during training

        # Note embedding layer
        self.note_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Duration embedding layer
        self.duration_embedding = nn.Embedding(duration_bins, embedding_dim // 2)

        # Combine note + duration embeddings
        combined_dim = embedding_dim + embedding_dim // 2

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output heads
        self.note_head = nn.Linear(hidden_dim, vocab_size)
        self.duration_head = nn.Linear(hidden_dim, duration_bins)

        self.dropout = nn.Dropout(dropout)

    def forward(self, note_seq, duration_seq, hidden=None):
        """
        Forward pass

        Args:
            note_seq: (batch_size, seq_len) note indices
            duration_seq: (batch_size, seq_len) duration bin indices
            hidden: Optional hidden state

        Returns:
            note_logits: (batch_size, vocab_size) note predictions
            duration_logits: (batch_size, duration_bins) duration predictions
            hidden: Updated hidden state
        """
        # Embed notes and durations
        note_emb = self.note_embedding(note_seq)  # (batch, seq_len, emb_dim)
        dur_emb = self.duration_embedding(duration_seq)  # (batch, seq_len, emb_dim//2)

        # Concatenate embeddings
        combined = torch.cat([note_emb, dur_emb], dim=-1)  # (batch, seq_len, combined_dim)

        # LSTM forward
        lstm_out, hidden = self.lstm(combined, hidden)

        # Take last timestep
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim)

        # Apply dropout
        last_output = self.dropout(last_output)

        # Predict note and duration
        note_logits = self.note_head(last_output)
        duration_logits = self.duration_head(last_output)

        return note_logits, duration_logits, hidden

    def generate(self, seed_notes, seed_durations, num_generate=10,
                 temperature=1.0, device='cpu'):
        """
        Generate new tabla sequence

        Args:
            seed_notes: Initial note sequence (list or tensor)
            seed_durations: Initial duration sequence (list or tensor)
            num_generate: Number of notes to generate
            temperature: Sampling temperature (higher = more random)
            device: torch device

        Returns:
            generated_notes: List of generated note indices
            generated_durations: List of generated duration indices
        """
        self.eval()

        # Convert to tensor if needed
        if not isinstance(seed_notes, torch.Tensor):
            seed_notes = torch.tensor(seed_notes, dtype=torch.long).unsqueeze(0)
        if not isinstance(seed_durations, torch.Tensor):
            seed_durations = torch.tensor(seed_durations, dtype=torch.long).unsqueeze(0)

        seed_notes = seed_notes.to(device)
        seed_durations = seed_durations.to(device)

        generated_notes = seed_notes[0].cpu().tolist()
        generated_durations = seed_durations[0].cpu().tolist()

        hidden = None

        with torch.no_grad():
            for _ in range(num_generate):
                # Get current sequence (last 8 notes)
                current_notes = torch.tensor([generated_notes[-8:]], dtype=torch.long, device=device)
                current_durs = torch.tensor([generated_durations[-8:]], dtype=torch.long, device=device)

                # Predict next
                note_logits, dur_logits, hidden = self.forward(current_notes, current_durs, hidden)

                # Sample with temperature
                note_probs = torch.softmax(note_logits / temperature, dim=-1)
                dur_probs = torch.softmax(dur_logits / temperature, dim=-1)

                # Sample from distribution
                next_note = torch.multinomial(note_probs, 1).item()
                next_dur = torch.multinomial(dur_probs, 1).item()

                generated_notes.append(next_note)
                generated_durations.append(next_dur)

        # Return only the generated part (excluding seed)
        return generated_notes[len(seed_notes[0]):], generated_durations[len(seed_durations[0]):]

    def set_note_mapping(self, note_to_idx):
        """
        Store note-to-index mapping for similarity loss computation

        Args:
            note_to_idx: Dictionary mapping note names to indices
        """
        self.note_to_idx = note_to_idx

    def embedding_similarity_loss(self):
        """
        Compute L2 distance between embeddings of acoustically similar note pairs

        This encourages the model to learn that similar-sounding notes (Na/Ta, Tin/Tun, Ki/Kat)
        should have similar internal representations, leading to more musically natural generation.

        Returns:
            Similarity loss (scalar tensor)
        """
        if self.note_to_idx is None:
            return torch.tensor(0.0, device=self.note_embedding.weight.device)

        # Define acoustically similar pairs (excluding Ti/Re which is handled by pattern rules)
        similar_pairs = [
            ('Na', 'Ta'),    # Both open ringing strokes
            ('Tin', 'Tun'),  # Both open mid-range strokes
            ('Ki', 'Kat'),   # Both crisp closed strokes
        ]

        loss = torch.tensor(0.0, device=self.note_embedding.weight.device)
        num_pairs = 0

        for note1, note2 in similar_pairs:
            # Check both notes exist in vocabulary
            if note1 in self.note_to_idx and note2 in self.note_to_idx:
                idx1 = self.note_to_idx[note1]
                idx2 = self.note_to_idx[note2]

                # Get embedding vectors
                emb1 = self.note_embedding.weight[idx1]
                emb2 = self.note_embedding.weight[idx2]

                # L2 distance (Euclidean) - we want this to be small
                pair_distance = torch.norm(emb1 - emb2, p=2)
                loss += pair_distance
                num_pairs += 1

        # Average over pairs
        if num_pairs > 0:
            loss = loss / num_pairs

        return loss

def create_model(vocab_size=12, duration_bins=5):
    """
    Create TablaLSTM model with default hyperparameters
    """
    model = TablaLSTM(
        vocab_size=vocab_size,
        duration_bins=duration_bins,
        embedding_dim=32,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    )

    return model

if __name__ == "__main__":
    # Test model creation
    model = create_model()

    print("="*80)
    print("TABLA LSTM MODEL")
    print("="*80)
    print(f"\nModel architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    seq_len = 8
    test_notes = torch.randint(0, 12, (batch_size, seq_len))
    test_durs = torch.randint(0, 5, (batch_size, seq_len))

    note_out, dur_out, _ = model(test_notes, test_durs)

    print(f"\n✅ Forward pass successful!")
    print(f"   Input shape: ({batch_size}, {seq_len})")
    print(f"   Note output shape: {note_out.shape}")
    print(f"   Duration output shape: {dur_out.shape}")
    print("="*80)
