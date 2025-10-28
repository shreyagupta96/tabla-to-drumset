"""
Phase 4: Meter-Conditional LSTM Architecture
Bar-aware tabla generation with taal embedding and duration prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MeterConditionalLSTM(nn.Module):
    """
    Meter-conditional LSTM for bar-aware tabla generation

    Architecture:
    - Note embedding (12 classes)
    - Duration embedding (5 bins)
    - Taal embedding (2 taals: Teental=0, Ektaal=1)
    - 2-layer LSTM with variable-length sequences
    - Separate prediction heads for notes and durations
    """

    def __init__(
        self,
        vocab_size=12,
        num_duration_bins=5,
        num_taals=2,
        note_embed_dim=64,
        duration_embed_dim=16,
        taal_embed_dim=32,
        hidden_size=256,
        num_layers=2,
        dropout=0.3
    ):
        super(MeterConditionalLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.num_duration_bins = num_duration_bins
        self.num_taals = num_taals
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embeddings
        self.note_embedding = nn.Embedding(vocab_size, note_embed_dim)
        self.duration_embedding = nn.Embedding(num_duration_bins, duration_embed_dim)
        self.taal_embedding = nn.Embedding(num_taals, taal_embed_dim)

        # Total input dimension: note + duration + taal
        input_dim = note_embed_dim + duration_embed_dim + taal_embed_dim

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Prediction heads
        self.note_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, vocab_size)
        )

        self.duration_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_duration_bins)
        )

    def forward(self, context_notes, context_durations, taal_ids, lengths):
        """
        Forward pass with variable-length sequences

        Args:
            context_notes: (batch, max_seq_len) - note indices
            context_durations: (batch, max_seq_len) - duration bin indices
            taal_ids: (batch,) - taal indices (0=Teental, 1=Ektaal)
            lengths: (batch,) - actual sequence lengths (before padding)

        Returns:
            note_logits: (batch, max_seq_len, vocab_size)
            duration_logits: (batch, max_seq_len, num_duration_bins)
        """
        batch_size, max_seq_len = context_notes.size()

        # Embed inputs
        note_embeds = self.note_embedding(context_notes)  # (batch, seq, note_embed_dim)
        dur_embeds = self.duration_embedding(context_durations)  # (batch, seq, dur_embed_dim)

        # Expand taal embedding to match sequence length
        taal_embeds = self.taal_embedding(taal_ids)  # (batch, taal_embed_dim)
        taal_embeds = taal_embeds.unsqueeze(1).expand(-1, max_seq_len, -1)  # (batch, seq, taal_embed_dim)

        # Concatenate all embeddings
        combined = torch.cat([note_embeds, dur_embeds, taal_embeds], dim=-1)  # (batch, seq, input_dim)

        # Pack for variable-length sequences
        packed = pack_padded_sequence(
            combined,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM forward
        packed_output, (hidden, cell) = self.lstm(packed)

        # Unpack
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)  # (batch, seq, hidden)

        # Prediction heads
        note_logits = self.note_head(lstm_out)  # (batch, seq, vocab_size)
        duration_logits = self.duration_head(lstm_out)  # (batch, seq, num_duration_bins)

        return note_logits, duration_logits

    def generate(
        self,
        seed_notes,
        seed_durations,
        taal_id,
        num_generate=32,
        temperature=1.0,
        device='cpu'
    ):
        """
        Generate new tabla sequence given seed context

        Args:
            seed_notes: (seq_len,) - seed note indices
            seed_durations: (seq_len,) - seed duration bin indices
            taal_id: int - taal type (0=Teental, 1=Ektaal)
            num_generate: int - number of notes to generate
            temperature: float - sampling temperature (higher = more random)
            device: torch device

        Returns:
            generated_notes: list of note indices
            generated_durations: list of duration bin indices
        """
        self.eval()

        # Convert to tensors
        context_notes = torch.tensor(seed_notes, dtype=torch.long, device=device).unsqueeze(0)
        context_durations = torch.tensor(seed_durations, dtype=torch.long, device=device).unsqueeze(0)
        taal_tensor = torch.tensor([taal_id], dtype=torch.long, device=device)

        generated_notes = []
        generated_durations = []

        with torch.no_grad():
            for _ in range(num_generate):
                # Get current sequence length
                seq_len = context_notes.size(1)
                lengths = torch.tensor([seq_len], dtype=torch.long)

                # Forward pass
                note_logits, duration_logits = self.forward(
                    context_notes,
                    context_durations,
                    taal_tensor,
                    lengths
                )

                # Get last timestep predictions
                note_logits_last = note_logits[0, -1, :] / temperature
                duration_logits_last = duration_logits[0, -1, :] / temperature

                # Sample from distributions
                note_probs = F.softmax(note_logits_last, dim=-1)
                duration_probs = F.softmax(duration_logits_last, dim=-1)

                next_note = torch.multinomial(note_probs, 1).item()
                next_duration = torch.multinomial(duration_probs, 1).item()

                generated_notes.append(next_note)
                generated_durations.append(next_duration)

                # Append to context
                context_notes = torch.cat([
                    context_notes,
                    torch.tensor([[next_note]], dtype=torch.long, device=device)
                ], dim=1)
                context_durations = torch.cat([
                    context_durations,
                    torch.tensor([[next_duration]], dtype=torch.long, device=device)
                ], dim=1)

        return generated_notes, generated_durations


class EmbeddingRegularization(nn.Module):
    """
    Regularization to encourage similar embeddings for similar notes
    (Na/Ta, Tin/Tun, Ki/Kat)
    """

    def __init__(self, note_labels):
        super(EmbeddingRegularization, self).__init__()

        # Define similar note pairs
        self.similar_pairs = []

        # Create index mapping
        note_to_idx = {note: idx for idx, note in enumerate(note_labels)}

        # Add pairs if both notes exist in vocabulary
        pairs = [('Na', 'Ta'), ('Tin', 'Tun'), ('Ki', 'Kat')]
        for note1, note2 in pairs:
            if note1 in note_to_idx and note2 in note_to_idx:
                self.similar_pairs.append((note_to_idx[note1], note_to_idx[note2]))

    def forward(self, embedding_matrix):
        """
        Compute L2 distance loss between similar note embeddings

        Args:
            embedding_matrix: (vocab_size, embed_dim) - note embedding weights

        Returns:
            reg_loss: scalar tensor
        """
        if len(self.similar_pairs) == 0:
            return torch.tensor(0.0, device=embedding_matrix.device)

        total_loss = 0.0
        for idx1, idx2 in self.similar_pairs:
            embed1 = embedding_matrix[idx1]
            embed2 = embedding_matrix[idx2]
            total_loss += torch.dist(embed1, embed2, p=2)

        return total_loss / len(self.similar_pairs)


def create_model(
    vocab_size=12,
    num_duration_bins=5,
    num_taals=2,
    hidden_size=256,
    num_layers=2,
    dropout=0.3,
    note_labels=None
):
    """
    Factory function to create model with optional regularization

    Args:
        vocab_size: number of note classes
        num_duration_bins: number of duration bins
        num_taals: number of taal types
        hidden_size: LSTM hidden size
        num_layers: number of LSTM layers
        dropout: dropout rate
        note_labels: list of note names for embedding regularization

    Returns:
        model: MeterConditionalLSTM
        regularizer: EmbeddingRegularization or None
    """
    model = MeterConditionalLSTM(
        vocab_size=vocab_size,
        num_duration_bins=num_duration_bins,
        num_taals=num_taals,
        note_embed_dim=64,
        duration_embed_dim=16,
        taal_embed_dim=32,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )

    regularizer = None
    if note_labels is not None:
        regularizer = EmbeddingRegularization(note_labels)

    return model, regularizer


if __name__ == "__main__":
    # Test model
    print("Testing Meter-Conditional LSTM...")

    note_labels = ["Dha", "Dhin", "Ghe", "Kat", "Ki", "Na", "Re", "T", "Ta", "Ti", "Tun", "Tin"]

    model, regularizer = create_model(
        vocab_size=12,
        num_duration_bins=5,
        num_taals=2,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        note_labels=note_labels
    )

    print(f"\n✅ Model created:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Note embedding: {model.note_embedding.weight.shape}")
    print(f"   Duration embedding: {model.duration_embedding.weight.shape}")
    print(f"   Taal embedding: {model.taal_embedding.weight.shape}")
    print(f"   LSTM hidden size: {model.hidden_size}")
    print(f"   LSTM layers: {model.num_layers}")

    # Test forward pass
    batch_size = 4
    seq_lens = [30, 35, 40, 45]
    max_len = max(seq_lens)

    # Create dummy batch with padding
    context_notes = torch.randint(0, 12, (batch_size, max_len))
    context_durations = torch.randint(0, 5, (batch_size, max_len))
    taal_ids = torch.tensor([0, 1, 0, 1])  # Mix of Teental and Ektaal
    lengths = torch.tensor(seq_lens)

    # Forward
    note_logits, duration_logits = model(context_notes, context_durations, taal_ids, lengths)

    print(f"\n✅ Forward pass successful:")
    print(f"   Note logits: {note_logits.shape}")
    print(f"   Duration logits: {duration_logits.shape}")

    # Test generation
    seed_notes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3]  # 16 notes (1 Teental bar)
    seed_durations = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    gen_notes, gen_durs = model.generate(
        seed_notes=seed_notes,
        seed_durations=seed_durations,
        taal_id=0,  # Teental
        num_generate=16,
        temperature=1.0,
        device='cpu'
    )

    print(f"\n✅ Generation successful:")
    print(f"   Generated {len(gen_notes)} notes")
    print(f"   First 10 notes: {[note_labels[idx] for idx in gen_notes[:10]]}")

    # Test embedding regularization
    if regularizer is not None:
        reg_loss = regularizer(model.note_embedding.weight)
        print(f"\n✅ Embedding regularization:")
        print(f"   Similar pairs: {len(regularizer.similar_pairs)}")
        print(f"   Regularization loss: {reg_loss.item():.4f}")

    print("\n" + "="*80)
    print("✅ Model test complete! Ready for training.")
    print("="*80)
