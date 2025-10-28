"""
Timbral-Based Constrained Generation
Rule: Any 4 consecutive closed/crisp strokes → enforce "Ti Re Ki T"
"""

import torch
import torch.nn.functional as F
from meter_conditional_lstm import create_model


class TimbalRuleEngine:
    """
    Timbral-based rule engine
    Enforces "Ti Re Ki T" for any 4 consecutive closed strokes
    """

    def __init__(self, note_labels):
        self.note_labels = note_labels
        self.note_to_idx = {note: idx for idx, note in enumerate(note_labels)}

        # Timbral categories
        self.closed_crisp = {'Ti', 'Re', 'Ki', 'T', 'Kat'}  # Closed/crisp strokes
        self.open_resonant = {'Dha', 'Dhin', 'Ghe', 'Tin', 'Tun', 'Ta', 'Na'}  # Open/resonant

        # Target pattern for 4 closed strokes
        self.target_pattern = ['Ti', 'Re', 'Ki', 'T']
        self.target_indices = [self.note_to_idx[note] for note in self.target_pattern]

    def is_closed(self, note_idx):
        """Check if a note index is a closed/crisp stroke"""
        note_name = self.note_labels[note_idx]
        return note_name in self.closed_crisp

    def check_timbral_rule(self, generated_sequence):
        """
        Check if we're in a 4-closed-stroke pattern starting with {Ti, Re, Ki, T}
        Returns: (triggered, position_in_pattern, next_required_note_idx)
        """
        if len(generated_sequence) < 1:
            return False, 0, None

        # Count consecutive closed strokes from the end
        closed_count = 0
        first_closed_idx = None
        for i in range(len(generated_sequence) - 1, -1, -1):
            if self.is_closed(generated_sequence[i]):
                closed_count += 1
                first_closed_idx = generated_sequence[i]  # Will be the earliest one
            else:
                break

        # Check if we have 1-3 consecutive closed strokes
        if 1 <= closed_count <= 3:
            # CRITICAL CHECK: Is the first closed stroke one of {Ti, Re, Ki, T}?
            first_closed_name = self.note_labels[first_closed_idx]

            if first_closed_name in {'Ti', 'Re', 'Ki', 'T'}:
                # YES! Enforce "Ti Re Ki T" pattern
                position = closed_count
                next_note_idx = self.target_indices[position]
                return True, position, next_note_idx
            else:
                # NO! First closed is Kat or other, don't enforce
                return False, 0, None

        return False, 0, None

    def apply_timbral_rule(self, generated_sequence, note_logits):
        """
        Apply timbral rule: 4 consecutive closed → Ti Re Ki T

        Returns: modified logits with rule enforced
        """
        constrained_logits = note_logits.clone()

        triggered, position, next_note_idx = self.check_timbral_rule(generated_sequence)

        if triggered:
            # HARD CONSTRAINT: Force next note
            mask = torch.ones_like(constrained_logits) * float('-inf')
            mask[next_note_idx] = 0
            constrained_logits = constrained_logits + mask

            print(f"   🎯 TIMBRAL RULE TRIGGERED!")
            print(f"      Detected {position} consecutive closed stroke(s)")
            print(f"      Pattern position: {position}/4")
            print(f"      Forcing next note: {self.note_labels[next_note_idx]} (completing 'Ti Re Ki T')")

        return constrained_logits, triggered


def generate_with_timbral_rules(
    model,
    seed_notes,
    seed_durations,
    taal_id,
    note_labels,
    num_generate=32,
    temperature=1.0,
    device='cpu'
):
    """
    Generate tabla sequence with timbral-based rules

    Args:
        model: Trained LSTM model
        seed_notes: Initial note sequence (indices)
        seed_durations: Initial duration sequence (indices)
        taal_id: Taal type (0=Teental, 1=Ektaal)
        note_labels: List of note names
        num_generate: Number of notes to generate
        temperature: Sampling temperature
        device: torch device

    Returns:
        generated_notes: List of note indices
        generated_durations: List of duration indices
        rule_applications: List of where rules were applied
    """

    model.eval()

    # Initialize rule engine
    rule_engine = TimbalRuleEngine(note_labels)

    # Convert to tensors
    context_notes = torch.tensor(seed_notes, dtype=torch.long, device=device).unsqueeze(0)
    context_durations = torch.tensor(seed_durations, dtype=torch.long, device=device).unsqueeze(0)
    taal_tensor = torch.tensor([taal_id], dtype=torch.long, device=device)

    generated_notes = []
    generated_durations = []
    rule_applications = []

    print(f"\n{'='*80}")
    print("TIMBRAL-BASED CONSTRAINED GENERATION")
    print(f"{'='*80}")
    print(f"\n📋 Rule: 4 consecutive closed strokes → 'Ti Re Ki T'")
    print(f"   Closed strokes: {', '.join(sorted(rule_engine.closed_crisp))}")
    print(f"\n🎲 Generating {num_generate} notes...\\n")

    with torch.no_grad():
        for step in range(num_generate):
            # Get current sequence length
            seq_len = context_notes.size(1)
            lengths = torch.tensor([seq_len], dtype=torch.long)

            # Forward pass
            note_logits, duration_logits = model(
                context_notes,
                context_durations,
                taal_tensor,
                lengths
            )

            # Get last timestep predictions
            note_logits_last = note_logits[0, -1, :] / temperature
            duration_logits_last = duration_logits[0, -1, :] / temperature

            # Apply timbral rule to note logits
            full_sequence = seed_notes + generated_notes
            constrained_note_logits, rule_applied = rule_engine.apply_timbral_rule(
                full_sequence, note_logits_last
            )

            # Sample from distributions
            note_probs = F.softmax(constrained_note_logits, dim=-1)
            duration_probs = F.softmax(duration_logits_last, dim=-1)

            next_note = torch.multinomial(note_probs, 1).item()
            next_duration = torch.multinomial(duration_probs, 1).item()

            generated_notes.append(next_note)
            generated_durations.append(next_duration)

            if rule_applied:
                rule_applications.append(step)

            # Append to context
            context_notes = torch.cat([
                context_notes,
                torch.tensor([[next_note]], dtype=torch.long, device=device)
            ], dim=1)
            context_durations = torch.cat([
                context_durations,
                torch.tensor([[next_duration]], dtype=torch.long, device=device)
            ], dim=1)

    print(f"\n✅ Generation complete!")
    print(f"   Timbral rule applied at positions: {rule_applications if rule_applications else 'None'}")

    return generated_notes, generated_durations, rule_applications


if __name__ == "__main__":
    print("="*80)
    print("TESTING TIMBRAL-BASED CONSTRAINED GENERATION")
    print("="*80)

    # Load model
    checkpoint = torch.load('models/best_bar_aware_lstm.pth', map_location='cpu')
    metadata = checkpoint['metadata']
    note_labels = metadata['note_labels']

    model, _ = create_model(
        vocab_size=metadata['num_classes'],
        num_duration_bins=metadata['num_duration_bins'],
        num_taals=len(metadata['taal_mapping']),
        hidden_size=checkpoint['hyperparameters']['hidden_size'],
        num_layers=checkpoint['hyperparameters']['num_layers'],
        dropout=checkpoint['hyperparameters']['dropout'],
        note_labels=metadata['note_labels']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\n✅ Model loaded")

    # Create a seed with closed strokes to trigger rule
    note_to_idx = {note: i for i, note in enumerate(note_labels)}

    # Seed: "Dha Ghe Kat" - next closed will start Ti Re Ki T
    seed_notes = [
        note_to_idx['Dha'],
        note_to_idx['Ghe'],
        note_to_idx['Kat'],  # This is closed, should trigger Ti
    ]
    seed_durations = [2, 2, 1]

    print(f"\n🌱 Seed: {' '.join([note_labels[i] for i in seed_notes])}")
    print(f"   (Last note 'Kat' is closed, next closed stroke should trigger 'Ti Re Ki T')")

    # Generate with timbral rules
    gen_notes, gen_durs, rule_apps = generate_with_timbral_rules(
        model=model,
        seed_notes=seed_notes,
        seed_durations=seed_durations,
        taal_id=1,  # Ektaal
        note_labels=note_labels,
        num_generate=32,
        temperature=1.0,
        device='cpu'
    )

    gen_note_labels = [note_labels[idx] for idx in gen_notes]

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"\n🎵 Generated sequence:")
    print(f"   {' '.join(gen_note_labels)}")

    # Verify rule was enforced
    full_sequence = [note_labels[i] for i in seed_notes] + gen_note_labels
    print(f"\n🔍 Full sequence (seed + generated):")
    print(f"   {' '.join(full_sequence)}")

    # Check for Ti Re Ki T
    pattern = ['Ti', 'Re', 'Ki', 'T']
    ti_re_ki_t_count = 0
    for i in range(len(full_sequence) - 3):
        if full_sequence[i:i+4] == pattern:
            ti_re_ki_t_count += 1
            print(f"\n✅ Found 'Ti Re Ki T' at position {i}: {' '.join(full_sequence[max(0,i-2):i+6])}")

    if ti_re_ki_t_count == 0:
        print(f"\n⚠️  'Ti Re Ki T' not found (maybe no 4 consecutive closed strokes generated)")
