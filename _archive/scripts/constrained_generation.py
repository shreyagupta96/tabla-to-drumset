"""
Constrained Generation with Embedded Rules
Combines LSTM generation with hard-coded tabla composition rules
"""

import torch
import torch.nn.functional as F
from meter_conditional_lstm import create_model

class TablaRuleEngine:
    """
    Rule engine for tabla composition
    Enforces musical rules during generation
    """

    def __init__(self, note_labels):
        self.note_labels = note_labels
        self.note_to_idx = {note: idx for idx, note in enumerate(note_labels)}

        # Define rules as patterns
        self.rules = []

    def add_rule(self, pattern, rule_name=""):
        """
        Add a sequential pattern rule
        pattern: list of note names that must appear in sequence
        """
        pattern_indices = [self.note_to_idx[note] for note in pattern]
        self.rules.append({
            'name': rule_name,
            'pattern': pattern,
            'indices': pattern_indices,
            'length': len(pattern)
        })

    def check_rule_trigger(self, generated_sequence, rule):
        """
        Check if we're in the middle of a rule pattern
        Returns: (triggered, next_required_note_idx)
        """
        pattern_len = rule['length']
        pattern_indices = rule['indices']

        # Check if last (n-1) notes match first (n-1) notes of pattern
        for start_pos in range(1, pattern_len):
            # How many notes we need to match
            check_len = start_pos

            if len(generated_sequence) >= check_len:
                # Get last check_len notes
                recent = generated_sequence[-check_len:]

                # Check if they match the first check_len notes of pattern
                if recent == pattern_indices[:check_len]:
                    # Pattern partially matched! Return next required note
                    next_idx = pattern_indices[check_len]
                    return True, next_idx, start_pos

        return False, None, 0

    def apply_rules(self, generated_sequence, note_logits):
        """
        Apply all rules to constrain the next note generation

        Returns: modified logits with rules enforced
        """
        constrained_logits = note_logits.clone()

        for rule in self.rules:
            triggered, next_note_idx, position = self.check_rule_trigger(generated_sequence, rule)

            if triggered:
                # HARD CONSTRAINT: Force next note to be the required one
                # Set all other logits to -inf (probability 0)
                mask = torch.ones_like(constrained_logits) * float('-inf')
                mask[next_note_idx] = 0
                constrained_logits = constrained_logits + mask

                print(f"   🔒 RULE TRIGGERED: {rule['name']}")
                print(f"      Pattern: {' → '.join(rule['pattern'])}")
                print(f"      Position: {position}/{rule['length']}")
                print(f"      Forcing next note: {self.note_labels[next_note_idx]}")

                # Only apply first triggered rule
                break

        return constrained_logits


def generate_with_rules(
    model,
    seed_notes,
    seed_durations,
    taal_id,
    note_labels,
    rules=None,
    num_generate=32,
    temperature=1.0,
    device='cpu'
):
    """
    Generate tabla sequence with embedded rules

    Args:
        model: Trained LSTM model
        seed_notes: Initial note sequence (indices)
        seed_durations: Initial duration sequence (indices)
        taal_id: Taal type (0=Teental, 1=Ektaal)
        note_labels: List of note names
        rules: List of rule patterns, e.g. [['Ti', 'Re', 'Ki', 'T']]
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
    rule_engine = TablaRuleEngine(note_labels)

    if rules:
        for i, pattern in enumerate(rules):
            rule_name = f"Rule {i+1}: {' → '.join(pattern)}"
            rule_engine.add_rule(pattern, rule_name)

    # Convert to tensors
    context_notes = torch.tensor(seed_notes, dtype=torch.long, device=device).unsqueeze(0)
    context_durations = torch.tensor(seed_durations, dtype=torch.long, device=device).unsqueeze(0)
    taal_tensor = torch.tensor([taal_id], dtype=torch.long, device=device)

    generated_notes = []
    generated_durations = []
    rule_applications = []

    print(f"\n{'='*80}")
    print("CONSTRAINED GENERATION WITH RULES")
    print(f"{'='*80}")
    print(f"\n📋 Active Rules:")
    for rule in rule_engine.rules:
        print(f"   • {rule['name']}")
    print(f"\n🎲 Generating {num_generate} notes...\n")

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

            # Apply rules to note logits
            full_sequence = seed_notes + generated_notes
            constrained_note_logits = rule_engine.apply_rules(full_sequence, note_logits_last)

            # Check if rule was applied
            rule_applied = not torch.equal(note_logits_last, constrained_note_logits)

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
    print(f"   Rules applied at positions: {rule_applications if rule_applications else 'None'}")

    return generated_notes, generated_durations, rule_applications


if __name__ == "__main__":
    import pickle

    print("="*80)
    print("TESTING CONSTRAINED GENERATION")
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

    # Create a seed that will trigger the rule
    # Seed: "Dha Ghe Ghe Ti" - next should be forced to "Re Ki T"
    note_to_idx = {note: i for i, note in enumerate(note_labels)}
    seed_notes = [
        note_to_idx['Dha'],
        note_to_idx['Ghe'],
        note_to_idx['Ghe'],
        note_to_idx['Ti'],  # This will trigger Ti Re Ki T rule
    ]
    seed_durations = [2, 2, 2, 2]

    print(f"\n🌱 Seed: {' '.join([note_labels[i] for i in seed_notes])}")
    print(f"   (Last note 'Ti' will trigger the rule 'Ti Re Ki T')")

    # Define rules
    rules = [
        ['Ti', 'Re', 'Ki', 'T']  # Ti Re Ki T must always appear in sequence
    ]

    # Generate with rules
    gen_notes, gen_durs, rule_apps = generate_with_rules(
        model=model,
        seed_notes=seed_notes,
        seed_durations=seed_durations,
        taal_id=1,  # Ektaal
        note_labels=note_labels,
        rules=rules,
        num_generate=16,
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

    # Check if Ti Re Ki T appears
    pattern_str = 'Ti Re Ki T'
    full_str = ' '.join(full_sequence)

    if pattern_str in full_str:
        print(f"\n✅ SUCCESS: Pattern '{pattern_str}' enforced in generation!")
        # Find position
        for i in range(len(full_sequence) - 3):
            if full_sequence[i:i+4] == ['Ti', 'Re', 'Ki', 'T']:
                print(f"   Found at position {i}: {' '.join(full_sequence[max(0,i-2):i+6])}")
    else:
        print(f"\n⚠️  Pattern '{pattern_str}' not found")
