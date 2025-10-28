"""
Generate snare drum variants for different tabla strokes.
Creates synthesized variations using pitch shift, filtering, and envelope shaping.

Mapping (excluding Dha which is compound):
- Ti, Re, Ki, T, Kat, Na → all map to snare drum
Goal: Create perceptible but subtle differentiation
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import os
import sys


def load_base_snare(path='core/drums/Na.wav'):
    """Load the base snare sample (using Na as reference)."""
    y, sr = librosa.load(path, sr=None)
    return y, sr


def pitch_shift_sample(y, sr, n_cents):
    """
    Pitch shift a sample by n_cents (100 cents = 1 semitone).

    Args:
        y: audio samples
        sr: sample rate
        n_cents: pitch shift in cents (+/- 100 = semitone)
    """
    n_steps = n_cents / 100.0  # Convert cents to semitones
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    return y_shifted


def apply_eq(y, sr, freq_ranges):
    """
    Apply EQ using bandpass/highpass filters.

    Args:
        y: audio samples
        sr: sample rate
        freq_ranges: dict with 'highpass', 'boost_low', 'boost_mid', 'boost_high'
    """
    result = y.copy()

    # High-pass filter if specified
    if 'highpass' in freq_ranges:
        sos = signal.butter(4, freq_ranges['highpass'], 'hp', fs=sr, output='sos')
        result = signal.sosfilt(sos, result)

    # Boost specific frequency ranges using parametric EQ
    if 'boost_mid' in freq_ranges:
        # Simple gain boost in frequency range using bandpass
        center_freq, gain_db = freq_ranges['boost_mid']
        Q = 2.0  # Quality factor
        w0 = 2 * np.pi * center_freq / sr
        alpha = np.sin(w0) / (2 * Q)
        A = 10 ** (gain_db / 40)  # Gain factor

        # Peaking EQ coefficients
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        # Normalize
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1/a0, a2/a0])

        result = signal.lfilter(b, a, result)

    if 'boost_high' in freq_ranges:
        center_freq, gain_db = freq_ranges['boost_high']
        Q = 2.0
        w0 = 2 * np.pi * center_freq / sr
        alpha = np.sin(w0) / (2 * Q)
        A = 10 ** (gain_db / 40)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1/a0, a2/a0])

        result = signal.lfilter(b, a, result)

    return result


def adjust_envelope(y, sr, decay_factor=1.0):
    """
    Adjust the decay envelope of the sample.

    Args:
        y: audio samples
        sr: sample rate
        decay_factor: <1.0 = shorter decay, >1.0 = longer decay
    """
    # Create exponential decay envelope
    envelope = np.exp(-np.arange(len(y)) / (len(y) * decay_factor))

    # Apply only to tail (after attack)
    attack_samples = int(0.01 * sr)  # First 10ms is attack
    envelope[:attack_samples] = 1.0

    return y * envelope


def generate_snare_variant(base_snare, sr, variant_name, params):
    """
    Generate a snare variant with specified parameters.

    Args:
        base_snare: base snare audio
        sr: sample rate
        variant_name: name of the variant (e.g., 'Ti', 'Re')
        params: dict with 'pitch_cents', 'eq', 'decay_factor'
    """
    result = base_snare.copy()

    # 1. Pitch shift
    if 'pitch_cents' in params and params['pitch_cents'] != 0:
        result = pitch_shift_sample(result, sr, params['pitch_cents'])

    # 2. EQ/Filtering
    if 'eq' in params:
        result = apply_eq(result, sr, params['eq'])

    # 3. Envelope shaping
    if 'decay_factor' in params:
        result = adjust_envelope(result, sr, params['decay_factor'])

    # Normalize to prevent clipping
    result = result / np.max(np.abs(result)) * 0.9

    return result


def test_incremental_pitch():
    """
    Test 1: Start with pitch shift only, incrementally increase until 'too much'.
    Generate variants at different pitch shift levels for comparison.
    """
    print("=== TEST 1: Incremental Pitch Shift ===\n")

    base_snare, sr = load_base_snare('core/drums/Na.wav')

    # Create output directory
    os.makedirs('snare_variants_test/pitch_test', exist_ok=True)

    # Test different pitch shift amounts
    pitch_levels = [
        (10, "subtle"),
        (20, "light"),
        (30, "moderate"),
        (40, "noticeable"),
        (50, "strong"),
        (60, "aggressive"),
    ]

    print("Generating pitch shift test variants...")
    for cents, label in pitch_levels:
        # Bright variant (Ti/Re/Ki/T)
        bright = pitch_shift_sample(base_snare, sr, cents)
        bright = bright / np.max(np.abs(bright)) * 0.9
        sf.write(f'snare_variants_test/pitch_test/bright_{label}_{cents}cents.wav',
                 bright, sr)

        # Deep variant (Na)
        deep = pitch_shift_sample(base_snare, sr, -cents)
        deep = deep / np.max(np.abs(deep)) * 0.9
        sf.write(f'snare_variants_test/pitch_test/deep_{label}_{cents}cents.wav',
                 deep, sr)

        print(f"  Generated: ±{cents} cents ({label})")

    # Save base reference
    sf.write('snare_variants_test/pitch_test/base_reference.wav', base_snare, sr)
    print(f"\n✓ Test files saved to: snare_variants_test/pitch_test/")
    print("Listen and identify 'too much' threshold, then we'll back off 30-40%\n")


def generate_full_variants_v1():
    """
    Generate complete set of snare variants (Version 1: Pitch shift only).
    Based on testing results, using conservative pitch values.
    """
    print("=== Generating Full Snare Variants (V1: Pitch Only) ===\n")

    base_snare, sr = load_base_snare('core/drums/Na.wav')

    # Create output directory
    os.makedirs('snare_variants', exist_ok=True)

    # Variant parameters (starting conservative)
    variants = {
        'Ti': {
            'pitch_cents': 40,  # Bright, tight
            'eq': {},
            'decay_factor': 0.85,  # Shorter
        },
        'Re': {
            'pitch_cents': 30,  # Bright
            'eq': {},
            'decay_factor': 0.85,
        },
        'Ki': {
            'pitch_cents': 50,  # Brightest
            'eq': {},
            'decay_factor': 0.80,  # Shortest
        },
        'T': {
            'pitch_cents': 45,  # Very bright
            'eq': {},
            'decay_factor': 0.80,
        },
        'Kat': {
            'pitch_cents': 0,  # Neutral, base sample
            'eq': {},
            'decay_factor': 1.0,
        },
        'Na': {
            'pitch_cents': -25,  # Fuller, deeper
            'eq': {},
            'decay_factor': 1.10,  # Longer
        },
    }

    print("Generating variants:")
    for name, params in variants.items():
        variant = generate_snare_variant(base_snare, sr, name, params)
        output_path = f'snare_variants/snare_{name}.wav'
        sf.write(output_path, variant, sr)
        print(f"  ✓ {name}: pitch={params['pitch_cents']:+3d} cents, "
              f"decay={params['decay_factor']:.2f}x → {output_path}")

    print(f"\n✓ All variants saved to: snare_variants/")


def generate_full_variants_v2():
    """
    Generate complete set with pitch + EQ.
    Call this after testing V1 and deciding on EQ parameters.
    """
    print("=== Generating Full Snare Variants (V2: Pitch + EQ) ===\n")

    base_snare, sr = load_base_snare('core/drums/Na.wav')
    os.makedirs('snare_variants', exist_ok=True)

    variants = {
        'Ti': {
            'pitch_cents': 40,
            'eq': {
                'highpass': 200,  # Remove low end
                'boost_high': (4000, 3),  # Boost 4kHz by 3dB
            },
            'decay_factor': 0.85,
        },
        'Re': {
            'pitch_cents': 30,
            'eq': {
                'highpass': 200,
                'boost_high': (3500, 2.5),
            },
            'decay_factor': 0.85,
        },
        'Ki': {
            'pitch_cents': 50,
            'eq': {
                'highpass': 250,  # Most high-pass
                'boost_high': (5000, 3.5),
            },
            'decay_factor': 0.80,
        },
        'T': {
            'pitch_cents': 45,
            'eq': {
                'highpass': 200,
                'boost_high': (4500, 3),
            },
            'decay_factor': 0.80,
        },
        'Kat': {
            'pitch_cents': 0,
            'eq': {
                'boost_mid': (1000, 2),  # Slight mid boost
            },
            'decay_factor': 1.0,
        },
        'Na': {
            'pitch_cents': -25,
            'eq': {},  # Full spectrum
            'decay_factor': 1.10,
        },
    }

    print("Generating variants with EQ:")
    for name, params in variants.items():
        variant = generate_snare_variant(base_snare, sr, name, params)
        output_path = f'snare_variants/snare_{name}.wav'
        sf.write(output_path, variant, sr)
        eq_str = ', '.join([f"{k}={v}" for k, v in params.get('eq', {}).items()])
        print(f"  ✓ {name}: pitch={params['pitch_cents']:+3d} cents, "
              f"decay={params['decay_factor']:.2f}x, EQ=[{eq_str}]")

    print(f"\n✓ All variants saved to: snare_variants/")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--v1':
            generate_full_variants_v1()
        elif sys.argv[1] == '--v2':
            generate_full_variants_v2()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python core/generate_snare_variants.py [--v1|--v2]")
    else:
        print("Snare Variant Generator")
        print("=" * 50)
        print("\nStep 1: Testing pitch shift incrementally...")
        print("We'll generate variants at different pitch levels.")
        print("Listen to identify the 'too much' threshold.\n")

        test_incremental_pitch()

        print("\n" + "=" * 50)
        print("Next steps:")
        print("1. Listen to files in snare_variants_test/pitch_test/")
        print("2. Identify which level sounds 'too much'")
        print("3. Then we'll generate V1 with conservative values")
        print("4. After testing V1, we'll add EQ (V2)")
        print("\nTo generate V1 variants, run:")
        print("  python core/generate_snare_variants.py --v1")
        print("To generate V2 variants (with EQ), run:")
        print("  python core/generate_snare_variants.py --v2")
        print("=" * 50)