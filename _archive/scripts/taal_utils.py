"""
Centralized Taal Utilities
Maps between meter (beats per bar) and taal_id for LSTM conditioning
"""

# Taal mapping for 4-taal system
TAAL_MAPPING = {
    16: {'id': 0, 'name': 'Teental', 'beats': 16},
    12: {'id': 1, 'name': 'Ektaal', 'beats': 12},
    10: {'id': 2, 'name': 'Jhaptaal', 'beats': 10},
    7:  {'id': 3, 'name': 'Rupak', 'beats': 7}
}

# Reverse mapping (id -> info)
ID_TO_TAAL = {
    0: {'name': 'Teental', 'beats': 16},
    1: {'name': 'Ektaal', 'beats': 12},
    2: {'name': 'Jhaptaal', 'beats': 10},
    3: {'name': 'Rupak', 'beats': 7}
}


def meter_to_taal_id(meter):
    """
    Convert detected meter (beats per bar) to taal_id for LSTM

    Args:
        meter: int, number of beats per bar (16, 12, 10, or 7)

    Returns:
        int, taal_id (0-3), or None if unsupported
    """
    if meter in TAAL_MAPPING:
        return TAAL_MAPPING[meter]['id']
    else:
        return None


def taal_id_to_name(taal_id):
    """
    Convert taal_id to taal name

    Args:
        taal_id: int (0-3)

    Returns:
        str, taal name
    """
    if taal_id in ID_TO_TAAL:
        return ID_TO_TAAL[taal_id]['name']
    else:
        return f"Unknown (id={taal_id})"


def taal_id_to_beats(taal_id):
    """
    Convert taal_id to number of beats

    Args:
        taal_id: int (0-3)

    Returns:
        int, number of beats per bar
    """
    if taal_id in ID_TO_TAAL:
        return ID_TO_TAAL[taal_id]['beats']
    else:
        return None


def get_all_taals():
    """
    Get list of all supported taals

    Returns:
        list of dicts with taal info
    """
    return [
        {'id': 0, 'name': 'Teental', 'beats': 16},
        {'id': 1, 'name': 'Ektaal', 'beats': 12},
        {'id': 2, 'name': 'Jhaptaal', 'beats': 10},
        {'id': 3, 'name': 'Rupak', 'beats': 7}
    ]


def validate_meter(meter):
    """
    Check if a detected meter is supported

    Args:
        meter: int, detected beats per bar

    Returns:
        bool, True if supported
    """
    return meter in TAAL_MAPPING


# Legacy 2-taal compatibility (for old scripts)
def meter_to_taal_id_legacy(meter):
    """
    Legacy 2-taal mapping for backwards compatibility
    Only supports Teental (16) and Ektaal (12)

    Args:
        meter: int, number of beats per bar

    Returns:
        int, taal_id (0 or 1), or 2 for unsupported
    """
    if meter == 16:
        return 0  # Teental
    elif meter == 12:
        return 1  # Ektaal
    else:
        return 2  # Other (unsupported in old models)
