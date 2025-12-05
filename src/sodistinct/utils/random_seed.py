"""
random_seed.py
----------------
Gestion centralisée des seeds pour assurer la reproductibilité dans SoDistinct.

Fonctionnalités :
- set_global_seed(seed)
- generate_child_seed(base_seed, offset)
- Context manager pour RNG stable
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Optional

try:
    import numpy as np
except Exception:
    np = None

from sodistinct.config.settings import settings


# ==============================================================================
# Fonctions principales
# ==============================================================================

def set_global_seed(seed: Optional[int] = None) -> int:
    """
    Initialise les RNG Python / NumPy avec un seed donné.
    Si seed=None → utiliser seed global dans settings.
    Retourne le seed utilisé.
    """
    if seed is None:
        seed = settings.extras.random_seed

    random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    return seed


def generate_child_seed(base_seed: Optional[int], offset: int) -> Optional[int]:
    """
    Génère un sous-seed reproductible.
    Exemple :
        base=42, offset=5 → seed = 42 + 5 = 47
    """
    if base_seed is None:
        return None
    return base_seed + offset


# ==============================================================================
# Contexte RNG temporaire
# ==============================================================================

@contextmanager
def temp_seed(seed: Optional[int]):
    """
    Context manager : applique un seed temporaire puis restaure l'ancien RNG.
    Utilisé dans les expériences qui nécessitent un seed local.
    """
    old_state = random.getstate()
    old_np_state = np.random.get_state() if np is not None else None

    try:
        if seed is not None:
            random.seed(seed)
            if np is not None:
                np.random.seed(seed)
        yield
    finally:
        # Restore states
        random.setstate(old_state)
        if np is not None:
            np.random.set_state(old_np_state)
