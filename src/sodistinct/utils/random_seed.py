from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Optional

try:
    import numpy as np
except Exception:
    np = None

from sodistinct.config.settings import settings



def set_global_seed(seed: Optional[int] = None) -> int:
 
    if seed is None:
        seed = settings.extras.random_seed

    random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    return seed


def generate_child_seed(base_seed: Optional[int], offset: int) -> Optional[int]:

    if base_seed is None:
        return None
    return base_seed + offset




@contextmanager
def temp_seed(seed: Optional[int]):

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
