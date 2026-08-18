"""
heuristics.py 

"""

from __future__ import annotations

import numpy as np

from .mapping import ENR_HIGH





def count_adjacent_high_rods(enr_grid: np.ndarray) -> int:
    """
    Count the number of directlyb adjacent (horizontal or vertical) high-enrichment rods.
    
    """
    hi = (np.round(enr_grid, 4) == round(ENR_HIGH, 4))

    
    h_pairs = int(np.sum(hi[:, :-1] & hi[:, 1:]))     # Horizontal touches: cell [r, c] and [r, c+1] are both HIGH

    
    v_pairs = int(np.sum(hi[:-1, :] & hi[1:, :]))     # Vertical touches: cell [r, c] and [r+1, c] are both HIGH

    return h_pairs + v_pairs
