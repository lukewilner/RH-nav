"""Decoupled texture layer generator.

Produces a repetitive dot-pattern array sized to match a TerrainGrid,
for later visual feature-density testing. This module intentionally
has no dependency on elevation, traversability, or rock data -- it
only needs a grid shape.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def generate_dot_texture(
    shape: Tuple[int, int],
    spacing_px: int,
    dot_radius_px: float = 1.5,
) -> np.ndarray:
    """Return a float64 array of `shape` holding a repeating dot grid.

    Values lie in [0, 1]: 1.0 at each dot's center, falling off to 0.0
    at `dot_radius_px` away, and repeating every `spacing_px` cells in
    both dimensions.
    """
    if spacing_px <= 0:
        raise ValueError("spacing_px must be positive")
    if dot_radius_px <= 0:
        raise ValueError("dot_radius_px must be positive")

    n_rows, n_cols = shape
    rows = np.arange(n_rows).reshape(-1, 1)
    cols = np.arange(n_cols).reshape(1, -1)

    row_phase = rows % spacing_px
    col_phase = cols % spacing_px
    row_dist = np.minimum(row_phase, spacing_px - row_phase)
    col_dist = np.minimum(col_phase, spacing_px - col_phase)
    dist = np.sqrt(row_dist**2 + col_dist**2)

    return np.clip(1.0 - dist / dot_radius_px, 0.0, 1.0)
