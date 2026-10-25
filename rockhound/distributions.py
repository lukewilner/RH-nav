"""Shared statistical sampling helpers for procedural populators."""

from __future__ import annotations

import numpy as np


def sample_power_law(
    exponent: float, x_min: float, x_max: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Inverse-CDF sampling of p(x) ~ x**exponent on [x_min, x_max]."""
    u = rng.uniform(0.0, 1.0, size=n)
    if abs(exponent + 1.0) < 1e-9:
        return x_min * (x_max / x_min) ** u
    exp = exponent + 1.0
    return (u * (x_max**exp - x_min**exp) + x_min**exp) ** (1.0 / exp)
