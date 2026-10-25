"""Procedural rock-field populator.

This is one interchangeable "populator": a plain function that writes
rocks into a TerrainGrid through its public interface (`add_rock`, and
the `elevation` / `traversable` / `rock_id` arrays). A future
lidar-based populator (e.g. in a ROS2 node) would have the same shape
-- grid in, rocks written -- without TerrainGrid itself changing at
all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .distributions import sample_power_law
from .grid import Rock, TerrainGrid


@dataclass
class RockFieldConfig:
    """Parameters for procedural rock-field generation."""

    num_rocks: int = 200

    # Rock sizes are sampled from a lognormal or bounded power-law
    # distribution (real planetary surfaces have many small rocks and
    # few large ones -- a uniform distribution is not representative).
    size_dist: str = "lognormal"  # "lognormal" or "power_law"
    min_radius_m: float = 0.02
    max_radius_m: float = 1.0

    # lognormal params: radius ~ exp(Normal(lognormal_mu, lognormal_sigma))
    lognormal_mu: float = -3.5
    lognormal_sigma: float = 0.9

    # power_law params: p(r) ~ r ** power_law_exponent on [min, max]
    power_law_exponent: float = -2.5

    # Rocks with radius above this threshold block traversal; smaller
    # rocks still appear (elevation bump, rock_id) but remain go-cells.
    no_go_radius_threshold_m: float = 0.08

    # Elevation bump height at a rock's center = radius * this ratio.
    height_to_radius_ratio: float = 0.6

    rng_seed: Optional[int] = None


def _sample_radii(cfg: RockFieldConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    if cfg.size_dist == "lognormal":
        radii = rng.lognormal(mean=cfg.lognormal_mu, sigma=cfg.lognormal_sigma, size=n)
    elif cfg.size_dist == "power_law":
        radii = sample_power_law(cfg.power_law_exponent, cfg.min_radius_m, cfg.max_radius_m, n, rng)
    else:
        raise ValueError(f"unknown size_dist: {cfg.size_dist!r} (expected 'lognormal' or 'power_law')")
    return np.clip(radii, cfg.min_radius_m, cfg.max_radius_m)


def populate_rock_field(grid: TerrainGrid, cfg: RockFieldConfig = RockFieldConfig()) -> None:
    """Generate a procedural rock field and write it into `grid`.

    For each sampled rock: registers its metadata, marks every cell
    within its radius with the rock's index, adds a dome-shaped
    elevation bump under its footprint, and -- if its radius exceeds
    `cfg.no_go_radius_threshold_m` -- marks that footprint as no-go.
    """
    rng = np.random.default_rng(cfg.rng_seed)
    radii = _sample_radii(cfg, cfg.num_rocks, rng)
    xs = rng.uniform(0.0, grid.size_m, size=cfg.num_rocks)
    ys = rng.uniform(0.0, grid.size_m, size=cfg.num_rocks)

    x_coords = grid.x_coords
    y_coords = grid.y_coords

    for x, y, r in zip(xs, ys, radii):
        height = r * cfg.height_to_radius_ratio
        rock = Rock(x_m=float(x), y_m=float(y), radius_m=float(r), height_m=float(height), source="procedural")
        idx = grid.add_rock(rock)

        dist_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
        footprint = dist_sq <= r * r
        if not np.any(footprint):
            continue

        grid.rock_id[footprint] = idx

        # Dome-shaped bump: height at center, tapering to 0 at the rim.
        normalized = np.clip(1.0 - dist_sq[footprint] / (r * r), 0.0, 1.0)
        grid.elevation[footprint] += height * np.sqrt(normalized)

        if r > cfg.no_go_radius_threshold_m:
            grid.traversable[footprint] = False
