"""Procedural crater-field populator.

Craters are a second, independent hazard type alongside rocks: real
planetary surfaces (the Moon especially) are pockmarked with impact
craters, and a crater can block a rover just as surely as a boulder --
via slope, not height. This populator writes into the same TerrainGrid
interface as rockhound.rocks (`add_crater`, and the `elevation` /
`traversable` / `crater_id` arrays); it doesn't touch rock data, and
the grid itself has no notion of "crater" baked in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .distributions import sample_power_law
from .grid import Crater, TerrainGrid


@dataclass
class CraterFieldConfig:
    """Parameters for procedural crater-field generation."""

    num_craters: int = 40

    # Lunar crater populations follow a power-law size-frequency
    # distribution (cumulative N(>D) ~ D^-b); diameter is the
    # conventional unit here, unlike rocks which use radius.
    min_diameter_m: float = 0.1
    max_diameter_m: float = 2.5
    size_frequency_exponent: float = -2.0

    # Simple bowl-shaped craters have a roughly constant depth/diameter
    # ratio (~0.2 for a fresh lunar crater; Pike 1974); older, degraded
    # craters are shallower, so each crater's ratio is drawn from a
    # normal distribution around the fresh value.
    depth_to_diameter_ratio: float = 0.2
    depth_to_diameter_ratio_sigma: float = 0.05

    # A raised rim just outside the crater radius, as is typical of
    # fresh simple craters (height and width given relative to depth
    # and radius respectively).
    rim_height_ratio: float = 0.15
    rim_width_ratio: float = 0.1

    # A crater blocks the rover only if it is both large enough to
    # matter (bigger than the rover can simply straddle) and steeper
    # than the rover can climb -- a small dimple or a wide shallow
    # crater are both drivable.
    no_go_diameter_threshold_m: float = 0.2
    max_traversable_slope_deg: float = 20.0

    rng_seed: Optional[int] = None


def populate_crater_field(grid: TerrainGrid, cfg: CraterFieldConfig = CraterFieldConfig()) -> None:
    """Generate a procedural crater field and write it into `grid`.

    For each sampled crater: registers its metadata, carves a
    paraboloid depression under its footprint with a small raised rim,
    and -- if its diameter exceeds `cfg.no_go_diameter_threshold_m`
    and its interior slope exceeds `cfg.max_traversable_slope_deg` --
    marks the bowl (not the rim) as no-go.
    """
    rng = np.random.default_rng(cfg.rng_seed)
    diameters = sample_power_law(
        cfg.size_frequency_exponent, cfg.min_diameter_m, cfg.max_diameter_m, cfg.num_craters, rng
    )
    diameters = np.clip(diameters, cfg.min_diameter_m, cfg.max_diameter_m)
    ratios = np.clip(
        rng.normal(cfg.depth_to_diameter_ratio, cfg.depth_to_diameter_ratio_sigma, size=cfg.num_craters),
        0.02, 0.35,
    )
    xs = rng.uniform(0.0, grid.size_m, size=cfg.num_craters)
    ys = rng.uniform(0.0, grid.size_m, size=cfg.num_craters)

    x_coords = grid.x_coords
    y_coords = grid.y_coords

    for x, y, diameter, ratio in zip(xs, ys, diameters, ratios):
        radius = diameter / 2.0
        depth = diameter * ratio
        slope_deg = np.degrees(np.arctan(2.0 * ratio))

        crater = Crater(x_m=float(x), y_m=float(y), radius_m=float(radius), depth_m=float(depth), source="procedural")
        idx = grid.add_crater(crater)

        dist_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
        bowl = dist_sq <= radius * radius

        if np.any(bowl):
            grid.crater_id[bowl] = idx
            normalized = np.clip(1.0 - dist_sq[bowl] / (radius * radius), 0.0, 1.0)
            grid.elevation[bowl] -= depth * normalized

        rim_width = radius * cfg.rim_width_ratio
        outer_radius = radius + rim_width
        rim = (dist_sq > radius * radius) & (dist_sq <= outer_radius * outer_radius)
        if np.any(rim):
            dist = np.sqrt(dist_sq[rim])
            rim_falloff = 1.0 - np.clip((dist - radius) / max(rim_width, 1e-9), 0.0, 1.0)
            grid.elevation[rim] += depth * cfg.rim_height_ratio * rim_falloff
            grid.crater_id[rim] = idx

        if diameter > cfg.no_go_diameter_threshold_m and slope_deg > cfg.max_traversable_slope_deg:
            grid.traversable[bowl] = False
