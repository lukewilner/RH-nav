"""Sun-angle dependent lighting for a TerrainGrid.

Stereo vision quality on the Moon is driven by Sun elevation, not by
how "sunny" a site is in the everyday sense. RockHound's target site
(the Ina irregular mare patch, ~18.6 deg N) is near-equatorial and
goes through a normal ~29.5-day lunar day/night cycle -- it is not a
polar site with a permanently low, grazing Sun. What matters is where
in that cycle a given traverse happens: near lunar midday the Sun is
high and terrain casts almost no shadow, which washes out the visual
contrast stereo matching depends on (hence the down-looking lidar);
near lunar dawn/dusk the Sun is low and terrain throws long shadows,
which is exactly the high-contrast condition stereo does well in.

This module renders that as two derived layers from a TerrainGrid's
elevation: a per-cell Lambertian hillshade intensity (cheap, local
self-shading from slope/aspect), and a cast-shadow mask from
ray-marching the heightfield toward the Sun (rocks/craters actually
occluding cells behind them). Both take a single `SunPosition` so a
caller can sweep through a simulated lunar day and see, e.g., feature
contrast collapse near midday -- the same effect the visual-nav stub
is meant to key its degradation model off of.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .grid import TerrainGrid


@dataclass
class SunPosition:
    """Sun direction at some point in the simulated lunar day.

    azimuth_deg: compass-style direction the Sun is coming from, measured
    in the grid's own x/y plane -- 0 points toward +y, 90 toward +x,
    increasing clockwise. There's no real "north" on this local grid, so
    this only needs to be internally consistent, not tied to a real
    lunar reference frame.

    elevation_deg: angle above the local horizon, 0-90. Low values are
    lunar dawn/dusk (long shadows, high stereo contrast); high values
    are lunar midday (short/no shadows, low contrast, lidar takes over).
    """

    azimuth_deg: float
    elevation_deg: float


def hillshade(grid: TerrainGrid, sun: SunPosition) -> np.ndarray:
    """Per-cell Lambertian illumination intensity in [0, 1].

    Purely local self-shading from surface slope/aspect vs. the Sun
    direction -- cheap, and already reproduces the qualitative effect
    that matters here: a low Sun throws slopes facing away from it into
    deep shadow (high contrast against lit slopes), while a high Sun
    flattens every surface toward similar brightness regardless of
    terrain shape (low contrast).
    """
    gy, gx = np.gradient(grid.elevation, grid.resolution_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)

    az = np.radians(sun.azimuth_deg)
    el = np.radians(sun.elevation_deg)

    intensity = np.sin(el) * np.cos(slope) + np.cos(el) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(intensity, 0.0, 1.0)


def cast_shadow_mask(grid: TerrainGrid, sun: SunPosition) -> np.ndarray:
    """Boolean mask, True where a cell sits in another feature's cast shadow.

    Rotates the heightfield so the Sun direction aligns with an axis,
    ray-marches every row toward the Sun (a cell is shadowed if any
    terrain between it and the Sun projects a ray that passes above it),
    then rotates the result back. This is what actually puts a rock's
    shadow on the ground behind it, as distinct from hillshade's purely
    local self-shading.
    """
    if sun.elevation_deg >= 90.0:
        return np.zeros(grid.shape, dtype=bool)
    if sun.elevation_deg <= 0.0:
        return np.ones(grid.shape, dtype=bool)

    # Rotate so the Sun direction aligns with increasing row index, i.e.
    # a photon travels in the direction of decreasing row as it moves
    # from the Sun toward the far side of the grid.
    elevation_rot = ndimage.rotate(
        grid.elevation, sun.azimuth_deg, reshape=False, order=1, mode="nearest"
    )

    tan_el = np.tan(np.radians(sun.elevation_deg))
    rows = np.arange(elevation_rot.shape[0], dtype=np.float64)[:, None]

    # h(row) = elevation - row * resolution * tan(el): a row is shadowed
    # iff some row closer to the Sun (larger row index, by this
    # rotation's convention) has a strictly greater h -- i.e. its ray,
    # projected forward at the Sun's grazing angle, still passes above
    # this row's actual terrain height.
    h = elevation_rot - rows * grid.resolution_m * tan_el
    suffix_max = np.maximum.accumulate(h[::-1, :], axis=0)[::-1, :]
    shadow_rot = suffix_max > h + 1e-9

    shadow = ndimage.rotate(
        shadow_rot.astype(np.float64), -sun.azimuth_deg, reshape=False, order=0, mode="nearest"
    )
    return shadow > 0.5
