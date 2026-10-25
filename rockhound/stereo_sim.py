"""Simulated stereo-vision terrain observation, gated by Sun angle.

This isn't a feature-matching/disparity simulation -- it models the
*consequence* the mission architecture cares about (Section 2): stereo
reliably resolves terrain within its field of view only where there's
enough visual contrast to match features on, and that contrast
collapses near lunar midday. A cell inside the camera's cone but below
the contrast threshold is geometrically visible but not confidently
resolved, so it's withheld here -- exactly the failure mode that
justifies backing stereo up with a down-looking lidar
(rockhound.lidar_terrain) rather than trusting it alone.

Note this gates on *local contrast* (spatial variation in shading),
not raw brightness. Those are opposite things here: near lunar midday
hillshade.py's output is uniformly bright almost everywhere (high
brightness, low contrast -- nothing to match features against), while
near dawn/dusk it's uniformly dim except for sharp shadow edges right
at rocks/craters (low brightness, high *local* contrast right where it
matters). Thresholding brightness itself would get the Sun-angle
dependence backwards.

It also gates on range, and not just as a hard cutoff. A rock's own
self-shadow gives it decent local contrast even at high Sun elevation
(its silhouette is locally steep regardless of Sun angle) -- flat
terrain is what loses contrast at midday, not obstacles themselves.
Left alone, that means a contrast-only gate barely ever fails: the
rover always eventually gets close enough for a hazard's own shading
to clear a flat threshold, no matter the Sun angle. Real stereo has a
second, independent limitation that has nothing to do with lighting:
depth precision from a fixed camera baseline degrades roughly with the
*square* of range (disparity shrinks as ~1/range, and depth error
scales with disparity error), so a distant, faint-contrast feature
needs much sharper contrast to be trusted than a close one does. This
module models that by scaling the required contrast up with range --
`contrast_threshold * (1 + (range / range_falloff_m)**2)` -- so a
sensor can have plenty of nominal max_range_m and still only reliably
resolve hazards once they're fairly close, independent of Sun angle.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter

from .grid import TerrainGrid
from .lighting import SunPosition, hillshade


def local_contrast(grid: TerrainGrid, sun: SunPosition, window_cells: int = 7) -> np.ndarray:
    """Windowed standard deviation of hillshade -- high right at shadow
    edges and rock/crater rims, low over uniformly flat or uniformly lit
    ground regardless of how bright that ground is.
    """
    shade = hillshade(grid, sun)
    mean = uniform_filter(shade, size=window_cells, mode="nearest")
    mean_sq = uniform_filter(shade**2, size=window_cells, mode="nearest")
    return np.sqrt(np.clip(mean_sq - mean**2, 0.0, None))


def stereo_observed_mask(
    grid: TerrainGrid,
    sun: SunPosition,
    rover_x_m: float,
    rover_y_m: float,
    heading_deg: float,
    fov_deg: float = 100.0,
    max_range_m: float = 6.0,
    contrast_threshold: float = 0.03,
    range_falloff_m: float = 2.0,
    contrast: Optional[np.ndarray] = None,
    occlusion_range_m: Optional[float] = 10.0,
) -> np.ndarray:
    """Boolean mask (grid.shape) of cells stereo confidently resolves.

    `contrast`, if given, should be a precomputed `local_contrast(grid,
    sun)` -- pass it in when calling this every simulation step of a
    fixed-Sun scenario so it isn't recomputed from scratch each time.

    `range_falloff_m` sets how quickly the *required* contrast grows
    with distance (see the module docstring) -- smaller means a
    shorter effective useful range regardless of `max_range_m`, larger
    means depth stays reliable further out (a bigger/better baseline).

    `occlusion_range_m` ray-marches out to that range (capped to
    `max_range_m`) so a no-go cell blocks the camera's view of whatever
    sits behind it, same as `lidar_terrain`'s ray-marching -- a real
    camera can't see through a rock. Beyond that range, occlusion is
    skipped (contrast/range gating still applies) purely for
    performance; pass None to disable occlusion checking entirely.
    """
    if contrast is None:
        contrast = local_contrast(grid, sun)

    dx = grid.x_coords - rover_x_m
    dy = grid.y_coords - rover_y_m
    dist = np.hypot(dx, dy)
    bearing_deg = np.degrees(np.arctan2(dy, dx))
    angle_diff = (bearing_deg - heading_deg + 180.0) % 360.0 - 180.0

    in_cone = (dist <= max_range_m) & (np.abs(angle_diff) <= fov_deg / 2.0)
    required_contrast = contrast_threshold * (1.0 + (dist / range_falloff_m) ** 2)
    mask = in_cone & (contrast >= required_contrast)

    if occlusion_range_m is None:
        return mask

    ray_range_m = min(occlusion_range_m, max_range_m)
    range_step_m = grid.resolution_m
    # Fine enough angular step that no cell at ray_range_m is skipped.
    angular_step_deg = np.degrees(range_step_m / max(ray_range_m, range_step_m))
    num_rays = max(int(fov_deg / angular_step_deg) + 1, 2)
    angles_deg = heading_deg + np.linspace(-fov_deg / 2.0, fov_deg / 2.0, num_rays)
    steps = np.arange(range_step_m, ray_range_m + range_step_m, range_step_m)

    # Ray-march out to ray_range_m; a ray stops at the first no-go cell it
    # hits (that cell's near face is visible, whatever's behind it isn't).
    visible = np.zeros(grid.shape, dtype=bool)
    for theta_deg in angles_deg:
        theta = np.radians(theta_deg)
        dxr, dyr = np.cos(theta), np.sin(theta)
        for r in steps:
            row, col = grid.world_to_index(rover_x_m + r * dxr, rover_y_m + r * dyr)
            if not grid.in_bounds(row, col):
                break
            visible[row, col] = True
            if not grid.traversable[row, col]:
                break

    beyond_ray_range = dist > ray_range_m
    return mask & (visible | beyond_ray_range)
