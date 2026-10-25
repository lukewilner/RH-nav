"""Simulated down-looking terrain lidar over a TerrainGrid.

rockhound.lidar_sim models a horizontal safety-scanner: one flat plane
of rays at a fixed height, good for "is something tall enough to hit
me" obstacle detection. That's not what the mission architecture
calls for -- Section 2 describes a *down-looking* lidar that
"observes short-range terrain in front of and beneath the rover" to
back up stereo when high Sun washes out visual contrast. That's a
terrain-*mapping* sensor, not an obstacle ring: each beam is angled
downward and returns an actual (x, y, z) ground point, and enough of
them together reconstruct a small heightmap patch directly ahead of
the rover -- including bumps stereo would also see, not just a binary
hit/no-hit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .grid import TerrainGrid


@dataclass
class LidarSpec:
    """A real (or hypothetical) multi-channel lidar's published spec
    sheet, in the sensor's own native mounting frame (0 deg = level).

    Real spinning multi-channel lidars publish each channel's vertical
    angle as +up/-down from horizontal *in the sensor's own frame* --
    they're not "down-looking" sensors by themselves. On a rover you'd
    physically tilt the whole unit down by some mounting angle so
    enough channels point at the ground ahead; see
    `depression_angles_for_mount`. Channels are assumed evenly spaced
    across `vertical_fov_deg`, which holds for most real multi-line
    lidars (Velodyne, Ouster) even if a few exotic ones use uneven
    spacing -- close enough for comparing sensors here.

    NOTE: the preset values in LIDAR_PRESETS below are approximate,
    from-memory published specs, not a live datasheet lookup -- verify
    against the current manufacturer datasheet before treating them as
    authoritative for a real design decision.
    """

    name: str
    num_channels: int
    vertical_fov_deg: float  # total spread, e.g. 30.0 for a +-15 deg unit
    horizontal_fov_deg: float = 360.0
    horizontal_resolution_deg: float = 0.2
    max_range_m: float = 100.0

    def channel_angles_deg(self) -> Tuple[float, ...]:
        """Native per-channel vertical angle, +up/-down, before mounting tilt."""
        if self.num_channels == 1:
            return (0.0,)
        half = self.vertical_fov_deg / 2.0
        return tuple(np.linspace(-half, half, self.num_channels))


def depression_angles_for_mount(spec: LidarSpec, mounting_tilt_deg: float) -> Tuple[float, ...]:
    """Each channel's effective downward angle once the unit is physically
    tilted down by `mounting_tilt_deg`. A channel that still points above
    the horizon after tilting (depression <= 0) can't see local terrain
    under this model at all and is dropped -- exactly what a real
    down-looking-terrain pipeline would do with it too.
    """
    depressions = [mounting_tilt_deg - angle for angle in spec.channel_angles_deg()]
    return tuple(d for d in depressions if d > 0.0)


# Approximate, from-memory published specs for a few real sensors people
# commonly reach for -- see the LidarSpec docstring's caveat above.
LIDAR_PRESETS: Dict[str, LidarSpec] = {
    "velodyne_vlp16": LidarSpec(
        name="Velodyne VLP-16 (Puck)", num_channels=16, vertical_fov_deg=30.0,
        horizontal_fov_deg=360.0, horizontal_resolution_deg=0.2, max_range_m=100.0,
    ),
    "ouster_os0_32": LidarSpec(
        name="Ouster OS0-32 (wide-FOV, short-range variant)", num_channels=32, vertical_fov_deg=90.0,
        horizontal_fov_deg=360.0, horizontal_resolution_deg=0.35, max_range_m=50.0,
    ),
    "single_point_rangefinder": LidarSpec(
        name="Single-point downward rangefinder (Garmin LIDAR-Lite-style altimeter)",
        num_channels=1, vertical_fov_deg=0.0,
        horizontal_fov_deg=1.0, horizontal_resolution_deg=1.0, max_range_m=40.0,
    ),
}


@dataclass
class TerrainScanResult:
    """One simulated down-looking sweep: several downward-tilted beam
    lines, each swept across the same azimuth range -- the down-looking
    analog of rockhound.lidar_sim.ScanResult.
    """

    sensor_x_m: float
    sensor_y_m: float
    sensor_height_m: float
    heading_deg: float
    azimuth_fov_deg: float
    depression_angles_deg: Tuple[float, ...]
    max_range_m: float
    azimuths_deg: np.ndarray  # shape (num_azimuth,)
    ranges_m: np.ndarray  # shape (n_pitch, num_azimuth), NaN where no return
    hit_x_m: np.ndarray
    hit_y_m: np.ndarray
    hit_z_m: np.ndarray  # measured ground elevation at the hit, in meters
    hit: np.ndarray  # bool, shape (n_pitch, num_azimuth)


def simulate_downlooking_scan(
    grid: TerrainGrid,
    sensor_x_m: float,
    sensor_y_m: float,
    sensor_height_m: float = 0.5,
    heading_deg: float = 0.0,
    azimuth_fov_deg: float = 60.0,
    num_azimuth: int = 61,
    depression_angles_deg: Sequence[float] = (15.0, 25.0, 35.0, 45.0, 55.0),
    max_range_m: float = 6.0,
    range_step_m: Optional[float] = None,
) -> TerrainScanResult:
    """Simulate a multi-line down-looking terrain scan.

    Each of `depression_angles_deg` is one beam line, angled that many
    degrees below horizontal, swept across `num_azimuth` directions
    spanning `azimuth_fov_deg`. Since the beam height now *descends*
    with range (unlike lidar_sim's flat plane), a beam eventually
    always meets the ground somewhere -- the return is wherever
    terrain elevation first rises to meet that descending height, and
    the recorded z is the terrain's actual elevation there. A steeper
    depression angle looks at closer ground; a shallower one reaches
    farther but skims lower, so a nearby bump can shadow what's behind
    it -- both are real properties of a real down-looking scanner, not
    simulation artifacts.
    """
    if range_step_m is None:
        range_step_m = grid.resolution_m

    sensor_row, sensor_col = grid.world_to_index(sensor_x_m, sensor_y_m)
    if not grid.in_bounds(sensor_row, sensor_col):
        raise ValueError("sensor position is outside the grid")
    sensor_z = grid.elevation[sensor_row, sensor_col] + sensor_height_m
    elevation_floor = grid.elevation.min()

    if num_azimuth == 1:
        azimuths_deg = np.array([heading_deg])  # a single fixed beam, dead ahead
    else:
        half_fov = azimuth_fov_deg / 2.0
        azimuths_deg = heading_deg + np.linspace(-half_fov, half_fov, num_azimuth)
    azimuths_rad = np.radians(azimuths_deg)
    steps = np.arange(range_step_m, max_range_m + range_step_m, range_step_m)

    n_pitch = len(depression_angles_deg)
    ranges_m = np.full((n_pitch, num_azimuth), np.nan)
    hit_x_m = np.full((n_pitch, num_azimuth), np.nan)
    hit_y_m = np.full((n_pitch, num_azimuth), np.nan)
    hit_z_m = np.full((n_pitch, num_azimuth), np.nan)
    hit = np.zeros((n_pitch, num_azimuth), dtype=bool)

    for p, depression_deg in enumerate(depression_angles_deg):
        tan_dep = np.tan(np.radians(depression_deg))
        for i, theta in enumerate(azimuths_rad):
            dx, dy = np.cos(theta), np.sin(theta)
            for r in steps:
                beam_z = sensor_z - r * tan_dep
                if beam_z < elevation_floor:
                    break  # beam is now below any terrain in the whole grid
                x = sensor_x_m + r * dx
                y = sensor_y_m + r * dy
                row, col = grid.world_to_index(x, y)
                if not grid.in_bounds(row, col):
                    break
                if grid.elevation[row, col] >= beam_z:
                    ranges_m[p, i] = r
                    hit_x_m[p, i] = x
                    hit_y_m[p, i] = y
                    hit_z_m[p, i] = grid.elevation[row, col]
                    hit[p, i] = True
                    break

    return TerrainScanResult(
        sensor_x_m=sensor_x_m,
        sensor_y_m=sensor_y_m,
        sensor_height_m=sensor_height_m,
        heading_deg=heading_deg,
        azimuth_fov_deg=azimuth_fov_deg,
        depression_angles_deg=tuple(depression_angles_deg),
        max_range_m=max_range_m,
        azimuths_deg=azimuths_deg,
        ranges_m=ranges_m,
        hit_x_m=hit_x_m,
        hit_y_m=hit_y_m,
        hit_z_m=hit_z_m,
        hit=hit,
    )


def simulate_downlooking_scan_from_spec(
    grid: TerrainGrid,
    spec: LidarSpec,
    mounting_tilt_deg: float,
    sensor_x_m: float,
    sensor_y_m: float,
    sensor_height_m: float,
    heading_deg: float = 0.0,
    azimuth_sector_deg: Optional[float] = None,
    max_range_m: Optional[float] = None,
    range_step_m: Optional[float] = None,
) -> TerrainScanResult:
    """simulate_downlooking_scan(), but parameterized by a real sensor's
    LidarSpec + how it's mounted, instead of picking depression angles
    and ray counts by hand.

    `azimuth_sector_deg` restricts the scan to a forward-facing wedge
    of the sensor's native horizontal FOV (most of these are 360 deg
    spinning units, but a rover only needs the ground ahead) -- default
    is the spec's own full horizontal_fov_deg. Angular resolution
    within that sector still matches the spec. `max_range_m` defaults
    to the spec's own max range, capped to whatever's actually useful
    for the grid at hand is the caller's job (a 30m grid doesn't
    benefit from simulating a 100m-range beam).
    """
    # Can't scan wider than the sensor's own native FOV -- a fixed single-
    # beam altimeter (horizontal_fov_deg=1) stays a single beam no matter
    # what sector a caller asks for.
    sector_deg = spec.horizontal_fov_deg if azimuth_sector_deg is None else min(azimuth_sector_deg, spec.horizontal_fov_deg)
    num_azimuth = max(1, round(sector_deg / spec.horizontal_resolution_deg) + 1)
    depression_angles_deg = depression_angles_for_mount(spec, mounting_tilt_deg)
    if not depression_angles_deg:
        raise ValueError(
            f"{spec.name!r} tilted {mounting_tilt_deg} deg has no channel pointed below the horizon"
        )

    return simulate_downlooking_scan(
        grid, sensor_x_m, sensor_y_m,
        sensor_height_m=sensor_height_m, heading_deg=heading_deg,
        azimuth_fov_deg=sector_deg, num_azimuth=num_azimuth,
        depression_angles_deg=depression_angles_deg,
        max_range_m=spec.max_range_m if max_range_m is None else max_range_m,
        range_step_m=range_step_m,
    )


def reconstruct_patch(
    scan: TerrainScanResult, resolution_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin this scan's hit points into a coarse local heightmap.

    Returns (x_edges, y_edges, height_grid): height_grid[row, col] is
    the mean measured elevation of hits landing in that bin, or NaN
    where nothing was observed -- this is what "the rover's own
    down-looking sensor thinks the ground ahead looks like," to be
    compared against the true grid.elevation over the same patch.
    """
    hit_mask = scan.hit
    xs, ys, zs = scan.hit_x_m[hit_mask], scan.hit_y_m[hit_mask], scan.hit_z_m[hit_mask]
    if xs.size == 0:
        raise ValueError("scan has no hits to reconstruct a patch from")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    n_cols = max(1, int(np.ceil((x_max - x_min) / resolution_m)) + 1)
    n_rows = max(1, int(np.ceil((y_max - y_min) / resolution_m)) + 1)
    x_edges = x_min + np.arange(n_cols + 1) * resolution_m
    y_edges = y_min + np.arange(n_rows + 1) * resolution_m

    sums = np.zeros((n_rows, n_cols))
    counts = np.zeros((n_rows, n_cols))
    col_idx = np.clip(((xs - x_min) / resolution_m).astype(int), 0, n_cols - 1)
    row_idx = np.clip(((ys - y_min) / resolution_m).astype(int), 0, n_rows - 1)
    np.add.at(sums, (row_idx, col_idx), zs)
    np.add.at(counts, (row_idx, col_idx), 1)

    with np.errstate(invalid="ignore"):
        height_grid = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return x_edges, y_edges, height_grid


def plot_downlooking_scan_overlay(ax: plt.Axes, scan: TerrainScanResult) -> None:
    """Draw a down-looking scan's hit points onto an existing terrain axes,
    colored by measured elevation -- the down-looking analog of
    rockhound.lidar_sim.plot_scan_overlay.
    """
    hit_mask = scan.hit
    sc = ax.scatter(
        scan.hit_x_m[hit_mask], scan.hit_y_m[hit_mask],
        c=scan.hit_z_m[hit_mask], cmap="viridis", s=16, zorder=6,
        label="down-looking hit (color = measured elevation)",
    )
    ax.plot(scan.sensor_x_m, scan.sensor_y_m, marker="^", color="black", markersize=10, zorder=6, label="sensor")
    ax.figure.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="measured elevation (m)")
    ax.legend(loc="upper right")
