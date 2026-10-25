"""Simulated 2D planar lidar/radar scan over a TerrainGrid.

Ray-marches outward from a sensor pose, testing each ray against the
grid's elevation field for a return. This only reads a TerrainGrid
through its public arrays -- the same interface any populator writes
into -- so it works identically on a procedurally-generated grid or a
future grid built from a real/simulated ROS2 sensor topic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .grid import TerrainGrid


@dataclass
class ScanResult:
    """One simulated sweep's worth of ray returns."""

    sensor_x_m: float
    sensor_y_m: float
    sensor_height_m: float
    heading_deg: float
    fov_deg: float
    max_range_m: float
    angles_deg: np.ndarray  # shape (num_rays,), absolute world angle per ray
    ranges_m: np.ndarray  # shape (num_rays,), NaN where no return
    hit_x_m: np.ndarray
    hit_y_m: np.ndarray
    hit: np.ndarray  # bool mask, True where the ray returned


def simulate_planar_scan(
    grid: TerrainGrid,
    sensor_x_m: float,
    sensor_y_m: float,
    sensor_height_m: float = 0.3,
    heading_deg: float = 0.0,
    fov_deg: float = 120.0,
    num_rays: int = 181,
    max_range_m: float = 5.0,
    range_step_m: Optional[float] = None,
) -> ScanResult:
    """Simulate one horizontal lidar/radar sweep from a fixed sensor pose.

    Models a beam traveling perfectly horizontally at `sensor_height_m`
    above the sensor's own local ground elevation. A ray returns at
    the first point along it where terrain elevation rises to meet
    that beam height -- i.e. something tall enough to intercept it.

    This means a rock registers a return once it's tall enough, but a
    crater's bowl (below grade) is invisible to a single horizontal
    sweep -- only its raised rim can return, if the rim is tall
    enough. That's a real sensing limitation of a planar scan, not an
    approximation error.
    """
    if range_step_m is None:
        range_step_m = grid.resolution_m

    sensor_row, sensor_col = grid.world_to_index(sensor_x_m, sensor_y_m)
    if not grid.in_bounds(sensor_row, sensor_col):
        raise ValueError("sensor position is outside the grid")
    beam_elevation = grid.elevation[sensor_row, sensor_col] + sensor_height_m

    half_fov = fov_deg / 2.0
    angles_deg = heading_deg + np.linspace(-half_fov, half_fov, num_rays)
    angles_rad = np.radians(angles_deg)

    ranges_m = np.full(num_rays, np.nan)
    hit_x_m = np.full(num_rays, np.nan)
    hit_y_m = np.full(num_rays, np.nan)
    hit = np.zeros(num_rays, dtype=bool)

    steps = np.arange(range_step_m, max_range_m + range_step_m, range_step_m)

    for i, theta in enumerate(angles_rad):
        dx, dy = np.cos(theta), np.sin(theta)
        for r in steps:
            x = sensor_x_m + r * dx
            y = sensor_y_m + r * dy
            row, col = grid.world_to_index(x, y)
            if not grid.in_bounds(row, col):
                break
            if grid.elevation[row, col] >= beam_elevation:
                ranges_m[i] = r
                hit_x_m[i] = x
                hit_y_m[i] = y
                hit[i] = True
                break

    return ScanResult(
        sensor_x_m=sensor_x_m,
        sensor_y_m=sensor_y_m,
        sensor_height_m=sensor_height_m,
        heading_deg=heading_deg,
        fov_deg=fov_deg,
        max_range_m=max_range_m,
        angles_deg=angles_deg,
        ranges_m=ranges_m,
        hit_x_m=hit_x_m,
        hit_y_m=hit_y_m,
        hit=hit,
    )


def plot_scan_polar(scan: ScanResult, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot a scan as an angular slice: angle around, range as radius --
    a lidar/radar plan-position-indicator (PPI) style sweep display,
    centered on the sensor's heading.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    elif ax.name != "polar":
        raise ValueError("plot_scan_polar requires a polar-projection Axes (subplot_kw={'projection': 'polar'})")

    theta = np.radians(scan.angles_deg - scan.heading_deg)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    hit_mask = scan.hit
    ax.plot(theta[hit_mask], scan.ranges_m[hit_mask], "o", color="crimson", markersize=4, label="return")
    ax.plot(
        theta[~hit_mask], np.full(np.count_nonzero(~hit_mask), scan.max_range_m),
        ".", color="lightgray", markersize=2, label="no return",
    )

    ax.set_thetamin(-scan.fov_deg / 2)
    ax.set_thetamax(scan.fov_deg / 2)
    ax.set_rmax(scan.max_range_m)
    ax.set_title(
        f"Lidar scan from ({scan.sensor_x_m:.1f}, {scan.sensor_y_m:.1f}) m, "
        f"sensor height {scan.sensor_height_m:.2f} m"
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    return ax


def plot_scan_overlay(ax: plt.Axes, scan: ScanResult) -> None:
    """Draw a scan's rays, FOV, and hit points onto an existing terrain
    axes (e.g. one returned by `plot_terrain`) for spatial context.
    """
    for angle_deg, hx, hy, hit in zip(scan.angles_deg, scan.hit_x_m, scan.hit_y_m, scan.hit):
        theta = np.radians(angle_deg)
        if hit:
            end_x, end_y = hx, hy
        else:
            end_x = scan.sensor_x_m + scan.max_range_m * np.cos(theta)
            end_y = scan.sensor_y_m + scan.max_range_m * np.sin(theta)
        ax.plot([scan.sensor_x_m, end_x], [scan.sensor_y_m, end_y], color="yellow", alpha=0.15, linewidth=0.6)

    hit_mask = scan.hit
    ax.scatter(scan.hit_x_m[hit_mask], scan.hit_y_m[hit_mask], color="red", s=14, zorder=6, label="lidar return")
    ax.plot(scan.sensor_x_m, scan.sensor_y_m, marker="^", color="black", markersize=10, zorder=6, label="sensor")
    ax.legend(loc="upper right")
