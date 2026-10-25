"""rockhound: terrain representation and path planning for rover navigation."""

from .grid import Crater, Rock, TerrainGrid
from .rocks import RockFieldConfig, populate_rock_field
from .craters import CraterFieldConfig, populate_crater_field
from .texture import generate_dot_texture
from .planner import astar, smooth_path
from .planner_dstar import DStarLite
from .lighting import SunPosition, cast_shadow_mask, hillshade
from .visualization import plot_terrain
from .lidar import min_ray_count
from .lidar_sim import ScanResult, simulate_planar_scan, plot_scan_polar, plot_scan_overlay
from .lidar_terrain import (
    TerrainScanResult,
    LidarSpec,
    LIDAR_PRESETS,
    depression_angles_for_mount,
    simulate_downlooking_scan,
    simulate_downlooking_scan_from_spec,
    reconstruct_patch,
    plot_downlooking_scan_overlay,
)
from .stereo_sim import local_contrast, stereo_observed_mask
from .navigation_ekf import NavigationEKF, MotionEstimate

__all__ = [
    "TerrainGrid",
    "Rock",
    "Crater",
    "RockFieldConfig",
    "populate_rock_field",
    "CraterFieldConfig",
    "populate_crater_field",
    "generate_dot_texture",
    "astar",
    "smooth_path",
    "DStarLite",
    "SunPosition",
    "cast_shadow_mask",
    "hillshade",
    "plot_terrain",
    "min_ray_count",
    "ScanResult",
    "simulate_planar_scan",
    "plot_scan_polar",
    "plot_scan_overlay",
    "TerrainScanResult",
    "simulate_downlooking_scan",
    "LidarSpec",
    "LIDAR_PRESETS",
    "depression_angles_for_mount",
    "simulate_downlooking_scan_from_spec",
    "reconstruct_patch",
    "plot_downlooking_scan_overlay",
    "stereo_observed_mask",
    "local_contrast",
    "NavigationEKF",
    "MotionEstimate",
]
