"""End-to-end demo: build a terrain grid, populate it procedurally, plan a
path across it, and visualize the result.

Run with: python examples/demo.py
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from rockhound.grid import TerrainGrid
from rockhound.rocks import RockFieldConfig, populate_rock_field
from rockhound.craters import CraterFieldConfig, populate_crater_field
from rockhound.texture import generate_dot_texture
from rockhound.planner import astar, smooth_path
from rockhound.visualization import plot_terrain
from rockhound.lidar import min_ray_count


def _nearest_traversable(grid: TerrainGrid, cell: tuple) -> tuple:
    if grid.traversable[cell]:
        return cell
    free = np.argwhere(grid.traversable)
    dists = np.sum((free - np.array(cell)) ** 2, axis=1)
    return tuple(free[np.argmin(dists)])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-m", type=float, default=50.0, help="Square patch size, in meters")
    parser.add_argument("--resolution-m", type=float, default=0.05, help="Cell size, in meters")
    parser.add_argument("--start-x", type=float, default=None, help="Start x-coordinate, in meters (default: bottom-left)")
    parser.add_argument("--start-y", type=float, default=None, help="Start y-coordinate, in meters (default: bottom-left)")
    parser.add_argument("--goal-x", type=float, default=None, help="Goal x-coordinate, in meters (default: bottom-right)")
    parser.add_argument("--goal-y", type=float, default=None, help="Goal y-coordinate, in meters (default: bottom-right)")
    parser.add_argument("--out", type=str, default=None, help="Save the plot to this file instead of showing it")
    parser.add_argument("--density", type=float, default=1.0, help="Multiplier on rock/crater density (1.0 = reference density)")
    return parser.parse_args()


# Reference densities from a 50m x 50m scene (2500 m^2), scaled to whatever
# --size-m is requested so a small grid doesn't end up far denser-looking
# than a large one just because the counts were hardcoded.
_ROCK_DENSITY_PER_M2 = 300 / 2500
_CRATER_DENSITY_PER_M2 = 25 / 2500


def main() -> None:
    args = _parse_args()
    grid = TerrainGrid(size_m=args.size_m, resolution_m=args.resolution_m)
    area_m2 = args.size_m**2

    cfg = RockFieldConfig(
        num_rocks=max(1, round(_ROCK_DENSITY_PER_M2 * area_m2 * args.density)),
        min_radius_m=0.02,
        max_radius_m=1.2,
        size_dist="lognormal",
        no_go_radius_threshold_m=0.1,
        rng_seed=42,
    )
    populate_rock_field(grid, cfg)

    crater_cfg = CraterFieldConfig(
        num_craters=max(1, round(_CRATER_DENSITY_PER_M2 * area_m2 * args.density)),
        min_diameter_m=0.15,
        max_diameter_m=3.0,
        rng_seed=7,
    )
    populate_crater_field(grid, crater_cfg)

    texture = generate_dot_texture(grid.shape, spacing_px=3, dot_radius_px=0.7)

    # Type in world coordinates (meters) via --start-x/--start-y/--goal-x/--goal-y.
    # Defaults: bottom-left area to bottom-right area, with a small margin
    # from the edges so the endpoints aren't right on the grid boundary.
    margin = args.size_m * 0.1
    start_x = args.start_x if args.start_x is not None else margin
    start_y = args.start_y if args.start_y is not None else margin
    goal_x = args.goal_x if args.goal_x is not None else args.size_m - margin
    goal_y = args.goal_y if args.goal_y is not None else margin

    start = _nearest_traversable(grid, grid.world_to_index(start_x, start_y))
    goal = _nearest_traversable(grid, grid.world_to_index(goal_x, goal_y))
    print(f"start: ({start_x:.2f}, {start_y:.2f}) m -> grid cell {start}")
    print(f"goal:  ({goal_x:.2f}, {goal_y:.2f}) m -> grid cell {goal}")

    path = astar(grid.traversable, start, goal)
    if path is None:
        print("No path found between start and goal.")
    else:
        smoothed = smooth_path(grid.traversable, path)
        print(f"Path found with {len(path)} grid cells, smoothed to {len(smoothed)} waypoints.")
        path = smoothed

    n_rays = min_ray_count(r_min_m=0.05, d_max_m=20.0, fov_deg=120.0)
    print(f"Minimum lidar rays for r_min=0.05m, d_max=20m, fov=120deg: {n_rays}")

    plot_terrain(grid, path=path, texture=texture)
    if args.out:
        plt.savefig(args.out, dpi=140, bbox_inches="tight")
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
