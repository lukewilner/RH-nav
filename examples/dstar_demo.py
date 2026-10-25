"""Incremental replanning demo: D* Lite vs. from-scratch A*.

Simulates a rover with limited sensing driving across a terrain it
does not fully know in advance -- it assumes the map is clear until
it senses otherwise, exactly like the "unknown terrain" scenario D*
Lite is built for. At each step it senses a radius around itself,
reveals any newly-discovered rocks/craters, and repairs its plan.

To show why that matters, at every step where new terrain was
revealed we also run astar() from scratch (purely for comparison,
its result is discarded) and tally how many cells it would have had
to (re-)expand. D* Lite's total should be a small fraction of that.

Run with: python examples/dstar_demo.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from rockhound.craters import CraterFieldConfig, populate_crater_field
from rockhound.grid import TerrainGrid
from rockhound.planner import astar
from rockhound.planner_dstar import DStarLite
from rockhound.rocks import RockFieldConfig, populate_rock_field
from rockhound.visualization import plot_terrain

# Same reference densities as examples/demo.py.
_ROCK_DENSITY_PER_M2 = 300 / 2500
_CRATER_DENSITY_PER_M2 = 25 / 2500


def _nearest_traversable(traversable: np.ndarray, cell: tuple) -> tuple:
    if traversable[cell]:
        return cell
    free = np.argwhere(traversable)
    dists = np.sum((free - np.array(cell)) ** 2, axis=1)
    return tuple(free[np.argmin(dists)])


def _reveal(known: np.ndarray, true: np.ndarray, center: tuple, radius_cells: int) -> list:
    """Copy ground truth into `known` within a square sensor footprint
    around `center`, in place, and return the cells that actually changed.
    """
    n_rows, n_cols = true.shape
    r0, c0 = center
    r_lo, r_hi = max(0, r0 - radius_cells), min(n_rows, r0 + radius_cells + 1)
    c_lo, c_hi = max(0, c0 - radius_cells), min(n_cols, c0 + radius_cells + 1)

    known_window = known[r_lo:r_hi, c_lo:c_hi]
    true_window = true[r_lo:r_hi, c_lo:c_hi]
    diff = known_window != true_window
    if not np.any(diff):
        return []

    rows, cols = np.nonzero(diff)
    known_window[diff] = true_window[diff]  # view into `known`, mutates in place
    return [(int(r) + r_lo, int(c) + c_lo) for r, c in zip(rows, cols)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-m", type=float, default=30.0, help="Square patch size, in meters")
    parser.add_argument("--resolution-m", type=float, default=0.1, help="Cell size, in meters")
    parser.add_argument("--sensor-radius-m", type=float, default=1.5, help="Sensing footprint radius, in meters")
    parser.add_argument(
        "--density", type=float, default=6.0,
        help="Multiplier on rock/crater density (default density has only ~5 "
             "no-go features in the whole scene, too sparse to reliably cross)",
    )
    parser.add_argument("--out", type=str, default=None, help="Save the plot to this file instead of showing it")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    grid = TerrainGrid(size_m=args.size_m, resolution_m=args.resolution_m)
    area_m2 = args.size_m**2

    populate_rock_field(grid, RockFieldConfig(
        num_rocks=max(1, round(_ROCK_DENSITY_PER_M2 * area_m2 * args.density)),
        min_radius_m=0.02, max_radius_m=1.2, size_dist="lognormal",
        no_go_radius_threshold_m=0.1, rng_seed=42,
    ))
    populate_crater_field(grid, CraterFieldConfig(
        num_craters=max(1, round(_CRATER_DENSITY_PER_M2 * area_m2 * args.density)),
        min_diameter_m=0.15, max_diameter_m=3.0, rng_seed=7,
    ))

    true_traversable = grid.traversable
    known_traversable = np.ones_like(true_traversable)  # optimistic: assume clear until sensed

    # Corner-to-corner rather than along one edge: it crosses much more of
    # the scene, which matters because rocks/craters are the only hazards
    # here (no staged obstacle) -- the route needs to actually pass near
    # several of them for D* Lite to have anything real to discover.
    margin = args.size_m * 0.1
    start = _nearest_traversable(true_traversable, grid.world_to_index(margin, margin))
    goal = _nearest_traversable(
        true_traversable, grid.world_to_index(args.size_m - margin, args.size_m - margin)
    )
    sensor_radius_cells = max(1, round(args.sensor_radius_m / args.resolution_m))

    _reveal(known_traversable, true_traversable, start, sensor_radius_cells)
    planner = DStarLite(known_traversable, start, goal)

    current = start
    traveled = [current]
    dstar_expansions = 0
    astar_expansions = 0
    replan_count = 0
    max_steps = 4 * grid.n

    steps = 0
    while current != goal and steps < max_steps:
        steps += 1
        path = planner.get_path()
        if path is None or len(path) < 2:
            print(f"Stuck at {current}: no known path to goal.")
            break

        current = path[1]
        traveled.append(current)
        planner.update_start(current)

        changed = _reveal(known_traversable, true_traversable, current, sensor_radius_cells)
        if changed:
            replan_count += 1
            planner.update_cells(changed)
            planner.replan()
            dstar_expansions += planner.last_expansions

            _, n_expanded = astar(known_traversable, current, goal, return_expansions=True)
            astar_expansions += n_expanded

    reached = current == goal
    print(f"{'Reached goal' if reached else 'Did not reach goal'} after {steps} steps, "
          f"{replan_count} replans triggered by newly-sensed terrain.")
    print(f"D* Lite total node expansions across all replans: {dstar_expansions}")
    print(f"From-scratch A* would have cost:                  {astar_expansions}")
    if astar_expansions:
        print(f"D* Lite did {dstar_expansions / astar_expansions:.1%} of the from-scratch work.")

    plot_terrain(grid, path=traveled)
    plt.gca().set_title("D* Lite: sensed-as-you-go traverse")
    if args.out:
        plt.savefig(args.out, dpi=140, bbox_inches="tight")
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
