"""D* Lite driven by realistic sensing: stereo (primary) + down-looking
lidar (backup), instead of examples/dstar_demo.py's idealized "reveal a
disk around the rover" sensor.

Stereo's coverage is gated by Sun-angle contrast (rockhound.stereo_sim)
-- exactly the mission architecture's stated failure mode (Section 2):
near lunar midday its usable field of view can collapse to almost
nothing. The down-looking lidar (rockhound.lidar_terrain) has no such
gating, but is genuinely short-range and has real gaps between its
fixed beam lines (see output/downlooking_lidar_sample.png).

There's no artificial safety net here: the rover moves according to
what it currently believes (optimistic: unsensed = assumed clear), and
if stereo+lidar coverage genuinely missed a no-go cell directly in its
path, it drives onto it -- that's reported as a collision, not
silently prevented. The point of running this is to measure whether
that actually happens, and how often, as a function of Sun angle and
sensor coverage -- not to assume it can't.

Run with: python examples/dstar_stereo_lidar_demo.py
Try: --sun-elevation-deg 15 (dawn: stereo carries most of the load)
     --sun-elevation-deg 80 (near midday: watch lidar_backup discoveries
     climb, and check whether collisions start appearing)
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from rockhound.craters import CraterFieldConfig, populate_crater_field
from rockhound.grid import Rock, TerrainGrid
from rockhound.lidar_terrain import LIDAR_PRESETS, simulate_downlooking_scan_from_spec
from rockhound.lighting import SunPosition
from rockhound.planner_dstar import DStarLite
from rockhound.rocks import RockFieldConfig, populate_rock_field
from rockhound.stereo_sim import local_contrast, stereo_observed_mask
from rockhound.visualization import plot_terrain

_ROCK_DENSITY_PER_M2 = 300 / 2500
_CRATER_DENSITY_PER_M2 = 25 / 2500


def _add_gauntlet(grid: TerrainGrid, size_m: float, rng_seed: int = 99, n_rocks: int = 45) -> None:
    """A dense band of small no-go rocks crossing the direct diagonal
    route, spaced close enough that missing even one late is a real
    collision risk. Unlike the sparse ambient rock/crater field (most of
    which sits off to the sides of a corner-to-corner route) or the one
    big obvious crater from the earlier demo, this is deliberately built
    to actually require detecting several close-together hazards in a
    row -- a real test of sensing coverage, not just easy geometry.
    """
    rng = np.random.default_rng(rng_seed)
    center = size_m * 0.45
    direction = np.array([1.0, 1.0]) / np.sqrt(2)  # along the diagonal route
    perp = np.array([-1.0, 1.0]) / np.sqrt(2)  # across it

    along = rng.uniform(-1.25, 1.25, n_rocks)  # meters, band thickness along travel
    across = rng.uniform(-size_m * 0.25, size_m * 0.25, n_rocks)
    radii = rng.uniform(0.12, 0.28, n_rocks)  # all above no_go_radius_threshold_m=0.1

    for a, c, r in zip(along, across, radii):
        x = center + a * direction[0] + c * perp[0]
        y = center + a * direction[1] + c * perp[1]
        if not (0.0 <= x <= size_m and 0.0 <= y <= size_m):
            continue
        rock = Rock(x_m=float(x), y_m=float(y), radius_m=float(r), height_m=float(r * 0.6), source="gauntlet")
        idx = grid.add_rock(rock)
        dist_sq = (grid.x_coords - x) ** 2 + (grid.y_coords - y) ** 2
        footprint = dist_sq <= r * r
        if not np.any(footprint):
            continue
        grid.rock_id[footprint] = idx
        normalized = np.clip(1.0 - dist_sq[footprint] / (r * r), 0.0, 1.0)
        grid.elevation[footprint] += rock.height_m * np.sqrt(normalized)
        grid.traversable[footprint] = False


def _nearest_traversable(traversable: np.ndarray, cell: tuple) -> tuple:
    if traversable[cell]:
        return cell
    free = np.argwhere(traversable)
    dists = np.sum((free - np.array(cell)) ** 2, axis=1)
    return tuple(free[np.argmin(dists)])


def _lidar_hit_mask(scan, grid: TerrainGrid) -> np.ndarray:
    mask = np.zeros(grid.shape, dtype=bool)
    hit_mask = scan.hit
    for x, y in zip(scan.hit_x_m[hit_mask], scan.hit_y_m[hit_mask]):
        row, col = grid.world_to_index(x, y)
        if grid.in_bounds(row, col):
            mask[row, col] = True
    return mask


def run_scenario(
    size_m: float = 30.0,
    resolution_m: float = 0.1,
    density: float = 6.0,
    sun_elevation_deg: float = 70.0,
    stereo_fov_deg: float = 100.0,
    stereo_range_m: float = 6.0,
    contrast_threshold: float = 0.03,
    range_falloff_m: float = 2.0,
    lidar_preset: str = "velodyne_vlp16",
    mounting_tilt_deg: float = 30.0,
    lidar_sector_deg: float = 60.0,
    lidar_max_range_m: float = 8.0,
    lidar_sensor_height_m: float = 0.5,
    gauntlet: bool = False,
    gauntlet_n_rocks: int = 45,
) -> dict:
    """Run one stereo+lidar D* Lite traverse and return everything needed
    to report or plot it -- the reusable core behind both this script's
    CLI and output/gen_stereo_lidar_progress_slide.py, so a progress slide
    can regenerate real numbers instead of hardcoding a copy of them.
    """
    grid = TerrainGrid(size_m=size_m, resolution_m=resolution_m)
    area_m2 = size_m**2

    populate_rock_field(grid, RockFieldConfig(
        num_rocks=max(1, round(_ROCK_DENSITY_PER_M2 * area_m2 * density)),
        min_radius_m=0.02, max_radius_m=1.2, size_dist="lognormal",
        no_go_radius_threshold_m=0.1, rng_seed=42,
    ))
    populate_crater_field(grid, CraterFieldConfig(
        num_craters=max(1, round(_CRATER_DENSITY_PER_M2 * area_m2 * density)),
        min_diameter_m=0.15, max_diameter_m=3.0, rng_seed=7,
    ))
    if gauntlet:
        _add_gauntlet(grid, size_m, n_rocks=gauntlet_n_rocks)

    true_traversable = grid.traversable
    known_traversable = np.ones_like(true_traversable)  # optimistic: assume clear until sensed
    revealed = np.zeros_like(true_traversable)

    sun = SunPosition(azimuth_deg=40.0, elevation_deg=sun_elevation_deg)
    contrast = local_contrast(grid, sun)  # Sun position is fixed for this run: compute once, reuse every step
    lidar_spec = LIDAR_PRESETS[lidar_preset]

    margin = size_m * 0.1
    start = _nearest_traversable(true_traversable, grid.world_to_index(margin, margin))
    goal = _nearest_traversable(true_traversable, grid.world_to_index(size_m - margin, size_m - margin))
    start_x, start_y = grid.index_to_world(*start)
    goal_x, goal_y = grid.index_to_world(*goal)
    heading_deg = np.degrees(np.arctan2(goal_y - start_y, goal_x - start_x))

    discovery_counts = {"stereo": 0, "lidar_backup": 0}

    def sense_from(pose: tuple, heading: float) -> list:
        """Stereo first, then lidar -- in that order, so a cell only ever
        gets counted under the first source that actually resolved it.
        That makes discovery_counts["lidar_backup"] a direct measurement
        of "hazards stereo's contrast gate missed that the lidar caught
        instead," with no other source filling in gaps either misses.
        """
        x_m, y_m = grid.index_to_world(*pose)
        stereo_mask = stereo_observed_mask(
            grid, sun, x_m, y_m, heading, stereo_fov_deg, stereo_range_m,
            contrast_threshold, range_falloff_m, contrast=contrast,
        )
        scan = simulate_downlooking_scan_from_spec(
            grid, lidar_spec, mounting_tilt_deg, x_m, y_m,
            sensor_height_m=lidar_sensor_height_m, heading_deg=heading,
            azimuth_sector_deg=lidar_sector_deg, max_range_m=lidar_max_range_m,
        )
        lidar_mask = _lidar_hit_mask(scan, grid)

        changed = []
        for mask, source in ((stereo_mask, "stereo"), (lidar_mask, "lidar_backup")):
            newly = mask & ~revealed
            if not np.any(newly):
                continue
            revealed[newly] = True
            diff = newly & (known_traversable != true_traversable)
            changed.extend(zip(*np.nonzero(diff)))
            known_traversable[newly] = true_traversable[newly]
            discovery_counts[source] += int(np.count_nonzero(diff))
        return changed

    changed = sense_from(start, heading_deg)
    planner = DStarLite(known_traversable, start, goal)
    if changed:
        planner.update_cells(changed)
        planner.replan()

    current = start
    trail = [current]
    steps = 0
    max_steps = 6 * grid.n
    collisions = []  # (step, cell): rover moved onto a cell it believed clear but wasn't
    stuck = False

    while current != goal and steps < max_steps:
        steps += 1
        path = planner.get_path()
        if path is None or len(path) < 2:
            stuck = True
            break

        # No pre-move check here on purpose: the rover moves according to
        # what it currently believes, exactly as it would for real. If
        # stereo+lidar coverage actually missed a no-go cell right in its
        # path, it drives onto it -- that's a real sensing failure, and
        # we want to see it happen, not paper over it.
        intended_next = path[1]
        prev = current
        current = intended_next
        trail.append(current)

        if not true_traversable[current]:
            collisions.append((steps, current))

        planner.update_start(current)

        px, py = grid.index_to_world(*prev)
        cx, cy = grid.index_to_world(*current)
        if (cx, cy) != (px, py):
            heading_deg = np.degrees(np.arctan2(cy - py, cx - px))

        # Contact/bump sensing: whether or not stereo or lidar resolved
        # this cell in advance, the rover now unambiguously knows its own
        # true state (it's either standing on it or it isn't).
        if not revealed[current]:
            revealed[current] = True
            if known_traversable[current] != true_traversable[current]:
                known_traversable[current] = true_traversable[current]

        changed = sense_from(current, heading_deg)
        if changed:
            planner.update_cells(changed)
            planner.replan()

    return {
        "grid": grid,
        "sun": sun,
        "lidar_spec": lidar_spec,
        "sun_elevation_deg": sun_elevation_deg,
        "trail": trail,
        "steps": steps,
        "reached": current == goal,
        "stuck": stuck,
        "discovery_counts": discovery_counts,
        "collisions": collisions,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-m", type=float, default=30.0)
    parser.add_argument("--resolution-m", type=float, default=0.1)
    parser.add_argument("--density", type=float, default=6.0, help="Rock/crater density multiplier")
    parser.add_argument(
        "--sun-elevation-deg", type=float, default=70.0,
        help="Low = lunar dawn/dusk (stereo works well), high = near lunar midday (stereo contrast collapses)",
    )
    parser.add_argument("--stereo-fov-deg", type=float, default=100.0)
    parser.add_argument("--stereo-range-m", type=float, default=6.0)
    parser.add_argument("--contrast-threshold", type=float, default=0.03)
    parser.add_argument(
        "--range-falloff-m", type=float, default=2.0,
        help="How quickly required stereo contrast grows with range (smaller = shorter effective useful range)",
    )
    parser.add_argument(
        "--lidar-preset", choices=sorted(LIDAR_PRESETS), default="velodyne_vlp16",
        help="Real (approximate) sensor spec sheet to simulate -- see rockhound.lidar_terrain.LIDAR_PRESETS",
    )
    parser.add_argument(
        "--mounting-tilt-deg", type=float, default=30.0,
        help="How far down the whole lidar unit is physically pitched on the rover",
    )
    parser.add_argument(
        "--lidar-sector-deg", type=float, default=60.0,
        help="Forward-facing horizontal sector actually scanned each step (not the sensor's full native FOV)",
    )
    parser.add_argument("--lidar-max-range-m", type=float, default=8.0, help="Capped scan range for this scene size")
    parser.add_argument("--lidar-sensor-height-m", type=float, default=0.5, help="Placeholder, see module docstring")
    parser.add_argument(
        "--gauntlet", action="store_true",
        help="Add a dense band of small no-go rocks across the direct route -- a real sensing stress test",
    )
    parser.add_argument("--gauntlet-n-rocks", type=int, default=45, help="Rock count in the --gauntlet band")
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    lidar_spec = LIDAR_PRESETS[args.lidar_preset]
    print(f"Lidar: {lidar_spec.name}, mounted tilted down {args.mounting_tilt_deg:.0f} deg")

    result = run_scenario(
        size_m=args.size_m, resolution_m=args.resolution_m, density=args.density,
        sun_elevation_deg=args.sun_elevation_deg, stereo_fov_deg=args.stereo_fov_deg,
        stereo_range_m=args.stereo_range_m, contrast_threshold=args.contrast_threshold,
        range_falloff_m=args.range_falloff_m,
        lidar_preset=args.lidar_preset, mounting_tilt_deg=args.mounting_tilt_deg,
        lidar_sector_deg=args.lidar_sector_deg, lidar_max_range_m=args.lidar_max_range_m,
        lidar_sensor_height_m=args.lidar_sensor_height_m, gauntlet=args.gauntlet,
        gauntlet_n_rocks=args.gauntlet_n_rocks,
    )

    collisions = result["collisions"]
    desired_no_collisions = len(collisions) == 0
    if result["stuck"]:
        print(f"Stuck at {result['trail'][-1]}: no known path to goal.")

    print(f"Sun elevation: {args.sun_elevation_deg:.0f} deg")
    print(
        f"{'Reached goal' if result['reached'] else 'Did not reach goal'} in {result['steps']} steps "
        f"({len(result['trail'])} cells traveled)."
    )
    print(f"Discoveries by source: {result['discovery_counts']}")
    print(f"Desired result: 0 collisions with undetected no-go cells.")
    print(f"Actual result:  {len(collisions)} collision(s){' -- PASS' if desired_no_collisions else ' -- FAIL'}")
    for step, cell in collisions:
        print(f"  step {step}: drove onto undetected no-go cell {cell}")

    ax = plot_terrain(result["grid"], path=result["trail"], sun=result["sun"])
    stats_text = (
        f"Sun elevation: {args.sun_elevation_deg:.0f}°\n"
        f"Discoveries -- stereo: {result['discovery_counts']['stereo']}, "
        f"lidar backup: {result['discovery_counts']['lidar_backup']}\n"
        f"Collisions (undetected no-go cells): {len(collisions)} "
        f"({'PASS' if desired_no_collisions else 'FAIL'})"
    )
    ax.text(
        0.03, 0.03, stats_text, transform=ax.transAxes, fontsize=9, zorder=10,
        va="bottom", ha="left", color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.85),
    )
    if args.out:
        plt.savefig(args.out, dpi=140, bbox_inches="tight")
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
