"""Sun-angle lighting demo: same terrain, three points in the lunar day.

Renders one terrain patch at a low Sun elevation (lunar dawn/dusk --
long shadows, high contrast), a mid elevation, and a high elevation
(lunar midday -- short shadows, low contrast) side by side. This is
the visual version of what rockhound.lighting exists for: giving the
visual-nav/stereo simulation something concrete to key its "how good
are my features right now" degradation model off of.

Run with: python examples/lighting_demo.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from rockhound.craters import CraterFieldConfig, populate_crater_field
from rockhound.grid import TerrainGrid
from rockhound.lighting import SunPosition
from rockhound.rocks import RockFieldConfig, populate_rock_field
from rockhound.visualization import plot_terrain

_ROCK_DENSITY_PER_M2 = 300 / 2500
_CRATER_DENSITY_PER_M2 = 25 / 2500


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-m", type=float, default=20.0, help="Square patch size, in meters")
    parser.add_argument("--resolution-m", type=float, default=0.05, help="Cell size, in meters")
    parser.add_argument("--sun-azimuth-deg", type=float, default=35.0, help="Sun compass direction")
    parser.add_argument("--density", type=float, default=1.5, help="Multiplier on rock/crater density")
    parser.add_argument("--out", type=str, default=None, help="Save the plot to this file instead of showing it")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    grid = TerrainGrid(size_m=args.size_m, resolution_m=args.resolution_m)
    area_m2 = args.size_m**2

    populate_rock_field(grid, RockFieldConfig(
        num_rocks=max(1, round(_ROCK_DENSITY_PER_M2 * area_m2 * args.density)),
        min_radius_m=0.05, max_radius_m=1.2, size_dist="lognormal",
        no_go_radius_threshold_m=0.1, rng_seed=42,
    ))
    populate_crater_field(grid, CraterFieldConfig(
        num_craters=max(1, round(_CRATER_DENSITY_PER_M2 * area_m2 * args.density)),
        min_diameter_m=0.3, max_diameter_m=3.0, rng_seed=7,
    ))

    sun_elevations_deg = [8.0, 40.0, 80.0]
    labels = ["lunar dawn/dusk", "mid-morning", "near lunar midday"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, elevation_deg, label in zip(axes, sun_elevations_deg, labels):
        sun = SunPosition(azimuth_deg=args.sun_azimuth_deg, elevation_deg=elevation_deg)
        plot_terrain(grid, sun=sun, ax=ax)
        ax.set_title(f"{label} (Sun el={elevation_deg:.0f}°)")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()  # each panel repeating the same rock/crater legend is just noise

    fig.suptitle("Same terrain across a simulated lunar day: shadow/contrast vs. Sun elevation")
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=140, bbox_inches="tight")
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
