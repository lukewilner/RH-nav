"""Does Sun angle affect lidar the way it affects stereo? No -- and this
demo shows why not, rather than just asserting it.

lidar_sim.simulate_planar_scan is a pure geometric ray-march against
elevation: it has no lighting input at all, so its hit rate is
identical at every Sun angle by construction -- lidar is an active,
time-of-flight sensor, and real down-looking lidar range measurements
genuinely don't depend on ambient sunlight either. That's the actual
justification (see the mission architecture doc, Section 2) for
pairing lidar with stereo: lidar is the sensor that *doesn't*
degrade near lunar midday.

Stereo, by contrast, is a passive camera -- it needs ambient visual
contrast to match features between the two eyes. rockhound doesn't
have a real stereo/feature-matching simulation yet, so this demo uses
hillshade's local contrast (its standard deviation across the scene)
as a stand-in for "how much visual signal is available," and plots it
against Sun elevation next to the (flat) lidar hit rate.

Run with: python examples/lighting_vs_lidar_demo.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from rockhound.craters import CraterFieldConfig, populate_crater_field
from rockhound.grid import TerrainGrid
from rockhound.lidar_sim import simulate_planar_scan
from rockhound.lighting import SunPosition, hillshade
from rockhound.rocks import RockFieldConfig, populate_rock_field

_ROCK_DENSITY_PER_M2 = 300 / 2500
_CRATER_DENSITY_PER_M2 = 25 / 2500


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-m", type=float, default=20.0, help="Square patch size, in meters")
    parser.add_argument("--resolution-m", type=float, default=0.05, help="Cell size, in meters")
    parser.add_argument("--density", type=float, default=1.5, help="Multiplier on rock/crater density")
    parser.add_argument("--sun-azimuth-deg", type=float, default=35.0, help="Sun compass direction")
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

    # A low sensor height matters here: most procedural rocks are small
    # (lognormal-distributed, median radius ~3cm), so the default 0.3m
    # planar-scan height clears nearly all of them and the "hit rate"
    # collapses to ~0 regardless of Sun angle -- not because lighting
    # matters, but because nothing tall enough is in view. Scanning close
    # to the ground gives a real, nonzero baseline to hold constant.
    sensor_x_m = sensor_y_m = args.size_m / 2.0
    scan = simulate_planar_scan(
        grid, sensor_x_m, sensor_y_m, sensor_height_m=0.02, fov_deg=360.0, num_rays=360
    )
    lidar_hit_rate = float(scan.hit.mean())

    sun_elevations_deg = np.linspace(2.0, 88.0, 30)
    stereo_contrast = [
        float(hillshade(grid, SunPosition(args.sun_azimuth_deg, el)).std())
        for el in sun_elevations_deg
    ]
    # Normalize against its own low-Sun value so the plot reads as "fraction
    # of best-case contrast," not an arbitrary hillshade unit.
    stereo_contrast = np.array(stereo_contrast) / stereo_contrast[0]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(sun_elevations_deg, stereo_contrast, color="darkorange", linewidth=2.2,
            label="stereo feature contrast (hillshade std, normalized)")
    ax.axhline(1.0, color="steelblue", linewidth=2.2, linestyle="--",
               label=f"lidar hit rate ({lidar_hit_rate:.0%}, constant -- active sensor)")

    ax.set_xlabel("Sun elevation (deg): low = lunar dawn/dusk, high = lunar midday")
    ax.set_ylabel("relative signal strength")
    ax.set_title("Why RockHound pairs lidar with stereo: only one degrades with Sun angle")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.15)
    fig.tight_layout()

    print(f"Lidar hit rate: {lidar_hit_rate:.1%} (identical at every Sun angle -- it never reads sun.elevation_deg)")
    print("Stereo contrast proxy, normalized to its low-Sun value:")
    for el, val in zip(sun_elevations_deg, stereo_contrast):
        print(f"  Sun el={el:5.1f} deg -> {val:.2f}")

    if args.out:
        fig.savefig(args.out, dpi=140, bbox_inches="tight")
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
