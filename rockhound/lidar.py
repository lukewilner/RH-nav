"""Minimum lidar ray count needed to guarantee detecting a rock.

Standalone module: usable as a library function, or runnable directly
as a script:

    python -m rockhound.lidar --r-min 0.05 --d-max 20 --fov 120
"""

from __future__ import annotations

import argparse
import math


def min_ray_count(r_min_m: float, d_max_m: float, fov_deg: float) -> int:
    """Minimum number of lidar rays across `fov_deg` that guarantees
    detecting a rock of radius `r_min_m` at range `d_max_m`.

    Derived from: N > FOV / (2 * atan(r_min / d_max)) + 1
    (the rock subtends an angle of 2*atan(r_min/d_max) at range
    d_max, and rays spaced any coarser than that could straddle it
    entirely). Returns the smallest integer strictly greater than the
    bound.
    """
    if r_min_m <= 0 or d_max_m <= 0:
        raise ValueError("r_min_m and d_max_m must be positive")
    if not (0 < fov_deg <= 360):
        raise ValueError("fov_deg must be in (0, 360]")

    fov_rad = math.radians(fov_deg)
    angular_size_rad = 2.0 * math.atan(r_min_m / d_max_m)
    bound = fov_rad / angular_size_rad + 1.0
    return math.floor(bound) + 1


def _main() -> None:
    parser = argparse.ArgumentParser(description="Minimum lidar ray count for guaranteed rock detection.")
    parser.add_argument("--r-min", type=float, required=True, help="Minimum rock radius to guarantee detecting, in meters")
    parser.add_argument("--d-max", type=float, required=True, help="Maximum detection range, in meters")
    parser.add_argument("--fov", type=float, required=True, help="Horizontal field of view, in degrees")
    args = parser.parse_args()

    n = min_ray_count(args.r_min, args.d_max, args.fov)
    print(f"Minimum ray count: {n}")


if __name__ == "__main__":
    _main()
