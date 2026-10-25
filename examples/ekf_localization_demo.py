"""Navigation EKF demo: simulates one outbound traverse with noisy IMU,
wheel-odometry, and stereo-VO pseudo-sensors (VO noise tied to Sun-angle
contrast, same model as rockhound.stereo_sim) feeding rockhound.navigation_ekf,
with periodic star-tracker absolute-heading fixes, and compares the
filter's estimated path against ground truth -- this is the actual
question Section 1 of the mission architecture asks: can this sensor
suite hold position well enough for 0.5m sample-location accuracy?

Run with: python examples/ekf_localization_demo.py
Try: --sun-elevation-deg 15 (good VO contrast) vs 85 (degraded contrast,
     more reliance on IMU/wheel, faster-growing error between star-tracker
     fixes) -- and --startracker-interval-s 60 vs 10 to see how much
     fix frequency matters.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from rockhound.craters import CraterFieldConfig, populate_crater_field
from rockhound.grid import TerrainGrid
from rockhound.lighting import SunPosition
from rockhound.navigation_ekf import MotionEstimate, NavigationEKF
from rockhound.rocks import RockFieldConfig, populate_rock_field
from rockhound.stereo_sim import local_contrast

# Illustrative sensor noise levels -- not from a real datasheet (flagged
# the same way the lidar/stereo camera placeholders were earlier in this
# project). Swap these for real component specs once available.
IMU_YAW_BIAS_RAD_S = 0.002          # ~0.11 deg/s constant drift -- the
                                     # thing star-tracker fixes exist to bound
IMU_YAW_NOISE_STD_RAD_S = 0.01
WHEEL_SLIP_STD_FRAC = 0.08           # skid-steer slip/scrub, per the mission doc
VO_TRANSLATION_STD_M_GOOD = 0.01     # at a "good contrast" reference level
VO_YAW_STD_RAD_GOOD = 0.002
VO_CONTRAST_REF = 0.05
VO_NOISE_SCALE_CAP = 25.0            # how much worse VO can get at low contrast
STARTRACKER_YAW_STD_RAD = np.radians(0.1)


def build_terrain(size_m: float, resolution_m: float) -> TerrainGrid:
    grid = TerrainGrid(size_m=size_m, resolution_m=resolution_m)
    area_m2 = size_m**2
    populate_rock_field(grid, RockFieldConfig(
        num_rocks=max(1, round(300 / 2500 * area_m2)),
        min_radius_m=0.02, max_radius_m=1.2, size_dist="lognormal",
        no_go_radius_threshold_m=0.1, rng_seed=42,
    ))
    populate_crater_field(grid, CraterFieldConfig(
        num_craters=max(1, round(25 / 2500 * area_m2)),
        min_diameter_m=0.15, max_diameter_m=3.0, rng_seed=7,
    ))
    return grid


def true_heading_rad(t_s: float, amplitude_rad: float, period_s: float) -> float:
    return amplitude_rad * np.sin(2.0 * np.pi * t_s / period_s)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--dt-s", type=float, default=0.5)
    parser.add_argument("--speed-m-s", type=float, default=0.3)
    parser.add_argument("--heading-amplitude-deg", type=float, default=20.0)
    parser.add_argument("--heading-period-s", type=float, default=60.0)
    parser.add_argument("--sun-elevation-deg", type=float, default=70.0)
    parser.add_argument("--startracker-interval-s", type=float, default=30.0)
    parser.add_argument("--rng-seed", type=int, default=1)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.rng_seed)

    amplitude_rad = np.radians(args.heading_amplitude_deg)
    # Path extent from a full sine-heading traverse at this speed/duration,
    # plus margin, sizes the terrain grid so the rover never runs off it.
    size_m = args.speed_m_s * args.duration_s * 1.3 + 10.0
    grid = build_terrain(size_m=size_m, resolution_m=0.1)
    sun = SunPosition(azimuth_deg=40.0, elevation_deg=args.sun_elevation_deg)
    contrast_field = local_contrast(grid, sun)

    def contrast_at(x_m: float, y_m: float) -> float:
        row, col = grid.world_to_index(x_m, y_m)
        row = int(np.clip(row, 0, grid.n - 1))
        col = int(np.clip(col, 0, grid.n - 1))
        return float(contrast_field[row, col])

    n_steps = int(args.duration_s / args.dt_s)
    origin = size_m / 2.0 - (args.speed_m_s * args.duration_s) / 2.0  # center the path on the grid

    true_x, true_y = origin, size_m / 2.0
    true_theta = true_heading_rad(0.0, amplitude_rad, args.heading_period_s)
    ekf = NavigationEKF(true_x, true_y, true_theta)

    startracker_interval_steps = max(1, round(args.startracker_interval_s / args.dt_s))

    history = {
        "t_s": [], "true_x": [], "true_y": [], "true_theta": [],
        "est_x": [], "est_y": [], "est_theta": [],
        "heading_std_deg": [], "position_error_m": [], "startracker_fix": [],
    }

    for step in range(n_steps):
        t_s = step * args.dt_s
        t_next_s = t_s + args.dt_s

        theta_prev = true_theta
        theta_next = true_heading_rad(t_next_s, amplitude_rad, args.heading_period_s)
        true_dtheta = theta_next - theta_prev
        true_forward = args.speed_m_s * args.dt_s
        mid_theta = theta_prev + true_dtheta / 2.0
        true_x += true_forward * np.cos(mid_theta)
        true_y += true_forward * np.sin(mid_theta)
        true_theta = theta_next

        c = contrast_at(true_x, true_y)
        vo_scale = min(VO_NOISE_SCALE_CAP, VO_CONTRAST_REF / max(c, VO_CONTRAST_REF / VO_NOISE_SCALE_CAP))
        vo_translation_std = VO_TRANSLATION_STD_M_GOOD * vo_scale
        vo_yaw_std = VO_YAW_STD_RAD_GOOD * vo_scale

        imu = MotionEstimate(
            yaw_delta_rad=true_dtheta + IMU_YAW_BIAS_RAD_S * args.dt_s + rng.normal(0, IMU_YAW_NOISE_STD_RAD_S * args.dt_s),
            yaw_var=(IMU_YAW_NOISE_STD_RAD_S * args.dt_s) ** 2,
        )
        wheel_noise_std = WHEEL_SLIP_STD_FRAC * true_forward
        wheel = MotionEstimate(
            forward_m=true_forward + rng.normal(0, wheel_noise_std),
            forward_var=wheel_noise_std**2,
        )
        vo = MotionEstimate(
            forward_m=true_forward + rng.normal(0, vo_translation_std),
            forward_var=vo_translation_std**2,
            yaw_delta_rad=true_dtheta + rng.normal(0, vo_yaw_std),
            yaw_var=vo_yaw_std**2,
        )
        ekf.predict([imu, wheel, vo])

        fired = (step + 1) % startracker_interval_steps == 0
        if fired:
            theta_meas = true_theta + rng.normal(0, STARTRACKER_YAW_STD_RAD)
            ekf.update_heading(theta_meas, STARTRACKER_YAW_STD_RAD)

        est_x, est_y, est_theta = ekf.state
        history["t_s"].append(t_next_s)
        history["true_x"].append(true_x)
        history["true_y"].append(true_y)
        history["true_theta"].append(true_theta)
        history["est_x"].append(est_x)
        history["est_y"].append(est_y)
        history["est_theta"].append(est_theta)
        history["heading_std_deg"].append(np.degrees(ekf.heading_std_rad))
        history["position_error_m"].append(float(np.hypot(true_x - est_x, true_y - est_y)))
        history["startracker_fix"].append(fired)

    final_error = history["position_error_m"][-1]
    print(f"Sun elevation: {args.sun_elevation_deg:.0f} deg, star-tracker every {args.startracker_interval_s:.0f}s")
    print(f"Final position error: {final_error:.3f} m ({'within' if final_error <= 0.5 else 'EXCEEDS'} the 0.5m spec)")
    print(f"Max position error during traverse: {max(history['position_error_m']):.3f} m")

    fig, (ax_path, ax_heading, ax_error) = plt.subplots(1, 3, figsize=(18, 5.5))

    ax_path.plot(history["true_x"], history["true_y"], color="black", linewidth=2.0, label="true path")
    ax_path.plot(history["est_x"], history["est_y"], color="crimson", linewidth=1.6, linestyle="--", label="EKF estimate")
    ax_path.set_xlabel("x (m)")
    ax_path.set_ylabel("y (m)")
    ax_path.set_title("True vs. estimated path")
    ax_path.set_aspect("equal")
    ax_path.legend()

    t = np.array(history["t_s"])
    ax_heading.plot(t, history["heading_std_deg"], color="steelblue")
    fix_times = t[np.array(history["startracker_fix"])]
    for ft in fix_times:
        ax_heading.axvline(ft, color="darkorange", alpha=0.4, linewidth=1.0)
    ax_heading.set_xlabel("time (s)")
    ax_heading.set_ylabel("heading std (deg)")
    ax_heading.set_title("Heading uncertainty (orange = star-tracker fix)")

    ax_error.plot(t, history["position_error_m"], color="steelblue")
    ax_error.axhline(0.5, color="crimson", linestyle="--", label="0.5m requirement")
    ax_error.set_xlabel("time (s)")
    ax_error.set_ylabel("position error (m)")
    ax_error.set_title("True vs. estimated position error")
    ax_error.legend()

    fig.suptitle(
        f"Navigation EKF: Sun el={args.sun_elevation_deg:.0f}°, "
        f"star-tracker every {args.startracker_interval_s:.0f}s -- final error {final_error:.2f}m",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if args.out:
        fig.savefig(args.out, dpi=140, bbox_inches="tight")
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
