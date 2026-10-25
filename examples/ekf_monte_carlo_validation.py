"""Monte Carlo validation of the navigation EKF against the 0.5m
sample-location requirement (mission doc Section 1) -- this is the
actual answer to "does the architecture achieve the spec," not any
single run of examples/ekf_localization_demo.py.

Two things get checked, not one:

1. Final-error distribution across many independent noise-seed trials
   -- a single run's error is one sample from a distribution; the spec
   is about the distribution, not any one draw from it.
2. NEES consistency (Bar-Shalom, "Estimation with Applications to
   Tracking and Navigation") -- whether the filter's own covariance is
   a *believable* description of its actual error, not just whether
   the error happens to be small. A filter that is confidently wrong
   (covariance too small for its real error) is a worse outcome for a
   real mission than one that is honestly uncertain, because nothing
   downstream would know to distrust it.

Run with: python examples/ekf_monte_carlo_validation.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from rockhound.craters import CraterFieldConfig, populate_crater_field
from rockhound.grid import TerrainGrid
from rockhound.lighting import SunPosition
from rockhound.navigation_ekf import MotionEstimate, NavigationEKF
from rockhound.rocks import RockFieldConfig, populate_rock_field
from rockhound.stereo_sim import local_contrast

IMU_YAW_BIAS_RAD_S = 0.002
IMU_YAW_NOISE_STD_RAD_S = 0.01
WHEEL_SLIP_STD_FRAC = 0.08
VO_TRANSLATION_STD_M_GOOD = 0.01
VO_YAW_STD_RAD_GOOD = 0.002
VO_CONTRAST_REF = 0.05
VO_NOISE_SCALE_CAP = 25.0
STARTRACKER_YAW_STD_RAD = np.radians(0.1)

STATE_DIM = 3  # [x, y, theta] -- NEES's expected value under a consistent filter


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


def run_one_trial(
    rng: np.random.Generator, grid: TerrainGrid, contrast_field: np.ndarray,
    duration_s: float, dt_s: float, speed_m_s: float, amplitude_rad: float,
    period_s: float, startracker_interval_s: float, size_m: float,
) -> dict:
    def contrast_at(x_m: float, y_m: float) -> float:
        row, col = grid.world_to_index(x_m, y_m)
        row = int(np.clip(row, 0, grid.n - 1))
        col = int(np.clip(col, 0, grid.n - 1))
        return float(contrast_field[row, col])

    n_steps = int(duration_s / dt_s)
    origin = size_m / 2.0 - (speed_m_s * duration_s) / 2.0
    true_x, true_y = origin, size_m / 2.0
    true_theta = true_heading_rad(0.0, amplitude_rad, period_s)
    ekf = NavigationEKF(true_x, true_y, true_theta)
    startracker_interval_steps = max(1, round(startracker_interval_s / dt_s))

    nees_values = []
    final_error = None

    for step in range(n_steps):
        t_s = step * dt_s
        t_next_s = t_s + dt_s
        theta_prev = true_theta
        theta_next = true_heading_rad(t_next_s, amplitude_rad, period_s)
        true_dtheta = theta_next - theta_prev
        true_forward = speed_m_s * dt_s
        mid_theta = theta_prev + true_dtheta / 2.0
        true_x += true_forward * np.cos(mid_theta)
        true_y += true_forward * np.sin(mid_theta)
        true_theta = theta_next

        c = contrast_at(true_x, true_y)
        vo_scale = min(VO_NOISE_SCALE_CAP, VO_CONTRAST_REF / max(c, VO_CONTRAST_REF / VO_NOISE_SCALE_CAP))
        vo_translation_std = VO_TRANSLATION_STD_M_GOOD * vo_scale
        vo_yaw_std = VO_YAW_STD_RAD_GOOD * vo_scale

        imu = MotionEstimate(
            yaw_delta_rad=true_dtheta + IMU_YAW_BIAS_RAD_S * dt_s + rng.normal(0, IMU_YAW_NOISE_STD_RAD_S * dt_s),
            yaw_var=(IMU_YAW_NOISE_STD_RAD_S * dt_s) ** 2,
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

        if (step + 1) % startracker_interval_steps == 0:
            theta_meas = true_theta + rng.normal(0, STARTRACKER_YAW_STD_RAD)
            ekf.update_heading(theta_meas, STARTRACKER_YAW_STD_RAD)

        error = np.array([true_x, true_y, true_theta]) - ekf.state
        error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
        nees = float(error @ np.linalg.inv(ekf.cov) @ error)
        nees_values.append(nees)
        final_error = float(np.hypot(true_x - ekf.state[0], true_y - ekf.state[1]))

    return {"final_error_m": final_error, "mean_nees": float(np.mean(nees_values))}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--dt-s", type=float, default=0.5)
    parser.add_argument("--speed-m-s", type=float, default=0.3)
    parser.add_argument("--heading-amplitude-deg", type=float, default=20.0)
    parser.add_argument("--heading-period-s", type=float, default=60.0)
    parser.add_argument("--sun-elevation-deg", type=float, default=70.0)
    parser.add_argument("--startracker-interval-s", type=float, default=30.0)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    amplitude_rad = np.radians(args.heading_amplitude_deg)
    size_m = args.speed_m_s * args.duration_s * 1.3 + 10.0
    grid = build_terrain(size_m=size_m, resolution_m=0.1)
    sun = SunPosition(azimuth_deg=40.0, elevation_deg=args.sun_elevation_deg)
    contrast_field = local_contrast(grid, sun)

    final_errors = []
    mean_nees_per_trial = []
    for trial in range(args.n_trials):
        rng = np.random.default_rng(trial)
        result = run_one_trial(
            rng, grid, contrast_field, args.duration_s, args.dt_s, args.speed_m_s,
            amplitude_rad, args.heading_period_s, args.startracker_interval_s, size_m,
        )
        final_errors.append(result["final_error_m"])
        mean_nees_per_trial.append(result["mean_nees"])

    final_errors = np.array(final_errors)
    mean_nees_per_trial = np.array(mean_nees_per_trial)

    # Two-sided 95% chi-squared consistency bounds on mean NEES across
    # N trials (state dim 3): the standard Bar-Shalom consistency test.
    lower = stats.chi2.ppf(0.025, df=STATE_DIM * args.n_trials) / args.n_trials
    upper = stats.chi2.ppf(0.975, df=STATE_DIM * args.n_trials) / args.n_trials
    grand_mean_nees = float(np.mean(mean_nees_per_trial))
    consistent = lower <= grand_mean_nees <= upper

    within_spec_frac = float(np.mean(final_errors <= 0.5))
    p95 = float(np.percentile(final_errors, 95))

    print(f"Trials: {args.n_trials}, Sun el={args.sun_elevation_deg:.0f} deg, star-tracker every {args.startracker_interval_s:.0f}s")
    print(f"Final position error: mean={final_errors.mean():.3f}m, std={final_errors.std():.3f}m, "
          f"p95={p95:.3f}m, max={final_errors.max():.3f}m")
    print(f"Fraction of trials within 0.5m spec: {within_spec_frac:.1%}")
    print(f"Mean NEES: {grand_mean_nees:.2f} (expect ~{STATE_DIM} if consistent), "
          f"95% bounds [{lower:.2f}, {upper:.2f}] -> {'CONSISTENT' if consistent else 'NOT CONSISTENT (filter is miscalibrated)'}")
    if grand_mean_nees > upper:
        print("  -> mean NEES above the upper bound: the filter is OVERCONFIDENT -- its covariance")
        print("     understates its real error. This is the dangerous direction for a real mission:")
        print("     downstream systems would trust a position estimate that's worse than it claims.")
    elif grand_mean_nees < lower:
        print("  -> mean NEES below the lower bound: the filter is UNDERCONFIDENT (conservative but not wrong).")

    fig, (ax_hist, ax_nees) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax_hist.hist(final_errors, bins=30, color="steelblue", edgecolor="white")
    ax_hist.axvline(0.5, color="crimson", linestyle="--", linewidth=2, label="0.5m requirement")
    ax_hist.axvline(p95, color="darkorange", linestyle=":", linewidth=2, label=f"p95 = {p95:.2f}m")
    ax_hist.set_xlabel("final position error (m)")
    ax_hist.set_ylabel("trials")
    ax_hist.set_title(f"Final-error distribution ({args.n_trials} trials, {within_spec_frac:.0%} within spec)")
    ax_hist.legend()

    ax_nees.hist(mean_nees_per_trial, bins=30, color="mediumseagreen", edgecolor="white")
    ax_nees.axvline(STATE_DIM, color="black", linewidth=2, label=f"ideal = {STATE_DIM}")
    ax_nees.axvline(lower, color="crimson", linestyle="--", label="95% consistency bounds")
    ax_nees.axvline(upper, color="crimson", linestyle="--")
    ax_nees.set_xlabel("per-trial mean NEES")
    ax_nees.set_ylabel("trials")
    ax_nees.set_title(f"NEES consistency -- grand mean {grand_mean_nees:.2f} ({'CONSISTENT' if consistent else 'NOT CONSISTENT'})")
    ax_nees.legend()

    fig.suptitle("Navigation EKF: Monte Carlo validation against the 0.5m spec", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if args.out:
        fig.savefig(args.out, dpi=140, bbox_inches="tight")
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
