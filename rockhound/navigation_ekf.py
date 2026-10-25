"""Extended Kalman Filter fusing IMU + wheel odometry + stereo VO motion
estimates, with periodic absolute star-tracker heading fixes.

This is the actual navigation/SLAM piece the mission architecture (Section
1) is built around -- RockHound's 0.5m sample-location requirement lives
or dies on how well this filter tracks pose, not on path planning. Path
planning (rockhound.planner_dstar) decides where to drive; this decides
where the rover *thinks* it is while driving there.

Design: IMU, wheel odometry, and VO all measure the same thing each
step -- how far forward and how much the heading changed -- with
different, independent error characteristics (Section 2 of the mission
doc): IMU drifts (bias) but is always available; wheel odometry is
corrupted by slip/scrub so it's a consistency check more than a source;
VO is the best source when lit well but degrades exactly when
rockhound.stereo_sim's contrast model says it should. Rather than
picking one as "the" motion source, every step fuses whichever of the
three are available by inverse-variance weighting (the closed-form
optimal way to combine independent Gaussian estimates of the same
quantity), then propagates state and covariance through a standard
odometry motion model (Thrun/Burgard/Fox, "Probabilistic Robotics").

Star-tracker fixes are separate: they measure absolute heading (not a
motion increment) and only arrive periodically, so they're a classic
EKF measurement update rather than part of the fused motion estimate.
This is deliberately the *only* thing in the filter that corrects
heading drift in an unbounded way -- IMU bias drifts forever between
fixes, which is the whole reason Section 2 calls the star tracker out
as necessary for long traverses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


def _wrap_angle(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def _fuse_gaussians(estimates: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Inverse-variance fusion of independent Gaussian estimates of the
    same scalar quantity. `estimates` is a list of (value, variance);
    variance <= 0 means "unavailable this step" and is skipped.
    """
    weights = [1.0 / var for _, var in estimates if var > 0.0]
    if not weights:
        raise ValueError("no available estimates to fuse")
    values = [val for val, var in estimates if var > 0.0]
    fused_var = 1.0 / sum(weights)
    fused_val = fused_var * sum(w * v for w, v in zip(weights, values))
    return fused_val, fused_var


@dataclass
class MotionEstimate:
    """One sensor's opinion of this step's motion, or None fields for
    whichever it doesn't measure (e.g. IMU has no forward-distance
    opinion; wheel odometry has no heading opinion).
    """

    forward_m: Optional[float] = None
    forward_var: Optional[float] = None
    yaw_delta_rad: Optional[float] = None
    yaw_var: Optional[float] = None


class NavigationEKF:
    """Tracks [x, y, theta] and its 3x3 covariance.

    Call `predict()` every step with whatever motion estimates are
    available (IMU/wheel/VO -- any subset), and `update_heading()`
    whenever a star-tracker fix arrives (much less often).
    """

    def __init__(self, x0_m: float, y0_m: float, theta0_rad: float, initial_std: Tuple[float, float, float] = (0.05, 0.05, 0.02)):
        self.state = np.array([x0_m, y0_m, theta0_rad], dtype=float)
        self.cov = np.diag([s**2 for s in initial_std])

    def predict(self, motions: List[MotionEstimate]) -> None:
        forward_candidates = [(m.forward_m, m.forward_var) for m in motions if m.forward_m is not None]
        yaw_candidates = [(m.yaw_delta_rad, m.yaw_var) for m in motions if m.yaw_delta_rad is not None]
        d, d_var = _fuse_gaussians(forward_candidates)
        dtheta, dtheta_var = _fuse_gaussians(yaw_candidates)

        x, y, theta = self.state
        mid_theta = theta + dtheta / 2.0
        cos_mid, sin_mid = np.cos(mid_theta), np.sin(mid_theta)

        self.state = np.array([
            x + d * cos_mid,
            y + d * sin_mid,
            _wrap_angle(theta + dtheta),
        ])

        # Jacobian of the motion model w.r.t. state [x, y, theta].
        F = np.array([
            [1.0, 0.0, -d * sin_mid],
            [0.0, 1.0, d * cos_mid],
            [0.0, 0.0, 1.0],
        ])
        # Jacobian w.r.t. the (independent) control noise [d, dtheta].
        V = np.array([
            [cos_mid, -d / 2.0 * sin_mid],
            [sin_mid, d / 2.0 * cos_mid],
            [0.0, 1.0],
        ])
        M = np.diag([d_var, dtheta_var])
        self.cov = F @ self.cov @ F.T + V @ M @ V.T

    def update_heading(self, theta_meas_rad: float, theta_std_rad: float) -> None:
        """Star-tracker fix: a direct measurement of absolute heading."""
        H = np.array([[0.0, 0.0, 1.0]])
        innovation = _wrap_angle(theta_meas_rad - self.state[2])
        S = (H @ self.cov @ H.T).item() + theta_std_rad**2
        K = (self.cov @ H.T) / S  # 3x1
        self.state = self.state + (K.flatten() * innovation)
        self.state[2] = _wrap_angle(self.state[2])
        self.cov = (np.eye(3) - K @ H) @ self.cov

    @property
    def heading_std_rad(self) -> float:
        return float(np.sqrt(self.cov[2, 2]))

    @property
    def position_std_m(self) -> Tuple[float, float]:
        return float(np.sqrt(self.cov[0, 0])), float(np.sqrt(self.cov[1, 1]))
