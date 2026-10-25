"""Publishes fake-but-properly-formatted sensor topics for a simulated
outbound traverse -- IMU, wheel odometry, stereo VO, and periodic
star-tracker fixes -- computed by the same noise models already
validated in examples/ekf_localization_demo.py, just emitted as real
ROS messages instead of direct Python function calls.

This node owns the only copy of "ground truth" (true pose, terrain,
Sun-driven contrast field) that exists anywhere in this ROS graph --
exactly as a real simulator (not a real rover) would, since a real
rover never has access to its own ground truth. The /ground_truth_pose
topic is clearly sim-only and would not exist on real hardware; nothing
downstream (the EKF node) is allowed to subscribe to it for anything
but sanity plots.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu

from rockhound.craters import CraterFieldConfig, populate_crater_field
from rockhound.grid import TerrainGrid
from rockhound.lighting import SunPosition
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


def _yaw_to_quaternion(yaw_rad: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=float(np.sin(yaw_rad / 2.0)), w=float(np.cos(yaw_rad / 2.0)))


class PseudoSensorsNode(Node):
    def __init__(self) -> None:
        super().__init__("pseudo_sensors_node")

        self.declare_parameter("dt_s", 0.5)
        self.declare_parameter("speed_m_s", 0.3)
        self.declare_parameter("heading_amplitude_deg", 20.0)
        self.declare_parameter("heading_period_s", 60.0)
        self.declare_parameter("sun_elevation_deg", 70.0)
        self.declare_parameter("startracker_interval_s", 30.0)
        self.declare_parameter("rng_seed", 1)

        self.dt_s = self.get_parameter("dt_s").value
        self.speed_m_s = self.get_parameter("speed_m_s").value
        self.heading_amplitude_rad = np.radians(self.get_parameter("heading_amplitude_deg").value)
        self.heading_period_s = self.get_parameter("heading_period_s").value
        startracker_interval_s = self.get_parameter("startracker_interval_s").value
        self.startracker_every_n_ticks = max(1, round(startracker_interval_s / self.dt_s))
        self.rng = np.random.default_rng(self.get_parameter("rng_seed").value)

        # Fixed, not scaled by speed*duration -- that formula used to blow up
        # the grid (and rock count, which scales with area) to enormous sizes
        # whenever speed_m_s was increased for a faster demo. 150m comfortably
        # covers the amplitude-20deg/period-60s path at any reasonable speed.
        size_m = 150.0
        # Coarser than the validated (non-ROS) EKF demo on purpose: under
        # Rosetta-emulated amd64 in this container, populate_rock_field's
        # per-rock full-grid distance computation gets very slow at fine
        # resolution over a 150m grid -- 0.5m is plenty for this visual demo.
        self.grid = TerrainGrid(size_m=size_m, resolution_m=0.5)
        area_m2 = size_m**2
        populate_rock_field(self.grid, RockFieldConfig(
            num_rocks=max(1, round(300 / 2500 * area_m2)),
            min_radius_m=0.02, max_radius_m=1.2, size_dist="lognormal",
            no_go_radius_threshold_m=0.1, rng_seed=42,
        ))
        populate_crater_field(self.grid, CraterFieldConfig(
            num_craters=max(1, round(25 / 2500 * area_m2)),
            min_diameter_m=0.15, max_diameter_m=3.0, rng_seed=7,
        ))
        sun = SunPosition(azimuth_deg=40.0, elevation_deg=self.get_parameter("sun_elevation_deg").value)
        self.contrast_field = local_contrast(self.grid, sun)

        # Start near the left edge regardless of speed -- the old formula
        # assumed a fixed 300s run and placed the start far outside the
        # grid entirely at higher speeds.
        self.true_x = size_m * 0.15
        self.true_y = size_m / 2.0
        self.true_theta = self._true_heading(0.0)
        self.t_s = 0.0
        self.tick = 0

        self.imu_pub = self.create_publisher(Imu, "imu", 10)
        self.wheel_pub = self.create_publisher(Odometry, "wheel_odom", 10)
        self.vo_pub = self.create_publisher(TwistWithCovarianceStamped, "visual_odometry", 10)
        self.startracker_pub = self.create_publisher(PoseWithCovarianceStamped, "startracker", 10)
        self.ground_truth_pub = self.create_publisher(PoseStamped, "ground_truth_pose", 10)

        self.timer = self.create_timer(self.dt_s, self._on_tick)
        self.get_logger().info(
            f"pseudo_sensors_node publishing at {1.0/self.dt_s:.1f} Hz, "
            f"star-tracker every {startracker_interval_s:.0f}s, Sun el={self.get_parameter('sun_elevation_deg').value:.0f} deg"
        )

    def _true_heading(self, t_s: float) -> float:
        return self.heading_amplitude_rad * np.sin(2.0 * np.pi * t_s / self.heading_period_s)

    def _contrast_at(self, x_m: float, y_m: float) -> float:
        row, col = self.grid.world_to_index(x_m, y_m)
        row = int(np.clip(row, 0, self.grid.n - 1))
        col = int(np.clip(col, 0, self.grid.n - 1))
        return float(self.contrast_field[row, col])

    def _on_tick(self) -> None:
        now = self.get_clock().now().to_msg()
        t_next_s = self.t_s + self.dt_s

        theta_prev = self.true_theta
        theta_next = self._true_heading(t_next_s)
        true_dtheta = theta_next - theta_prev
        true_forward = self.speed_m_s * self.dt_s
        mid_theta = theta_prev + true_dtheta / 2.0
        self.true_x += true_forward * np.cos(mid_theta)
        self.true_y += true_forward * np.sin(mid_theta)
        self.true_theta = theta_next
        self.t_s = t_next_s
        self.tick += 1

        c = self._contrast_at(self.true_x, self.true_y)
        vo_scale = min(VO_NOISE_SCALE_CAP, VO_CONTRAST_REF / max(c, VO_CONTRAST_REF / VO_NOISE_SCALE_CAP))
        vo_translation_std = VO_TRANSLATION_STD_M_GOOD * vo_scale
        vo_yaw_std = VO_YAW_STD_RAD_GOOD * vo_scale

        imu_msg = Imu()
        imu_msg.header.stamp = now
        imu_msg.header.frame_id = "base_link"
        imu_msg.orientation_covariance[0] = -1.0  # per sensor_msgs/Imu: orientation not provided
        yaw_rate_meas = (true_dtheta + IMU_YAW_BIAS_RAD_S * self.dt_s) / self.dt_s + self.rng.normal(0, IMU_YAW_NOISE_STD_RAD_S)
        imu_msg.angular_velocity.z = float(yaw_rate_meas)
        imu_msg.angular_velocity_covariance[8] = IMU_YAW_NOISE_STD_RAD_S**2
        self.imu_pub.publish(imu_msg)

        wheel_noise_std = WHEEL_SLIP_STD_FRAC * true_forward
        wheel_speed_meas = (true_forward + self.rng.normal(0, wheel_noise_std)) / self.dt_s
        wheel_msg = Odometry()
        wheel_msg.header.stamp = now
        wheel_msg.header.frame_id = "odom"
        wheel_msg.child_frame_id = "base_link"
        wheel_msg.twist.twist.linear.x = float(wheel_speed_meas)
        wheel_msg.twist.covariance[0] = (wheel_noise_std / self.dt_s) ** 2
        self.wheel_pub.publish(wheel_msg)

        vo_forward_meas = (true_forward + self.rng.normal(0, vo_translation_std)) / self.dt_s
        vo_yaw_rate_meas = (true_dtheta + self.rng.normal(0, vo_yaw_std)) / self.dt_s
        vo_msg = TwistWithCovarianceStamped()
        vo_msg.header.stamp = now
        vo_msg.header.frame_id = "base_link"
        vo_msg.twist.twist.linear.x = float(vo_forward_meas)
        vo_msg.twist.twist.angular.z = float(vo_yaw_rate_meas)
        vo_msg.twist.covariance[0] = (vo_translation_std / self.dt_s) ** 2
        vo_msg.twist.covariance[35] = (vo_yaw_std / self.dt_s) ** 2
        self.vo_pub.publish(vo_msg)

        if self.tick % self.startracker_every_n_ticks == 0:
            theta_meas = self.true_theta + self.rng.normal(0, STARTRACKER_YAW_STD_RAD)
            st_msg = PoseWithCovarianceStamped()
            st_msg.header.stamp = now
            st_msg.header.frame_id = "map"
            st_msg.pose.pose.orientation = _yaw_to_quaternion(theta_meas)
            # Position isn't measured by a star tracker -- mark it as
            # effectively infinite variance so a consumer never mistakes
            # this for a position fix, only an attitude one.
            st_msg.pose.covariance[0] = 1e6
            st_msg.pose.covariance[7] = 1e6
            st_msg.pose.covariance[35] = STARTRACKER_YAW_STD_RAD**2
            self.startracker_pub.publish(st_msg)

        truth_msg = PoseStamped()
        truth_msg.header.stamp = now
        truth_msg.header.frame_id = "map"
        truth_msg.pose.position.x = float(self.true_x)
        truth_msg.pose.position.y = float(self.true_y)
        truth_msg.pose.orientation = _yaw_to_quaternion(self.true_theta)
        self.ground_truth_pub.publish(truth_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PseudoSensorsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
