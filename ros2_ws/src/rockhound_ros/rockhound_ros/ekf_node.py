"""Subscribes to the pseudo-sensor topics (IMU, wheel odometry, stereo
VO, star tracker) and fuses them with rockhound.navigation_ekf.NavigationEKF,
publishing the result as a standard nav_msgs/Odometry -- this is the
ROS-wrapped form of examples/ekf_localization_demo.py's filter, with the
exact same validated fusion/propagation logic, just driven by real
topic callbacks instead of a single in-process loop.

Initial pose is seeded from the one-time first /ground_truth_pose
message, matching how a real rover is initialized relative to the
lander's known position (mission doc Section 3) -- not an ongoing
subscription, since a real EKF node would never have continuing access
to ground truth.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu

from rockhound.navigation_ekf import MotionEstimate, NavigationEKF


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class EkfNode(Node):
    def __init__(self) -> None:
        super().__init__("ekf_node")
        self.declare_parameter("dt_s", 0.5)
        self.dt_s = self.get_parameter("dt_s").value

        self.ekf: Optional[NavigationEKF] = None
        self._latest_imu: Optional[Imu] = None
        self._latest_wheel: Optional[Odometry] = None
        self._latest_vo: Optional[TwistWithCovarianceStamped] = None

        self.create_subscription(PoseStamped, "ground_truth_pose", self._on_ground_truth_once, 10)
        self.create_subscription(Imu, "imu", self._on_imu, 10)
        self.create_subscription(Odometry, "wheel_odom", self._on_wheel, 10)
        self.create_subscription(TwistWithCovarianceStamped, "visual_odometry", self._on_vo, 10)
        self.create_subscription(PoseWithCovarianceStamped, "startracker", self._on_startracker, 10)

        self.fused_pub = self.create_publisher(Odometry, "odometry/filtered", 10)
        self.timer = self.create_timer(self.dt_s, self._on_predict_tick)
        self.get_logger().info("ekf_node waiting for initial ground_truth_pose to seed starting state...")

    def _on_ground_truth_once(self, msg: PoseStamped) -> None:
        if self.ekf is not None:
            return
        theta0 = _yaw_from_quaternion(
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        )
        self.ekf = NavigationEKF(msg.pose.position.x, msg.pose.position.y, theta0)
        self.get_logger().info(
            f"EKF seeded at x={msg.pose.position.x:.2f} y={msg.pose.position.y:.2f} theta={theta0:.3f}"
        )

    def _on_imu(self, msg: Imu) -> None:
        self._latest_imu = msg

    def _on_wheel(self, msg: Odometry) -> None:
        self._latest_wheel = msg

    def _on_vo(self, msg: TwistWithCovarianceStamped) -> None:
        self._latest_vo = msg

    def _on_startracker(self, msg: PoseWithCovarianceStamped) -> None:
        if self.ekf is None:
            return
        theta_meas = _yaw_from_quaternion(
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w,
        )
        theta_std = float(np.sqrt(msg.pose.covariance[35]))
        self.ekf.update_heading(theta_meas, theta_std)
        self.get_logger().info(f"star-tracker fix applied: theta_meas={theta_meas:.4f} rad")

    def _on_predict_tick(self) -> None:
        if self.ekf is None or self._latest_imu is None or self._latest_wheel is None or self._latest_vo is None:
            return  # not seeded yet, or haven't seen one of each sensor at least once

        motions = [
            MotionEstimate(
                yaw_delta_rad=self._latest_imu.angular_velocity.z * self.dt_s,
                yaw_var=self._latest_imu.angular_velocity_covariance[8] * self.dt_s**2,
            ),
            MotionEstimate(
                forward_m=self._latest_wheel.twist.twist.linear.x * self.dt_s,
                forward_var=self._latest_wheel.twist.covariance[0] * self.dt_s**2,
            ),
            MotionEstimate(
                forward_m=self._latest_vo.twist.twist.linear.x * self.dt_s,
                forward_var=self._latest_vo.twist.covariance[0] * self.dt_s**2,
                yaw_delta_rad=self._latest_vo.twist.twist.angular.z * self.dt_s,
                yaw_var=self._latest_vo.twist.covariance[35] * self.dt_s**2,
            ),
        ]
        self.ekf.predict(motions)

        x, y, theta = self.ekf.state
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = float(np.sin(theta / 2.0))
        msg.pose.pose.orientation.w = float(np.cos(theta / 2.0))
        msg.pose.covariance[0] = float(self.ekf.cov[0, 0])
        msg.pose.covariance[7] = float(self.ekf.cov[1, 1])
        msg.pose.covariance[35] = float(self.ekf.cov[2, 2])
        self.fused_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EkfNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
