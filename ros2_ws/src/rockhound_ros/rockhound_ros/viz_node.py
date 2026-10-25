"""Watchable version of the pipeline: subscribes to /ground_truth_pose
(from pseudo_sensors_node) and /odometry/filtered (from ekf_node), and
draws them in RViz as a black "true rover" icon and a red "EKF guess"
icon with trails behind each -- plus a live lidar scan (a fan of yellow
dots) cast from the rover's true position, so there's something
concretely visible instead of numbers to read.

This node owns its own copy of the same terrain used by
pseudo_sensors_node (same seeds -> identical rocks/craters) purely to
compute what the lidar would see; it has no other coupling to the rest
of the pipeline.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from rockhound.craters import CraterFieldConfig, populate_crater_field
from rockhound.grid import TerrainGrid
from rockhound.lidar_sim import simulate_planar_scan
from rockhound.rocks import RockFieldConfig, populate_rock_field


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _robot_marker(ns: str, marker_id: int, x: float, y: float, yaw: float, color: ColorRGBA) -> Marker:
    m = Marker()
    m.header.frame_id = "map"
    m.ns, m.id = ns, marker_id
    m.type, m.action = Marker.ARROW, Marker.ADD
    m.pose.position.x, m.pose.position.y = x, y
    m.pose.orientation.z, m.pose.orientation.w = float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))
    m.scale.x, m.scale.y, m.scale.z = 2.5, 0.6, 0.6
    m.color = color
    return m


def _trail_marker(ns: str, marker_id: int, points: list, color: ColorRGBA) -> Marker:
    m = Marker()
    m.header.frame_id = "map"
    m.ns, m.id = ns, marker_id
    m.type, m.action = Marker.LINE_STRIP, Marker.ADD
    m.scale.x = 0.15
    m.color = color
    m.points = [Point(x=px, y=py, z=0.0) for px, py in points]
    return m


class VizNode(Node):
    def __init__(self) -> None:
        super().__init__("viz_node")
        size_m = 150.0
        self.grid = TerrainGrid(size_m=size_m, resolution_m=0.5)
        area_m2 = size_m**2
        populate_rock_field(self.grid, RockFieldConfig(
            num_rocks=max(1, round(300 / 2500 * area_m2)),
            min_radius_m=0.3, max_radius_m=3.0, size_dist="lognormal",
            no_go_radius_threshold_m=0.5, rng_seed=42,
        ))
        populate_crater_field(self.grid, CraterFieldConfig(
            num_craters=max(1, round(25 / 2500 * area_m2)),
            min_diameter_m=1.0, max_diameter_m=6.0, rng_seed=7,
        ))

        self.true_trail: list = []
        self.est_trail: list = []
        self.max_trail_len = 120  # ~60s of trail at 2Hz -- unbounded trails got visually noisy
        self.true_heading = 0.0
        self.pub = self.create_publisher(MarkerArray, "viz_markers", 10)

        self.create_subscription(PoseStamped, "ground_truth_pose", self._on_ground_truth, 10)
        self.create_subscription(Odometry, "odometry/filtered", self._on_estimate, 10)
        self.get_logger().info("viz_node up, publishing MarkerArray on /viz_markers")

    def _scan_points(self, x_m: float, y_m: float, heading_deg: float) -> list:
        scan = simulate_planar_scan(
            self.grid, x_m, y_m, sensor_height_m=1.0, heading_deg=heading_deg,
            fov_deg=140.0, num_rays=61, max_range_m=15.0,
        )
        hit = scan.hit
        return list(zip(scan.hit_x_m[hit].tolist(), scan.hit_y_m[hit].tolist()))

    def _on_ground_truth(self, msg: PoseStamped) -> None:
        x, y = msg.pose.position.x, msg.pose.position.y
        self.true_heading = _yaw_from_quaternion(
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        )
        self.true_trail.append((x, y))
        self.true_trail = self.true_trail[-self.max_trail_len:]

        array = MarkerArray()
        array.markers.append(_robot_marker("true_robot", 0, x, y, self.true_heading, ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)))
        array.markers.append(_trail_marker("true_trail", 2, self.true_trail, ColorRGBA(r=0.0, g=0.0, b=0.0, a=0.7)))

        points = self._scan_points(x, y, np.degrees(self.true_heading))
        lidar = Marker()
        lidar.header.frame_id = "map"
        lidar.ns, lidar.id = "lidar", 4
        lidar.type, lidar.action = Marker.POINTS, Marker.ADD
        lidar.scale.x = lidar.scale.y = 0.9
        lidar.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.9)
        lidar.points = [Point(x=px, y=py, z=0.05) for px, py in points]
        array.markers.append(lidar)

        self.pub.publish(array)

    def _on_estimate(self, msg: Odometry) -> None:
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        yaw = _yaw_from_quaternion(
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w,
        )
        self.est_trail.append((x, y))
        self.est_trail = self.est_trail[-self.max_trail_len:]

        array = MarkerArray()
        array.markers.append(_robot_marker("est_robot", 1, x, y, yaw, ColorRGBA(r=0.85, g=0.1, b=0.1, a=1.0)))
        array.markers.append(_trail_marker("est_trail", 3, self.est_trail, ColorRGBA(r=0.85, g=0.1, b=0.1, a=0.6)))
        self.pub.publish(array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VizNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
