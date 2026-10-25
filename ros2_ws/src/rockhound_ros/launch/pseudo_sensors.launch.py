import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    sun_elevation_deg = DeclareLaunchArgument("sun_elevation_deg", default_value="70.0")
    startracker_interval_s = DeclareLaunchArgument("startracker_interval_s", default_value="30.0")
    dt_s = DeclareLaunchArgument("dt_s", default_value="0.5")
    start_rviz = DeclareLaunchArgument("start_rviz", default_value="false")
    speed_m_s = DeclareLaunchArgument("speed_m_s", default_value="0.3")
    heading_amplitude_deg = DeclareLaunchArgument("heading_amplitude_deg", default_value="20.0")
    heading_period_s = DeclareLaunchArgument("heading_period_s", default_value="60.0")

    sensors_node = Node(
        package="rockhound_ros",
        executable="pseudo_sensors_node",
        name="pseudo_sensors_node",
        output="screen",
        parameters=[{
            "sun_elevation_deg": LaunchConfiguration("sun_elevation_deg"),
            "startracker_interval_s": LaunchConfiguration("startracker_interval_s"),
            "dt_s": LaunchConfiguration("dt_s"),
            "speed_m_s": LaunchConfiguration("speed_m_s"),
            "heading_amplitude_deg": LaunchConfiguration("heading_amplitude_deg"),
            "heading_period_s": LaunchConfiguration("heading_period_s"),
        }],
    )
    ekf_node = Node(
        package="rockhound_ros",
        executable="ekf_node",
        name="ekf_node",
        output="screen",
        parameters=[{"dt_s": LaunchConfiguration("dt_s")}],
    )
    viz_node = Node(
        package="rockhound_ros",
        executable="viz_node",
        name="viz_node",
        output="screen",
    )
    rviz_config = os.path.join(get_package_share_directory("rockhound_ros"), "rviz", "rockhound.rviz")
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(LaunchConfiguration("start_rviz")),
    )

    return LaunchDescription([
        sun_elevation_deg, startracker_interval_s, dt_s, start_rviz,
        speed_m_s, heading_amplitude_deg, heading_period_s,
        sensors_node, ekf_node, viz_node, rviz,
    ])
