from setuptools import find_packages, setup

package_name = "rockhound_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/pseudo_sensors.launch.py"]),
        ("share/" + package_name + "/rviz", ["rviz/rockhound.rviz"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Luke Wilner",
    maintainer_email="lukewilner@gmail.com",
    description="ROS 2 pseudo-sensor nodes and navigation EKF wrapping the rockhound Python simulation library",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pseudo_sensors_node = rockhound_ros.pseudo_sensors_node:main",
            "ekf_node = rockhound_ros.ekf_node:main",
            "viz_node = rockhound_ros.viz_node:main",
        ],
    },
)
