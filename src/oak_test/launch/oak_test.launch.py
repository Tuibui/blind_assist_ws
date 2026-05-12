from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory("oak_test"), "config", "oak_test.yaml"
    )

    return LaunchDescription(
        [
            Node(
                package="oak_test",
                executable="oak_test_node",
                name="oak_test_node",
                output="screen",
                parameters=[config_path],
            )
        ]
    )
