from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    detector_config = os.path.join(
        get_package_share_directory("oak_detectors"), "config", "main_pipeline.yaml"
    )
    decision_audio_config = os.path.join(
        get_package_share_directory("oak_decision_audio"), "config", "oak_decision_audio.yaml"
    )

    return LaunchDescription(
        [
            Node(
                package="oak_detectors",
                executable="main_pipeline",
                name="main_pipeline_node",
                output="screen",
                parameters=[detector_config],
            ),
            Node(
                package="oak_decision_audio",
                executable="decision_audio_node",
                name="decision_audio_node",
                output="screen",
                parameters=[decision_audio_config],
            ),
        ]
    )
