import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import os

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    """Use composition for all image-processing nodes.
    Keeps overhead low since image data can – theoretically – reside in shared memory."""

    default_config_file = os.path.join(
        get_package_share_directory('zoe_bringup'),
        'config',
        'cameras',
        'a2A1920-51gcPRO.pfs'
    )

    camera_info_config_file = os.path.join(
        get_package_share_directory('zoe_bringup'),
        'config',
        'cameras',
        'camera_front_left_calibration.yaml'
    )
    namespace = 'camera_front_left'

    image_processing = ComposableNodeContainer(
            name = 'container',
            namespace = 'pylon_camera_front_left',
            package = 'rclcpp_components',
            executable = 'component_container',
            composable_node_descriptions = [
                ComposableNode(
                    name = 'pylon_camera',
                    namespace = namespace,
                    package = 'pylon_instant_camera',
                    plugin = 'pylon_instant_camera::PylonCameraNode',
                    parameters = [{
                        'camera_settings_pfs': default_config_file,
                        'camera_info_yaml': camera_info_config_file,
                        'user_defined_name': 'front_left',
                        'serial_number': 40488014
                        }]
                ),
                ComposableNode(
                    package='image_proc',
                    plugin='image_proc::DebayerNode',
                    name='debayer_node',
                    namespace = namespace
                )
            ]
    )

    return launch.LaunchDescription([image_processing])
