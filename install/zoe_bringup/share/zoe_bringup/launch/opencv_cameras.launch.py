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

    camera_front_info_config_file = os.path.join(
        get_package_share_directory('zoe_bringup'),
        'config',
        'cameras',
        'camera_front_calibration.yaml'
    )
    camera_front_left_info_config_file = os.path.join(
        get_package_share_directory('zoe_bringup'),
        'config',
        'cameras',
        'camera_front_left_calibration.yaml'
    )
    camera_front_right_info_config_file = os.path.join(
        get_package_share_directory('zoe_bringup'),
        'config',
        'cameras',
        'camera_front_right_calibration.yaml'
    )
    namespace = 'cameras'

    image_processing = ComposableNodeContainer(
            name = 'container',
            namespace = 'pylon_camera',
            package = 'rclcpp_components',
            executable = 'component_container',
            composable_node_descriptions = [
                ComposableNode(
                    name = 'pylon_camera',
                    namespace = namespace,
                    package = 'pylon_instant_camera',
                    plugin = 'pylon_instant_camera::PylonCameraNode',
                    parameters = [{
                        'undistort': True,
                        'save_images': False,
                        'save_image_path': '/home/jlhv/Desktop/test/',

                        'grab_timeout':1000.0,

                        'camera_front_frame_id': 'camera_front',
                        'camera_front_info_yaml': camera_front_info_config_file,
                        'camera_front_user_defined_name': 'front',
                        'camera_front_ip_address': '192.168.0.120',
                        'camera_front_serial_number': 40488011, 
                        'camera_front_settings_pfs': default_config_file,

                        'camera_front_left_frame_id': 'camera_front_left',
                        'camera_front_left_info_yaml': camera_front_left_info_config_file,
                        'camera_front_left_user_defined_name': 'front_left',
                        'camera_front_left_ip_address': '192.168.0.122',
                        'camera_front_left_serial_number': 40488014, 
                        'camera_front_left_settings_pfs': default_config_file,

                        'camera_front_right_frame_id': 'camera_front_right',
                        'camera_front_right_info_yaml': camera_front_right_info_config_file,
                        'camera_front_right_user_defined_name': 'front_right',
                        'camera_front_right_ip_address': '192.168.0.121',
                        'camera_front_right_serial_number': 40488002, 
                        'camera_front_right_settings_pfs': default_config_file,
                        }]
                ),
            ]
    )

    return launch.LaunchDescription([image_processing])