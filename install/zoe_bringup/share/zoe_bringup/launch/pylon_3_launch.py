import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    """Launch three Pylon cameras and their debayer nodes in a single ComposableNodeContainer."""

    bringup_pkg_share = get_package_share_directory('zoe_bringup')
    default_camera_settings_file = os.path.join(
        bringup_pkg_share,
        'config',
        'cameras',
        'a2A1920-51gcPRO.pfs'
    )

    # Configuration for Camera Front
    camera_info_front_file = os.path.join(
        bringup_pkg_share,
        'config',
        'cameras',
        'camera_front_calibration.yaml'
    )
    
    # Configuration for Camera Front Left
    camera_info_front_left_file = os.path.join(
        bringup_pkg_share,
        'config',
        'cameras',
        'camera_front_left_calibration.yaml'
    )

    # Configuration for Camera Front Right
    camera_info_front_right_file = os.path.join(
        bringup_pkg_share,
        'config',
        'cameras',
        'camera_front_right_calibration.yaml'
    )

    all_cameras_container = ComposableNodeContainer(
        name='all_cameras_container',
        namespace='',  # Global namespace for the container itself
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            # --- Front Camera ---
            ComposableNode(
                package='pylon_instant_camera',
                plugin='pylon_instant_camera::PylonCameraNode',
                name='pylon_camera_front',  # Unique node name
                namespace='camera_front',    # Namespace for topics
                parameters=[{
                    'camera_settings_pfs': default_camera_settings_file,
                    'camera_info_yaml': camera_info_front_file,
                    'user_defined_name': 'front',
                    'serial_number': 40488011
                }]
            ),
            ComposableNode(
                package='image_proc',
                plugin='image_proc::DebayerNode',
                name='debayer_node_front', # Unique node name
                namespace='camera_front'     # Same namespace for topic connections
            ),
            # --- Front Left Camera ---
            ComposableNode(
                package='pylon_instant_camera',
                plugin='pylon_instant_camera::PylonCameraNode',
                name='pylon_camera_front_left',
                namespace='camera_front_left',
                parameters=[{
                    'camera_settings_pfs': default_camera_settings_file,
                    'camera_info_yaml': camera_info_front_left_file,
                    'user_defined_name': 'front_left',
                    'serial_number': 40488014
                }]
            ),
            ComposableNode(
                package='image_proc',
                plugin='image_proc::DebayerNode',
                name='debayer_node_front_left',
                namespace='camera_front_left'
            ),
            # --- Front Right Camera ---
            ComposableNode(
                package='pylon_instant_camera',
                plugin='pylon_instant_camera::PylonCameraNode',
                name='pylon_camera_front_right',
                namespace='camera_front_right',
                parameters=[{
                    'camera_settings_pfs': default_camera_settings_file,
                    'camera_info_yaml': camera_info_front_right_file,
                    'user_defined_name': 'front_right',
                    'serial_number': 40488002
                }]
            ),
            ComposableNode(
                package='image_proc',
                plugin='image_proc::DebayerNode',
                name='debayer_node_front_right',
                namespace='camera_front_right'
            ),
        ],
        output='screen',
    )

    return launch.LaunchDescription([
        all_cameras_container
    ])