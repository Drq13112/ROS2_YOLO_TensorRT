import launch
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    mcap_config_file = os.path.join(
        get_package_share_directory('zoe_bringup'),
        'config',
        'mcap_writer_options.yaml'
    )


    return launch.LaunchDescription([
        launch.actions.ExecuteProcess(
            cmd=['ros2', 'bag', 'record', 
            '/pylon_camera_node/image_color',
            '/helios_left_points',
            '/helios_right_points',
            '/rubyplus_points',
            '/simpleRTK2BLite/ublox_gps_node/fix',
            '/simpleRTK2BLite/ublox_gps_node/fix_velocity',
            '/simpleRTK2BLite/nmea',
            '/simpleRTK2B/ublox_gps_node/fix',
            '/simpleRTK2B/ublox_gps_node/fix_velocity',
            '/simpleRTK2B/nmea',
            '/rtcm',
            '/simpleRTK2B/navrelposned',
            '-s', 'mcap',
            '--max-cache-size', '1048576000',
            '--storage-config-file', mcap_config_file],
            output='screen'
        )
    ])
