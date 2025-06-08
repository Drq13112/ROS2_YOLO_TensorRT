from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os

import ament_index_python.packages
import launch
import launch_ros.actions

def generate_launch_description():

    config_file = os.path.join(
        get_package_share_directory('zoe_bringup'),
        'config',
        'robosense_ruby_plus.yaml'
    )


    return LaunchDescription([
        Node(namespace='rslidar_sdk', 
            package='rslidar_sdk', 
            executable='rslidar_sdk_node',  
            parameters=[{'config_path': config_file}],
            output='screen'),
    ])
