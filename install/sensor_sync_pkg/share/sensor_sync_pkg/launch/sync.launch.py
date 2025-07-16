from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sensor_sync_pkg',
            executable='sync_node',
            name='sensor_sync_node',
            output='screen',
            parameters=[{
                'lidar_topic': '/rubyplus_points',
                'camera_topic': '/camera_front/image_raw',
                'sync_threshold': 0.1  # seconds
            }]
        )
    ])