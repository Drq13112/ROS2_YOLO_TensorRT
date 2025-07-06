#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # Node for the LEFT camera and LiDAR
    left_node = Node(
        package='ProyectCloud2image',
        executable='projection_node',
        name='left_projection_node',
        output='screen',
        parameters=[
            {'camera_id': 'left'},
            {'image_topic': '/segmentation/left/instance_info'},
            {'lidar_topic': 'helios_left_points'},
            {'camera_frame_id': 'left_camera_optical_frame'} # ADJUST THIS FRAME ID
        ]
    )

    # Node for the FRONT camera and LiDAR
    front_node = Node(
        package='ProyectCloud2image',
        executable='projection_node',
        name='front_projection_node',
        output='screen',
        parameters=[
            {'camera_id': 'front'},
            {'image_topic': '/segmentation/front/instance_info'},
            {'lidar_topic': 'rubyplus_points'},
            {'camera_frame_id': 'front_camera_optical_frame'} # ADJUST THIS FRAME ID
        ]
    )


    # Node for the RIGHT camera and LiDAR
    right_node = Node(
        package='ProyectCloud2image',
        executable='projection_node',
        name='right_projection_node',
        output='screen',
        parameters=[
            {'camera_id': 'right'},
            {'image_topic': '/segmentation/right/instance_info'},
            {'lidar_topic': 'helios_right_points'},
            {'camera_frame_id': 'right_camera_optical_frame'} # ADJUST THIS FRAME ID
        ]
    )

    return LaunchDescription([
        left_node,
        front_node,
        right_node
    ])