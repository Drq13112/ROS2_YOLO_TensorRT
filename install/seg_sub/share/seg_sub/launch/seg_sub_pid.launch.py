#!/usr/bin/env python3
import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='seg_sub',
            executable='seg_sub_pid_node',
            name='seg_sub_pid_node',
            output='screen'
        )
    ])