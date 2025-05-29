from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

pkg_share = get_package_share_directory('tensorrt_yolo')
model = os.path.join(pkg_share, 'models', 'yolo11s-seg.engine')

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tensorrt_yolo',
            executable='segment_node',
            name='segment_node',
            parameters=[
                {
                    'engine_path': model,
                    'rescale_factor': 1.0,
                    'input_width': 640,
                    'input_height': 418,
                    'image_topic_1': '/left/image_raw',
                    'image_topic_2': '/front/image_raw',
                    'image_topic_3': '/right/image_raw',
                    'use_pinned_input_memory': True,
                }
            ],
        ),
    ])
