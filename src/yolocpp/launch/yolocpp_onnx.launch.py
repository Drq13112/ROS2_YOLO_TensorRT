from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
pkg_share = get_package_share_directory('yolocpp')
model = os.path.join(pkg_share, 'models', 'yolo11m-seg.onnx')
labels = os.path.join(pkg_share, 'models', 'coco.names')
video_path = os.path.join(pkg_share, 'data', 'panoramic_02.avi')


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolocpp',
            executable='yolocpp',
            name='yolocpp_node',
            parameters=[
                {'model_path': model,
                'labels_path': labels,
                'video_path': video_path,
                'use_gpu': True,
                'output_dir': '/home/david/seg_out'},
            ],
        ),
    ])