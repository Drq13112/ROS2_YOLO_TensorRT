from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

pkg_share = get_package_share_directory('tensorrt_yolo')
model = os.path.join(pkg_share, 'models', 'yolo11m-seg.engine')
labels = os.path.join(pkg_share, 'models', 'coco.names')
video_path = os.path.join(pkg_share, 'data', 'test1.jpg')
output_path = os.path.join(pkg_share, 'data')
label_path = os.path.join(pkg_share, 'models', 'labels.txt')


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tensorrt_yolo',
            executable='segment_node',
            name='segment_node',
            parameters=[
                {'engine_path': model,
                'labels_path': labels,
                'input_path': video_path,
                'output_path': output_path,
                'label_path': label_path,
                'use_gpu': True,
                'output_dir': '/home/david/seg_out'},
            ],
        ),
    ])
