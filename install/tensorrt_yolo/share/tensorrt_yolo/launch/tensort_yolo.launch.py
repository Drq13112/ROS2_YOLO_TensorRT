from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

pkg_share = get_package_share_directory('tensorrt_yolo')
model = os.path.join(pkg_share, 'models', 'yolo11s-seg_03_b3_v2.engine')

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tensorrt_yolo',
            executable='segment_node',
            name='segment_node',
            parameters=[
                {
            'engine_path': model,
            'rescale_factor': 0.33,
            'input_width': 640,
            'input_height': 416,
            'image_topic_1': '/camera_front_left/image_raw',
            'image_topic_2': '/camera_front/image_raw',
            'image_topic_3': '/camera_front_right/image_raw',
            'use_pinned_input_memory': True,
            'enable_mask_video_writing': True,
            'enable_inferred_video_writing': True,
            'video_output_path': '/home/david/ros_videos',
            'inferred_video_filename': 'inferred_video.avi',
            'mask_video_filename': 'mask_video.avi',
            'video_fps': 10.0,
            'video_frame_width': 1920*3,
            'video_frame_height': 1200,
            'image_transport': 'raw'
            }
            ],
        ),
    ])
