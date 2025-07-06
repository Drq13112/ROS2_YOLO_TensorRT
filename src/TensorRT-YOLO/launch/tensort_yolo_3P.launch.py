from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('tensorrt_yolo')
    default_model_path = os.path.join(pkg_share, 'models', 'yolo11s-seg_02_b1_v2.engine')
    default_video_output_path = os.path.join(os.path.expanduser('~'), 'ros_videos', 'segment_node_3P_out')

    node_executable = 'segment_node_3P' 

    return LaunchDescription([
        Node(
            package='tensorrt_yolo',
            executable=node_executable,
            name='yolo_segmentation_processor',
            output='screen',
            parameters=[
                # --- Parámetros del modelo (comunes a los 3 nodos internos) ---
                {'engine_path': default_model_path},
                {'input_width': 960},
                {'input_height': 608},
                {'mask_encoding': 'mono8'}, # 'mono8' o 'mono16'
                {'use_pinned_input_memory': True},
                {'input_channels': 3},

                # --- Parámetros de configuración general (leídos en main) ---
                {'enable_inferred_video': False},
                {'enable_mask_video': False},
                {'measure_times': True},
                {'realtime_display': True},
                {'output_video_path': default_video_output_path},
                {'video_fps': 10.0},
                {'video_width': 1920},
                {'video_height': 1200},
                {'image_transport_type': 'raw'},

                # --- Topics de las cámaras ---
                {'left_camera_topic': '/camera_front_left/image_raw'},
                {'front_camera_topic': '/camera_front/image_raw'},
                {'right_camera_topic': '/camera_front_right/image_raw'},
            ],
        ),
    ])
