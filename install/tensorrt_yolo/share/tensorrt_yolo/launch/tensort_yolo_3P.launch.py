from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('tensorrt_yolo')
    # Asegúrate que el nombre del modelo aquí coincide con el que quieres usar.
    # El C++ tiene "yolo11m-seg.engine" como default si no se especifica este parámetro.
    default_model_path = os.path.join(pkg_share, 'models', 'yolo11s-seg_03_b1_v2.engine')

    node_executable = 'segment_node_3P' 

    return LaunchDescription([
        Node(
            package='tensorrt_yolo',
            executable=node_executable,
            name='yolo_segmentation_processor', # Un nombre general para el proceso que lanza los 3 nodos
            output='screen', # Para ver los logs de los 3 nodos
            parameters=[
                {
                    'engine_path': default_model_path,
                    'input_width': 640,
                    'input_height': 416, # El C++ usa 416 por defecto
                    'mask_encoding': 'mono8', # 'mono8' o 'mono16'
                    'use_pinned_input_memory': True,
                    'input_channels': 3,
                    'image_transport': 'compressed', # 'raw' o 'compressed'
                }
            ],
        ),
    ])
