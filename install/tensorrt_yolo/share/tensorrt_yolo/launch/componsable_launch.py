from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """Generate launch description with a single container for all nodes."""
    
    # Directorio base para las imágenes
    image_dir = "/home/david/Documents/Datasets/imagenes_luis/02"
    
    # CORREGIDO: El nombre del paquete que contiene el nodo YoloBatchNode
    # Probablemente es 'tensorrt_yolo' basado en la ruta del archivo. ¡Verifícalo!
    segmentation_node_package = 'tensorrt_yolo'

    publisher_config_file = os.path.join(
        get_package_share_directory('image_directory_publisher'),
        'config',
        'config.yaml'
    )

    # Configuración común para los nodos de segmentación
    video_output_path = "/home/david/ros_videos/segment_node_3P_out"
    engine_path = '/home/david/yolocpp_ws/src/TensorRT-YOLO/models/yolo11s-seg_02_b1_v2.engine'
    
    container = ComposableNodeContainer(
            name='yolo_pipeline_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                # --- Nodo Publicador ---
                ComposableNode(
                    package='image_directory_publisher',
                    plugin='image_directory_publisher::DirectoryPublisherNode',
                    name='directory_publisher',
                    parameters=[publisher_config_file, # Carga los parámetros del archivo
                        { # Estos valores sobreescribirán los del archivo si existen
                            'image_directory': image_dir,
                            'publish_rate': 10.0,
                            'loop_playback': True,
                            'left_image_pattern': "front_left*.jpg",
                            'front_image_pattern': "front_*.jpg",
                            'right_image_pattern': "front_right*.jpg",
                            'frame_id_left': 'camera_left_link',
                            'frame_id_front': 'camera_front_link',
                            'frame_id_right': 'camera_right_link',
                        }]),
                
                # --- Nodos de Segmentación (Subscriptores) ---
                ComposableNode(
                    package=segmentation_node_package, 
                    plugin='YoloBatchNode',
                    name='yolo_segment_node_left',
                    parameters=[{
                        'engine_path': engine_path,
                        'image_topic': '/camera_front_left/image_raw',
                        'output_topic_suffix': 'left',
                        'video_path': video_output_path,
                        'image_transport': 'compressed', # Configurable por nodo
                        'input_width': 640,
                        'input_height': 416,
                        'enable_mask_video': False,
                    }]),
                ComposableNode(
                    package=segmentation_node_package,
                    plugin='YoloBatchNode',
                    name='yolo_segment_node_front',
                    parameters=[{
                        'engine_path': engine_path,
                        'image_topic': '/camera_front/image_raw',
                        'output_topic_suffix': 'front',
                        'video_path': video_output_path,
                        'image_transport': 'compressed', # Configurable por nodo
                        'input_width': 640,
                        'input_height': 416,
                        'enable_mask_video': False,
                    }]),
                ComposableNode(
                    package=segmentation_node_package,
                    plugin='YoloBatchNode',
                    name='yolo_segment_node_right',
                    parameters=[{
                        'engine_path': engine_path,
                        'image_topic': '/camera_front_right/image_raw',
                        'output_topic_suffix': 'right',
                        'video_path': video_output_path,
                        'image_transport': 'compressed', # Configurable por nodo
                        'input_width': 640,
                        'input_height': 416,
                        'enable_mask_video': False,
                    }]),
            ],
            output='screen',
    )

    return LaunchDescription([container])