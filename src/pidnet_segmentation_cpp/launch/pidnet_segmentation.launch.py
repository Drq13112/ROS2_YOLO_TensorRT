import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    
    # --- Obtener la ruta del paquete ---
    pkg_dir = get_package_share_directory('pidnet_segmentation_cpp')
    
    # --- Declarar los argumentos del lanzamiento ---
    
    # Argumento para la ruta del motor TensorRT
    engine_path_arg = DeclareLaunchArgument(
        'engine_path',
        default_value=os.path.join(pkg_dir, 'models', 'pidnet_s_1200x1920.trt'),
        description='Path to the TensorRT engine file.'
    )
    
    # Argumento para la ruta de salida de los videos
    output_video_path_arg = DeclareLaunchArgument(
        'output_video_path',
        default_value='/home/david/ros_videos/pidnet_out',
        description='Directory to save output videos.'
    )

    # Argumento para habilitar/deshabilitar la medición de tiempos
    measure_times_arg = DeclareLaunchArgument(
        'measure_times',
        default_value='true',
        description='Enable/disable performance measurement.'
    )

    # Argumento para habilitar/deshabilitar la visualización en tiempo real
    realtime_display_arg = DeclareLaunchArgument(
        'realtime_display',
        default_value='false',
        description='Enable/disable realtime panoramic display.'
    )

    # --- Configuración del Nodo ---
    
    # El ejecutable 'pidnet_segmentation_node' lanza internamente los 3 nodos
    # y el nodo de parámetros. Solo necesitamos lanzarlo una vez.
    pidnet_node = Node(
        package='pidnet_segmentation_cpp',
        executable='pidnet_segmentation_node',
        name='pidnet_segmentation_manager', # Nombre del proceso principal
        output='screen',
        parameters=[{
            # Parámetros para el nodo 'pidnet_param_node' (leídos en main)
            'measure_times': LaunchConfiguration('measure_times'),
            'realtime_display': LaunchConfiguration('realtime_display'),
            'output_video_path': LaunchConfiguration('output_video_path'),
            'video_fps': 10.0,
            'video_width': 1920,
            'video_height': 1200,
            'image_transport_type': 'raw',
            'left_camera_topic': '/camera_front_left/image_raw',
            'front_camera_topic': '/camera_front/image_raw',
            'right_camera_topic': '/camera_front_right/image_raw',
            
            # Parámetros para los nodos 'pidnet_node_left/front/right'
            'engine_path': LaunchConfiguration('engine_path'),
            'overlay_alpha': 0.4
        }]
    )

    return LaunchDescription([
        engine_path_arg,
        output_video_path_arg,
        measure_times_arg,
        realtime_display_arg,
        pidnet_node
    ])