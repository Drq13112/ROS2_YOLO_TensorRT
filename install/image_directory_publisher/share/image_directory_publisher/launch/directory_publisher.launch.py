import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('image_directory_publisher'),
        'config',
        'config.yaml'
    )

    container = ComposableNodeContainer(
        name='image_directory_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='image_directory_publisher',
                plugin='image_directory_publisher::DirectoryPublisherNode',
                name='directory_publisher',
                # Se carga el archivo de parámetros para incluir la configuración de transporte y calidad JPEG.
                parameters=[config_file]
            ),
        ],
        output='screen',
    )

    return LaunchDescription([container])