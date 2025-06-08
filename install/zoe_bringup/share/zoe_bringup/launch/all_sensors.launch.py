import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    rviz_config=get_package_share_directory('zoe_bringup')+'/rviz/zoe.rviz'

    camera_front_launch = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('zoe_bringup'), 'launch'),
         '/camera_front.launch.py'])
      )

    camera_front_left_launch = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('zoe_bringup'), 'launch'),
         '/camera_front_left.launch.py'])
      )

    camera_front_right_launch = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('zoe_bringup'), 'launch'),
         '/camera_front_right.launch.py'])
      )

    camera_back_launch = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('zoe_bringup'), 'launch'),
         '/camera_back.launch.py'])
      )

    camera_back_left_launch = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('zoe_bringup'), 'launch'),
         '/camera_back_left.launch.py'])
      )

    camera_back_right_launch = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('zoe_bringup'), 'launch'),
         '/camera_back_right.launch.py'])
      )

    top_lidar_launch = IncludeLaunchDescription( 
        PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('zoe_bringup'), 'launch'),
         '/robosense_ruby_plus.launch.py']))

    left_lidar_launch = IncludeLaunchDescription( 
        PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('zoe_bringup'), 'launch'),
         '/robosense_helios_left.launch.py']))

    right_lidar_launch = IncludeLaunchDescription( 
        PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('zoe_bringup'), 'launch'),
         '/robosense_helios_right.launch.py']))

    static_transform_launch = IncludeLaunchDescription( 
        PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('zoe_bringup'), 'launch'),
         '/static_transform.launch.py']))


    return LaunchDescription([
        camera_front_launch,
        camera_front_left_launch,
        camera_front_right_launch,
        #camera_back_launch,
        #camera_back_left_launch,
        #camera_back_right_launch,
        top_lidar_launch,
        left_lidar_launch,
        right_lidar_launch,
        static_transform_launch,
        Node(namespace='rviz2', package='rviz2', executable='rviz2', arguments=['-d',rviz_config])
    ])
