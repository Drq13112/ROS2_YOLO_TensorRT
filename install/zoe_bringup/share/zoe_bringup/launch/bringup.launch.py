import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    rviz_config=get_package_share_directory('zoe_bringup')+'/rviz/zoe.rviz'

    camera_launch = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('zoe_bringup'), 'launch'),
         '/pylon_instant_camera.launch.py'])
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

    ntrip_launch = IncludeLaunchDescription( 
        PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('zoe_bringup'), 'launch'),
         '/zoe_ntrip_client.launch.py']))

    gps_launch = IncludeLaunchDescription( 
        PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('zoe_bringup'), 'launch'),
         '/zed_f9p.launch.py']))

    gps_heading_launch = IncludeLaunchDescription( 
        PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('zoe_bringup'), 'launch'),
         '/zed_f9p_lite.launch.py']))


    return LaunchDescription([
        camera_launch,
        top_lidar_launch,
        left_lidar_launch,
        right_lidar_launch,
        ntrip_launch,
        gps_launch,
        gps_heading_launch,
        Node(namespace='rviz2', package='rviz2', executable='rviz2', arguments=['-d',rviz_config])
    ])
