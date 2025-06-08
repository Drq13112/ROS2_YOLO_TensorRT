import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

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
        ntrip_launch,
        gps_launch,
        gps_heading_launch,
    ])
