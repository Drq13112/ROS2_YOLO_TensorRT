from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    ld = LaunchDescription()

    camera_front_tf = Node(package = "tf2_ros", 
                       executable = "static_transform_publisher",
                       arguments = ["0.161445", "0.0174674", "-0.257808", "0.018211", "0.145767", "-0.00378974", "rubyplus", "camera_front"])

    camera_front_left_tf = Node(package = "tf2_ros", 
                       executable = "static_transform_publisher",
                       arguments = ["0.171841", "0.142724", "-0.26274", "0.693539", "0.120547", "0.0217674", "rubyplus", "camera_front_left"])

    camera_front_right_tf = Node(package = "tf2_ros", 
                       executable = "static_transform_publisher",
                       arguments = ["0.142016", "-0.131321", "-0.241215", "-0.551337", "0.169142", "-0.0122094", "rubyplus", "camera_front_right"])

    ruby_plus_tf = Node(package = "tf2_ros", 
                       executable = "static_transform_publisher",
                       arguments = ["1.2", "0", "1.9", "0", "0", "0", "odom", "rubyplus"])

    helios_left_tf = Node(package = "tf2_ros", 
                   executable = "static_transform_publisher",
                   arguments = ["0.81", "0.55", "1.7", "1.6441", "0.148353", "0.006108652", "odom", "helios_left"])

    helios_right_tf = Node(package = "tf2_ros", 
               executable = "static_transform_publisher",
               arguments = ["0.85", "-0.55", "1.68", "-1.5132", "0.1396263", "0.008726646", "odom", "helios_right"])

    ld.add_action(ruby_plus_tf)
    ld.add_action(helios_left_tf)
    ld.add_action(helios_right_tf)
    ld.add_action(camera_front_tf)
    ld.add_action(camera_front_left_tf)
    ld.add_action(camera_front_right_tf)

    return ld