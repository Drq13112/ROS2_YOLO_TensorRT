cmake_minimum_required(VERSION 3.8)
project(yolo_custom_interfaces)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED) 
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/InstanceSegmentationInfo.msg"
  "msg/PidnetResult.msg"
  DEPENDENCIES std_msgs sensor_msgs builtin_interfaces
)


ament_export_dependencies(std_msgs sensor_msgs builtin_interfaces rosidl_default_runtime)
ament_package()