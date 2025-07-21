cmake_minimum_required(VERSION 3.5)
project(sensor_sync_pkg)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(sync_node src/sync_node.cpp)
ament_target_dependencies(sync_node rclcpp sensor_msgs)

install(TARGETS sync_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY 
launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()