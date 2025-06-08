cmake_minimum_required(VERSION 3.5)
project(image_directory_publisher)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rclcpp_components REQUIRED)

# Include directories
include_directories(include)


# Create the shared library for the component
add_library(directory_publisher_node SHARED src/directory_publisher_node.cpp) 


# Link libraries
ament_target_dependencies(directory_publisher_node
  rclcpp
  image_transport
  sensor_msgs
  cv_bridge
  rclcpp_components
)
# Register the component with rclcpp_components
rclcpp_components_register_nodes(directory_publisher_node "image_directory_publisher::DirectoryPublisherNode") 

# Install the library
install(TARGETS
  directory_publisher_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install (DIRECTORY 
  launch
  include
  config
  DESTINATION share/${PROJECT_NAME}
)


ament_package()