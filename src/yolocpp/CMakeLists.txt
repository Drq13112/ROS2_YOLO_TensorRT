cmake_minimum_required(VERSION 3.8)
project(yolocpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(bboxes_ex_msgs REQUIRED)

add_executable(yolocpp src/yolocpp.cpp)

set(ONNX_LIB_DIR "/home/david/onnxruntime-linux-x64-gpu-1.22.0/lib")

target_include_directories(yolocpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  /home/david/onnxruntime-linux-x64-gpu-1.22.0/include
)

target_link_directories(yolocpp PUBLIC
  /home/david/onnxruntime-linux-x64-gpu-1.22.0/lib
)

target_link_libraries(yolocpp onnxruntime)

ament_target_dependencies(yolocpp
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  vision_msgs
  OpenCV
  image_transport
  cv_bridge
  bboxes_ex_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

endif()

install (DIRECTORY 
  launch
  include
  models
  data
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS yolocpp
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
