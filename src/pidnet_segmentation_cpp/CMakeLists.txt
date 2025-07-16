cmake_minimum_required(VERSION 3.8)
# Enable CUDA language in the project declaration
project(pidnet_segmentation_cpp LANGUAGES CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenMP)
find_package(yolo_custom_interfaces REQUIRED)

# Manually set CUDA_TOOLKIT_ROOT_DIR if CMAKE_CUDA_COMPILER is defined
if(DEFINED CMAKE_CUDA_COMPILER)
  get_filename_component(CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  get_filename_component(CUDA_TOOLKIT_ROOT_DIR_FROM_COMPILER "${CUDA_BIN_DIR}" DIRECTORY)
  # Override the variable used by find_library
  set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR_FROM_COMPILER} CACHE PATH "Path to CUDA Toolkit" FORCE)
  message(STATUS "Inferred CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")
endif()


# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/x86_64-linux-gnu)
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS /usr/lib/x86_64-linux-gnu)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS /usr/lib/x86_64-linux-gnu)

# Find NPP (NVIDIA Performance Primitives) libraries
# We now know the exact path from the 'dpkg -L' command.
set(CUDA_NPP_LIB_PATH ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)
message(STATUS "Searching for NPP libraries in: ${CUDA_NPP_LIB_PATH}")

find_library(NPP_LIBRARY_IG nppig
  HINTS ${CUDA_NPP_LIB_PATH}
  REQUIRED)
find_library(NPP_LIBRARY_C nppc
  HINTS ${CUDA_NPP_LIB_PATH}
  REQUIRED)


# ---- Create CUDA kernels library ----
add_library(cuda_kernels STATIC
  include/pidnet_segmentation_cpp/cuda_kernels.cu
)

# Set properties for CUDA compilation
set_target_properties(cuda_kernels PROPERTIES
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
  POSITION_INDEPENDENT_CODE ON
)

# Add includes for CUDA kernels
target_include_directories(cuda_kernels PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${CUDA_INCLUDE_DIRS}
)

# Add executable
add_executable(pidnet_segmentation_node
  src/pidnet_segmentation_node.cpp
  include/pidnet_segmentation_cpp/tensorrt_inference.cpp
  include/pidnet_segmentation_cpp/ChronoTimer.cpp
  # include/pidnet_segmentation_cpp/cuda_kernels.cu 
)

# Link OpenMP if found
if(OPENMP_FOUND)
  # Remove PUBLIC to match the "plain" signature used by ament_target_dependencies
  target_link_libraries(pidnet_segmentation_node OpenMP::OpenMP_CXX)
endif()

target_include_directories(pidnet_segmentation_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Link ROS 2 libraries using ament_target_dependencies
ament_target_dependencies(pidnet_segmentation_node
  "rclcpp"
  "rclcpp_components"
  "sensor_msgs"
  "cv_bridge"
  "image_transport"
  "yolo_custom_interfaces"
)

# Link ALL other system libraries using target_link_libraries
target_link_libraries(pidnet_segmentation_node
  ${OpenCV_LIBS}
  nvinfer
  nvonnxparser
  ${CUDA_LIBRARIES}
  ${NPP_LIBRARY_C}
  ${NPP_LIBRARY_IG}
  cuda_kernels
)

# Install targets
install(TARGETS pidnet_segmentation_node
  cuda_kernels
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  models
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include
)

ament_package()