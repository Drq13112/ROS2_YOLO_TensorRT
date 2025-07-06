cmake_minimum_required(VERSION 3.8)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.9/bin/nvcc)
project(perception_obstacles CUDA C CXX)

list(APPEND CMAKE_PREFIX_PATH /usr/local/cuda-12.9)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(perception_pcl REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
set(CMAKE_CUDA_ARCHITECTURES 89)
# endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=20012") # quita una mierda de warning


include_directories(include) # Include "propio"
include_directories(${PCL_INCLUDE_DIRS}) # Include del paquete PCL
link_directories(${PCL_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})


# Set the CONFIG_DIR variable to the relative config folder path
set(CONFIG_DIR "${CMAKE_SOURCE_DIR}/config")
add_definitions(-DCONFIG_DIR="${CONFIG_DIR}")

set(MAP_DIR "${CMAKE_SOURCE_DIR}/symbolic_link_maps")
add_definitions(-DMAP_DIR="${MAP_DIR}")

add_definitions(-DDEBUG_DOG=0)

# Get all .cpp files in the src folder and its subdirectories
file(GLOB_RECURSE SRC_FILES_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE SRC_FILES_CU "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
add_executable(perception_obstacles  ${SRC_FILES_CPP} ${SRC_FILES_CU})


##### Complete node

target_include_directories(perception_obstacles PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(perception_obstacles PUBLIC c_std_99 cxx_std_17)

# Añadir dependencias al ejecutable (podriamos definir varios nodos en un mismo paquete)
ament_target_dependencies(perception_obstacles rclcpp CUDA CUDAToolkit sensor_msgs tf2 
                          pcl_conversions pcl_ros PCL tf2_sensor_msgs nav_msgs rosbag2_cpp grid_map_msgs grid_map_ros)

target_link_libraries(perception_obstacles ${CUDA_LIBRARIES})

##### TARGETS
install(TARGETS 
  # pointcloud_segmentation
  perception_obstacles 
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY
  # launch
  # config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()


# para compilar: colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --parallel-workers 1




