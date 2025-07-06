cmake_minimum_required(VERSION 3.8)
project(ProyectCloud2image)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(yolo_custom_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
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


# CONSTRUCCIÓN DE LA LIBRERÍA 'chrono'
#-------------------------------------------------------------------------------
set(CHRONO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/chrono")
set(CHRONO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include") # This is for targets that link chrono

if(NOT EXISTS "${CHRONO_SRC_DIR}/ChronoTimer.cpp")
    message(FATAL_ERROR "ChronoTimer.cpp not found in ${CHRONO_SRC_DIR}. Please check the path.")
endif()
# Corrected path for the hpp existence check
if(NOT EXISTS "${CHRONO_SRC_DIR}/ChronoTimer.hpp") 
    message(FATAL_ERROR "ChronoTimer.hpp not found in ${CHRONO_SRC_DIR}. Please check the path.")
endif()

add_library(chrono SHARED
    "${CHRONO_SRC_DIR}/ChronoTimer.cpp"
)

target_include_directories(chrono PUBLIC
    $<BUILD_INTERFACE:${CHRONO_INCLUDE_DIR}> # Allows #include "chrono/ChronoTimer.hpp"
    $<INSTALL_INTERFACE:include> # Corresponds to the general include install directory
)

#-------------------------------------------------------------------------------

add_executable(projection_node src/projection_node.cpp)
ament_target_dependencies(projection_node OpenCV cv_bridge rclcpp yolo_custom_interfaces CUDA CUDAToolkit sensor_msgs tf2_ros
                          pcl_conversions pcl_ros PCL tf2_sensor_msgs nav_msgs rosbag2_cpp grid_map_msgs grid_map_ros)


target_link_libraries(projection_node
    chrono
    ${CUDA_LIBRARIES} # Librerías de CUDA (si son necesarias directamente por el nodo o por los headers de deploy)
    ${TRT_LIBS}     # Librerías de TensorRT (si son necesarias directamente por el nodo o por los headers de deploy)
)
#-------------------------------------------------------------------------------   

install(TARGETS
  chrono
  projection_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  include
  DESTINATION share/${PROJECT_NAME}
)

ament_package()