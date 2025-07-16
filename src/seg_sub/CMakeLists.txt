cmake_minimum_required(VERSION 3.8)
project(seg_sub)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(yolo_custom_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)


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


#-------------------- Yolo Sub Node --------------------

add_executable(seg_sub_yolo_node src/seg_sub_yolo.cpp)
ament_target_dependencies(seg_sub_yolo_node OpenCV cv_bridge rclcpp yolo_custom_interfaces)

target_link_libraries(seg_sub_yolo_node
    chrono
    ${TRT_LIBS}     # Librerías de TensorRT (si son necesarias directamente por el nodo o por los headers de deploy)
)
#-------------------------------------------------------------------------------   

add_executable(seg_sub_pid_node src/seg_sub_pid.cpp)
ament_target_dependencies(seg_sub_pid_node OpenCV cv_bridge rclcpp yolo_custom_interfaces)

target_link_libraries(seg_sub_pid_node
    chrono
    ${TRT_LIBS}     # Librerías de TensorRT (si son necesarias directamente por el nodo o por los headers de deploy)
)

#----------------- PidNet Sub Node -----------------
install(TARGETS
  chrono
  seg_sub_yolo_node
  seg_sub_pid_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  include
  DESTINATION share/${PROJECT_NAME}
)

ament_package()