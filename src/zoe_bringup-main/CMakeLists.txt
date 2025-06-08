cmake_minimum_required(VERSION 3.5)
project(zoe_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY rviz DESTINATION share/${PROJECT_NAME})

ament_package()
