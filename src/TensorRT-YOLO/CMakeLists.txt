cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.9/bin/nvcc)
project(tensorrt_yolo LANGUAGES C CXX CUDA)

# Encuentra paquetes de ROS 2 y dependencias estándar
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED) # If you plan to use sensor_msgs/Image directly
find_package(cv_bridge REQUIRED) # For OpenCV image conversion in ROS 2
find_package(yolo_custom_interfaces REQUIRED)
find_package(image_transport REQUIRED) # For image transport in ROS 2
find_package(image_transport_plugins REQUIRED) # For image transport plugins
find_package(rclcpp_components REQUIRED)
#-------------------------------------------------------------------------------

# Configuración estándar de C++ 
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Opciones de compilación 
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")

  add_compile_options(-Wall -Wextra -Wpedantic) # Habilita advertencias comunes
endif()

# # # Configuración de arquitecturas CUDA (del original)
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#     # Define las arquitecturas de GPU para las que se compilará el código CUDA.
#     set(CMAKE_CUDA_ARCHITECTURES "89;86;75;70;61")
# endif()

set(CMAKE_CUDA_ARCHITECTURES "89") # Para la GPU RTX 4060, para 3060 sería "86"


# Configuración de TensorRT 
# TRT_PATH se debe pasar al compilar, ej: -DTRT_PATH=/usr
set(TRT_PATH "/usr")
if(NOT TRT_PATH)
    message(FATAL_ERROR "TensorRT path (TRT_PATH) must be specified. Set it with -DTRT_PATH=/path/to/tensorrt during colcon build.")
else()
    message(STATUS "Using TRT_PATH: ${TRT_PATH}")
endif()

# Define los directorios de inclusión y librerías de TensorRT basados en TRT_PATH
set(TRT_INCLUDE_DIR "${TRT_PATH}/include")
set(TRT_LIB_DIR "${TRT_PATH}/lib")

# Define las librerías de TensorRT a enlazar 
# Asumiendo un entorno Linux, no MSVC
# Nombres genéricos de las librerías de TensorRT a enlazar
set(TRT_LIBS nvinfer nvinfer_plugin nvonnxparser)
message(STATUS "TensorRT libraries to link: ${TRT_LIBS}")

#-------------------------------------------------------------------------------
# FUNCIONES AUXILIARES (copiadas del CMakeLists original)
#-------------------------------------------------------------------------------

# set_target_compile_options_custom: Define opciones de compilación específicas
#      (optimizaciones para Release, flags de debug para Debug) para C++ y CUDA.
function(set_target_compile_options_custom target)
    if(MSVC)
        # ... (Configuración MSVC omitida por brevedad, asumiendo Linux para ROS 2) ...
    else() # GCC/Clang
        target_compile_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -march=native -DNDEBUG> # -flto=auto removido por simplicidad inicial
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g3 -fno-omit-frame-pointer -fno-inline>
        )
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g -G>
        )
        # target_link_options(${target} PRIVATE
        #     $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -flto=auto>
        #     $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g3>
        # )
    endif()
    target_compile_definitions(${target} PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
    )
endfunction()

# Función para configurar un target para CUDA y TensorRT
function(configure_cuda_trt_target_custom target)
    target_include_directories(${target} PRIVATE
        ${TRT_INCLUDE_DIR}
    )
    target_link_directories(${target} PRIVATE
        ${TRT_LIB_DIR}
    )
    target_link_libraries(${target} PRIVATE
        CUDA::cudart
        ${TRT_LIBS}
    )
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endfunction()

# Función para recolectar archivos fuente para la librería 'deploy'
# Modificada para tomar la ruta base de los fuentes de 'deploy'
function(add_deploy_lib_source_files target_name deploy_source_root_dir)
    target_include_directories(${target_name} PRIVATE
        ${deploy_source_root_dir} # Para includes relativos dentro de la lib deploy
        ${CMAKE_CURRENT_SOURCE_DIR}/include # Para "deploy/model.hpp" desde el nodo
    )
    file(GLOB_RECURSE CURRENT_DEPLOY_SOURCES CONFIGURE_DEPENDS
        "${deploy_source_root_dir}/core/*.cpp"
        "${deploy_source_root_dir}/utils/*.cpp"
        "${deploy_source_root_dir}/infer/*.cpp"
        "${deploy_source_root_dir}/infer/*.cu"
        "${deploy_source_root_dir}/model.cpp"
    )
    if(NOT CURRENT_DEPLOY_SOURCES)
        message(WARNING "No source files found for the '${target_name}' library under ${deploy_source_root_dir}. Check paths.")
    else()
        message(STATUS "Source files for '${target_name}': ${CURRENT_DEPLOY_SOURCES}")
    endif()
    target_sources(${target_name} PRIVATE ${CURRENT_DEPLOY_SOURCES})
endfunction()

#-------------------------------------------------------------------------------
# CONSTRUCCIÓN DE LA LIBRERÍA 'deploy'
#-------------------------------------------------------------------------------
# Esta es la librería principal que contiene la lógica de inferencia de YOLO.
#-------------------------------------------------------------------------------

set(DEPLOY_LIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/deploy")

if(NOT EXISTS ${DEPLOY_LIB_SRC_DIR})
    message(FATAL_ERROR "Directory for 'deploy' library source files not found: ${DEPLOY_LIB_SRC_DIR}. "
                        "This path should point to where model.cpp and the core/, infer/, utils/ subdirectories (containing .cpp/.cu files) are located.")
endif()

add_library(deploy SHARED) # Crea la librería 'deploy'
# La función add_deploy_lib_source_files usará el DEPLOY_LIB_SRC_DIR actualizado
add_deploy_lib_source_files(deploy ${DEPLOY_LIB_SRC_DIR}) # Añade sus archivos fuente
configure_cuda_trt_target_custom(deploy) # Configura CUDA/TRT para 'deploy'
set_target_compile_options_custom(deploy) # Aplica opciones de compilación

set_target_properties(deploy PROPERTIES CXX_VISIBILITY_PRESET "hidden")

#-------------------------------------------------------------------------------
# CONSTRUCCIÓN DE LA LIBRERÍA 'custom_plugins' (desde include/plugin/CMakeLists.txt)
#-------------------------------------------------------------------------------
set(PLUGIN_DIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include/plugin")

if(EXISTS "${PLUGIN_DIR_PATH}/CMakeLists.txt") # Solo si existe la carpeta y su CMake
    message(STATUS "Configuring custom_plugins from ${PLUGIN_DIR_PATH}")
    add_library(custom_plugins SHARED)

    # Añadir directorios de inclusión para custom_plugins
    target_include_directories(custom_plugins PRIVATE
        ${PLUGIN_DIR_PATH} # Para headers dentro de la carpeta plugin
    )

    # Definir los directorios de fuentes del plugin
    set(PLUGIN_SOURCE_SUBDIRS
        "${PLUGIN_DIR_PATH}/common"
        "${PLUGIN_DIR_PATH}/efficientRotatedNMSPlugin"
        "${PLUGIN_DIR_PATH}/efficientIdxNMSPlugin"
    )

    set(PLUGIN_SOURCES_LIST)
    foreach(PLUGIN_SUBDIR ${PLUGIN_SOURCE_SUBDIRS})
        if(EXISTS ${PLUGIN_SUBDIR})
            file(GLOB CPP_FILES "${PLUGIN_SUBDIR}/*.cpp")
            file(GLOB CU_FILES "${PLUGIN_SUBDIR}/*.cu")
            list(APPEND PLUGIN_SOURCES_LIST ${CPP_FILES} ${CU_FILES})
        else()
            message(WARNING "Plugin source subdirectory not found: ${PLUGIN_SUBDIR}")
        endif()
    endforeach()

    if(NOT PLUGIN_SOURCES_LIST)
        message(WARNING "No source files found for 'custom_plugins'.")
    else()
        message(STATUS "Source files for 'custom_plugins': ${PLUGIN_SOURCES_LIST}")
    endif()

    target_sources(custom_plugins PRIVATE ${PLUGIN_SOURCES_LIST})

    # Aplicar configuración CUDA/TensorRT y opciones de compilación
    configure_cuda_trt_target_custom(custom_plugins)
    set_target_compile_options_custom(custom_plugins)

    set_target_properties(custom_plugins PROPERTIES
        OUTPUT_NAME "custom_plugins"
        CXX_VISIBILITY_PRESET "hidden"
    )

    # Install the custom_plugins library
    install(TARGETS custom_plugins
        LIBRARY DESTINATION lib/${PROJECT_NAME}
        ARCHIVE DESTINATION lib/${PROJECT_NAME}
    )

else()
    message(STATUS "Plugin directory or its CMakeLists.txt not found at ${PLUGIN_DIR_PATH}, skipping custom_plugins build.")
endif()


#-------------------------------------------------------------------------------
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

set_target_compile_options_custom(chrono)

#-------------------------------------------------------------------------------
# CONSTRUCCIÓN DEL NODO ROS 2 'segment_node'
#-------------------------------------------------------------------------------
# add_executable(segment_node src/segment_node.cpp)


# # Dependencias de ROS 2 para el nodo
# ament_target_dependencies(segment_node rclcpp OpenCV rosidl_default_runtime std_msgs sensor_msgs cv_bridge yolo_custom_interfaces image_transport)

# # Directorios de inclusión para el nodo
# target_include_directories(segment_node PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include  # Donde están "deploy/model.hpp", etc.
#     ${TRT_INCLUDE_DIR}                   # Para los encabezados de TensorRT (NvInfer.h, etc.)
#     ${PLUGIN_DIR_PATH}                   # Si el nodo necesita directamente headers de plugin
#     ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp # Para los encabezados de mensajes generados
# )

# # Enlaza las librerías necesarias al nodo
# # `ament_target_dependencies` ya maneja rclcpp y OpenCV.
# # Usamos la signatura "plain" para target_link_libraries para ser compatible con ament_cmake.
# target_link_libraries(segment_node
#     deploy          # Tu librería 'deploy' compilada
#     chrono           
#     CUDA::cudart    # Para el runtime de CUDA (si segment_node.cpp lo usa directamente o vía headers de deploy)
#     ${TRT_LIBS}     # Librerías de TensorRT (si son necesarias directamente por el nodo o por los headers de deploy)
# )
# # Asegura que el enlazador encuentre las librerías de TensorRT
# target_link_directories(segment_node PRIVATE ${TRT_LIB_DIR})


# # Explicitly set RPATH for the segment_node target
# set_target_properties(segment_node PROPERTIES
#     INSTALL_RPATH "$ORIGIN:${CUDAToolkit_LIBRARY_DIRS}" # Look in the executable's own directory and CUDA lib dir
#     BUILD_RPATH "$ORIGIN:${CUDAToolkit_LIBRARY_DIRS}"   # Also useful for running from the build directory
# )



#-------------------------------------------------------------------------------
# INSTALACIÓN
#-------------------------------------------------------------------------------
# install(TARGETS
#     deploy          # Instala la librería libdeploy.so
#     chrono
#     segment_node    # Instala el ejecutable del nodo
#     DESTINATION lib/${PROJECT_NAME}
# )

# # Link custom plugins if the target exists
# if(TARGET custom_plugins)
#     target_link_libraries(segment_node PRIVATE custom_plugins)
#     message(STATUS "Linking segment_node with custom_plugins.")
# else()
#     message(WARNING "custom_plugins target does not exist. segment_node will not be linked against it.")
# endif()



#-------------------------------------------------------------------------------
# CONSTRUCCIÓN DEL NODO ROS 2 'segment_node_3P'
#-------------------------------------------------------------------------------
add_executable(segment_node_3P src/segment_node_3P.cpp)

# Dependencias de ROS 2 para el nodo
ament_target_dependencies(segment_node_3P rclcpp OpenCV rosidl_default_runtime std_msgs sensor_msgs cv_bridge yolo_custom_interfaces image_transport)

# Directorios de inclusión para el nodo
target_include_directories(segment_node_3P PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # Donde están "deploy/model.hpp", etc.
    ${TRT_INCLUDE_DIR}                   # Para los encabezados de TensorRT (NvInfer.h, etc.)
    ${PLUGIN_DIR_PATH}                   # Si el nodo necesita directamente headers de plugin
    ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp # Para los encabezados de mensajes generados
)

# Enlaza las librerías necesarias al nodo
# `ament_target_dependencies` ya maneja rclcpp y OpenCV.
# Usamos la signatura "plain" para target_link_libraries para ser compatible con ament_cmake.
target_link_libraries(segment_node_3P
    deploy          # Tu librería 'deploy' compilada
    chrono
    CUDA::cudart    # Para el runtime de CUDA (si segment_node.cpp lo usa directamente o vía headers de deploy)
    ${TRT_LIBS}     # Librerías de TensorRT (si son necesarias directamente por el nodo o por los headers de deploy)
)
# Asegura que el enlazador encuentre las librerías de TensorRT
target_link_directories(segment_node_3P PRIVATE ${TRT_LIB_DIR})


# Explicitly set RPATH for the segment_node target
set_target_properties(segment_node_3P PROPERTIES
    INSTALL_RPATH "$ORIGIN:${CUDAToolkit_LIBRARY_DIRS}" # Look in the executable's own directory and CUDA lib dir
    BUILD_RPATH "$ORIGIN:${CUDAToolkit_LIBRARY_DIRS}"   # Also useful for running from the build directory
)

#-------------------------------------------------------------------------------
# INSTALACIÓN del nodo 'segment_node_3P'
#-------------------------------------------------------------------------------
install(TARGETS
    deploy          # Instala la librería libdeploy.so
    chrono
    segment_node_3P # Instala el ejecutable del nodo
    DESTINATION lib/${PROJECT_NAME}
)

# Link custom plugins if the target exists
if(TARGET custom_plugins)
    target_link_libraries(segment_node_3P PRIVATE custom_plugins)
    message(STATUS "Linking segment_node_3P with custom_plugins.")
else()
    message(WARNING "custom_plugins target does not exist. segment_node_3P will not be linked against it.")
endif()



# -------------------------------------------------------------------------------
# CONSTRUCCIÓN DEL NODO ROS 2 'segment_node_3P_composable_component'
# -------------------------------------------------------------------------------
# add_library(segment_node_3P_composable_component SHARED
#   src/segment_node_3P_composable.cpp
# )
# ament_target_dependencies(segment_node_3P_composable_component rclcpp rclcpp_components OpenCV rosidl_default_runtime std_msgs sensor_msgs cv_bridge yolo_custom_interfaces image_transport)
# target_include_directories(segment_node_3P_composable_component PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     ${TRT_INCLUDE_DIR}
#     ${PLUGIN_DIR_PATH}
#     ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp
# )
# target_link_libraries(segment_node_3P_composable_component
#     deploy
#     chrono
#     CUDA::cudart
#     ${TRT_LIBS}
# )
# target_link_directories(segment_node_3P_composable_component PRIVATE ${TRT_LIB_DIR})
# rclcpp_components_register_nodes(segment_node_3P_composable_component "YoloBatchNode")

# # Esto le dice al nodo componible que busque sus librerías dependientes
# # (como libdeploy.so) en su propio directorio de instalación.
# set_target_properties(segment_node_3P_composable_component PROPERTIES
#     INSTALL_RPATH "$ORIGIN:${CUDAToolkit_LIBRARY_DIRS}"
# )

# if(TARGET custom_plugins)
#     target_link_libraries(segment_node_3P_composable_component PRIVATE custom_plugins)
# endif()

# -------------------------------------------------------------------------------
# install(TARGETS
#     deploy
#     chrono
#     segment_node_3P_composable_component
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
#     RUNTIME DESTINATION bin
# )

# install(TARGETS
#     segment_node
#     segment_node_3P
#     DESTINATION lib/${PROJECT_NAME}
# )

# if(TARGET custom_plugins)
#     install(TARGETS custom_plugins
#         ARCHIVE DESTINATION lib
#         LIBRARY DESTINATION lib
#         RUNTIME DESTINATION bin
#     )
# endif()

# -------------------------------------------------------------------------------
# FIN DEL BLOQUE AÑADIDO
# -------------------------------------------------------------------------------

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/deploy/
    DESTINATION include/deploy
)


install(
    DIRECTORY ${PLUGIN_DIR_PATH}/ 
    DESTINATION include/plugin
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" 
)


install (DIRECTORY 
  launch
  include
  models
  data
  config
  DESTINATION share/${PROJECT_NAME}
)

#-------------------------------------------------------------------------------
# FINALIZACIÓN DEL PAQUETE ROS 2
#-------------------------------------------------------------------------------
ament_package()