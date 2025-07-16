# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_pylon_instant_camera_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED pylon_instant_camera_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(pylon_instant_camera_FOUND FALSE)
  elseif(NOT pylon_instant_camera_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(pylon_instant_camera_FOUND FALSE)
  endif()
  return()
endif()
set(_pylon_instant_camera_CONFIG_INCLUDED TRUE)

# output package information
if(NOT pylon_instant_camera_FIND_QUIETLY)
  message(STATUS "Found pylon_instant_camera: 3.0.0 (${pylon_instant_camera_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'pylon_instant_camera' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${pylon_instant_camera_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(pylon_instant_camera_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${pylon_instant_camera_DIR}/${_extra}")
endforeach()
