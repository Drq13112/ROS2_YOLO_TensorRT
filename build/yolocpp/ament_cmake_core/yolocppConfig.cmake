# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_yolocpp_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED yolocpp_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(yolocpp_FOUND FALSE)
  elseif(NOT yolocpp_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(yolocpp_FOUND FALSE)
  endif()
  return()
endif()
set(_yolocpp_CONFIG_INCLUDED TRUE)

# output package information
if(NOT yolocpp_FIND_QUIETLY)
  message(STATUS "Found yolocpp: 0.0.0 (${yolocpp_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'yolocpp' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${yolocpp_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(yolocpp_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${yolocpp_DIR}/${_extra}")
endforeach()
