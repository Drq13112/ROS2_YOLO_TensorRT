// NOLINT: This file starts with a BOM since it contain non-ASCII characters
// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from yolo_custom_interfaces:msg/InstanceSegmentationInfo.idl
// generated code does not contain a copyright notice

#ifndef YOLO_CUSTOM_INTERFACES__MSG__DETAIL__INSTANCE_SEGMENTATION_INFO__STRUCT_H_
#define YOLO_CUSTOM_INTERFACES__MSG__DETAIL__INSTANCE_SEGMENTATION_INFO__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'mask'
#include "sensor_msgs/msg/detail/image__struct.h"
// Member 'scores'
// Member 'classes'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/InstanceSegmentationInfo in the package yolo_custom_interfaces.
typedef struct yolo_custom_interfaces__msg__InstanceSegmentationInfo
{
  /// Header con timestamp y frame_id
  std_msgs__msg__Header header;
  /// Máscara de segmentación. mask.data = instance_id
  /// instance_id = 0 para fondo, 1 para la primera instancia, etc.
  /// mask.encoding será "mono8" o "mono16"
  sensor_msgs__msg__Image mask;
  /// Array de scores, scores es el score de la instancia j+1
  rosidl_runtime_c__float__Sequence scores;
  /// Array de class_ids, classes es la clase de la instancia j+1
  /// El tamaño de scores y classes debe ser igual al número de instancias detectadas.
  rosidl_runtime_c__int32__Sequence classes;
} yolo_custom_interfaces__msg__InstanceSegmentationInfo;

// Struct for a sequence of yolo_custom_interfaces__msg__InstanceSegmentationInfo.
typedef struct yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence
{
  yolo_custom_interfaces__msg__InstanceSegmentationInfo * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} yolo_custom_interfaces__msg__InstanceSegmentationInfo__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // YOLO_CUSTOM_INTERFACES__MSG__DETAIL__INSTANCE_SEGMENTATION_INFO__STRUCT_H_
