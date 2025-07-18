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
// Member 'mask_data'
// Member 'scores'
// Member 'classes'
#include "rosidl_runtime_c/primitives_sequence.h"
// Member 'image_source_monotonic_capture_time'
// Member 'processing_node_monotonic_entry_time'
// Member 'processing_node_inference_start_time'
// Member 'processing_node_inference_end_time'
// Member 'processing_node_monotonic_publish_time'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in msg/InstanceSegmentationInfo in the package yolo_custom_interfaces.
typedef struct yolo_custom_interfaces__msg__InstanceSegmentationInfo
{
  std_msgs__msg__Header header;
  uint16_t mask_width;
  uint16_t mask_height;
  rosidl_runtime_c__uint8__Sequence mask_data;
  rosidl_runtime_c__float__Sequence scores;
  rosidl_runtime_c__uint8__Sequence classes;
  builtin_interfaces__msg__Time image_source_monotonic_capture_time;
  builtin_interfaces__msg__Time processing_node_monotonic_entry_time;
  builtin_interfaces__msg__Time processing_node_inference_start_time;
  builtin_interfaces__msg__Time processing_node_inference_end_time;
  builtin_interfaces__msg__Time processing_node_monotonic_publish_time;
  uint64_t packet_sequence_number;
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
