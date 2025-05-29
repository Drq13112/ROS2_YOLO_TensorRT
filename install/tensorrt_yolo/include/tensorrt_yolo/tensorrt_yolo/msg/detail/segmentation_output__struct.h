// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tensorrt_yolo:msg/SegmentationOutput.idl
// generated code does not contain a copyright notice

#ifndef TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__STRUCT_H_
#define TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__STRUCT_H_

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
// Member 'class_id_map'
// Member 'instance_id_map'
// Member 'instance_confidences'
// Member 'instance_class_ids'
// Member 'detected_instance_ids'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/SegmentationOutput in the package tensorrt_yolo.
typedef struct tensorrt_yolo__msg__SegmentationOutput
{
  std_msgs__msg__Header header;
  /// Original height of the processed frame
  uint32_t image_height;
  /// Original width of the processed frame
  uint32_t image_width;
  /// Pixel-wise class ID map (flat array, row-major)
  /// Each value is the class ID for the corresponding pixel.
  /// 0 can be used for background or pixels with no assigned class.
  rosidl_runtime_c__int32__Sequence class_id_map;
  /// Pixel-wise instance ID map (flat array, row-major)
  /// Each value is a unique ID for the detected instance covering that pixel.
  /// 0 can be used for background or pixels with no assigned instance.
  rosidl_runtime_c__int32__Sequence instance_id_map;
  /// Confidence scores for each detected instance
  rosidl_runtime_c__float__Sequence instance_confidences;
  /// Class IDs for each detected instance (corresponds to instance_confidences)
  rosidl_runtime_c__int32__Sequence instance_class_ids;
  /// Unique instance IDs assigned (corresponds to instance_confidences)
  /// These are the IDs used in instance_id_map
  rosidl_runtime_c__int32__Sequence detected_instance_ids;
} tensorrt_yolo__msg__SegmentationOutput;

// Struct for a sequence of tensorrt_yolo__msg__SegmentationOutput.
typedef struct tensorrt_yolo__msg__SegmentationOutput__Sequence
{
  tensorrt_yolo__msg__SegmentationOutput * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tensorrt_yolo__msg__SegmentationOutput__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TENSORRT_YOLO__MSG__DETAIL__SEGMENTATION_OUTPUT__STRUCT_H_
